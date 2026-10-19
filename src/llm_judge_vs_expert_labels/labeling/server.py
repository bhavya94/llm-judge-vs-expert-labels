"""The labeling page, served on localhost: one answer at a time, a Pass/Fail choice and a note, saved to the label file
as soon as the labeler submits. Two rounds: a first pass over the sample, and a blind relabel of part of it."""

import json
import random
import tomllib
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from llm_judge_vs_expert_labels.labeling.items import Item
from llm_judge_vs_expert_labels.labeling.labels import FailureMode, Label, LabelStore

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "labeling.toml"
TEMPLATE = Path(__file__).with_name("template.html")
MAX_BODY = 64 * 1024


class BadLabel(ValueError):
    pass


def render(data: dict, template: str) -> str:
    """Embed the data in the page. Every "<" is escaped, so text like "</script>" in an answer can't end the script."""
    if template.count("__DATA__") != 1:
        raise ValueError("the template needs exactly one __DATA__ placeholder")
    return template.replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))


@dataclass(frozen=True)
class Rounds:
    first: LabelStore
    relabel: LabelStore
    relabel_size: int
    relabel_seed: int
    port: int
    modes: tuple[FailureMode, ...]
    """Empty during open coding; filled after axial coding."""

    @classmethod
    def from_config(cls, path: Path = CONFIG) -> "Rounds":
        config = tomllib.loads(path.read_text())
        return cls(LabelStore((path.parent / config["first"]).resolve()),
                   LabelStore((path.parent / config["relabel"]).resolve()),
                   config["relabel_size"], config["relabel_seed"], config["port"],
                   tuple(FailureMode(m["name"], m["definition"]) for m in config.get("failure_modes", [])))

    def first_round(self, items: list[Item]) -> "LabelingServer":
        return LabelingServer(items, self.first, blind=False, modes=self.modes)

    def relabel_round(self, items: list[Item]) -> "LabelingServer":
        """A seeded subset in a new order. Blind: the page never shows the first label."""
        if missing := [item.id for item in items if item.id not in self.first.latest()]:
            raise ValueError(f"{len(missing)} of {len(items)} answers have no first label yet; finish the first pass")
        chosen = random.Random(self.relabel_seed).sample(items, self.relabel_size)
        return LabelingServer(chosen, self.relabel, blind=True, modes=self.modes)


class LabelingServer:
    def __init__(self, items: list[Item], labels: LabelStore, blind: bool, modes: tuple[FailureMode, ...] = ()):
        self.items = {item.id: item for item in items}
        self.order = [item.id for item in items]
        self.labels = labels
        self.blind = blind
        self.modes = modes

    def page(self) -> str:
        """Rendered on every request, so a reload shows the labels on disk."""
        latest = self.labels.latest()
        data = {"items": [self.items[i].to_row() for i in self.order], "blind": self.blind,
                "modes": [{"name": m.name, "definition": m.definition} for m in self.modes],
                "labels": {i: latest[i].to_row() for i in self.order if i in latest}}
        return render(data, TEMPLATE.read_text())

    def label(self, body: dict) -> Label:
        """Saves one label from the page's {"candidate_id", "verdict", "note", "tag"}. Once failure modes exist, every
        Fail needs one of them as its tag; a Pass never has a tag."""
        if not isinstance(body, dict) or (item := self.items.get(body.get("candidate_id"))) is None:
            raise BadLabel("unknown candidate_id")
        tag, names = str(body.get("tag") or ""), {m.name for m in self.modes}
        if body.get("verdict") == "fail" and names and tag not in names:
            raise BadLabel(f"a Fail needs one of the failure modes as its tag: {', '.join(sorted(names))}")
        if tag and (body.get("verdict") != "fail" or tag not in names):
            raise BadLabel(f"unexpected tag {tag!r}")
        try:
            label = Label.of(item, body.get("verdict"), str(body.get("note") or ""), tag)
        except ValueError as error:
            raise BadLabel(str(error)) from None
        self.labels.save(label)
        return label

    def server(self, port: int) -> HTTPServer:
        """One request at a time, so two saves never write the label file at once. Bound to 127.0.0.1 only. Labels
        must be posted as JSON, which a page on another site can't do without a preflight request that this server
        never answers."""
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/":
                    return self._send(HTTPStatus.NOT_FOUND, "text/plain", "not found")
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", app.page())

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if (self.path != "/label" or self.headers.get_content_type() != "application/json"
                        or not 0 < length <= MAX_BODY):
                    return self._send(HTTPStatus.BAD_REQUEST, "text/plain", "POST JSON to /label")
                try:
                    label = app.label(json.loads(self.rfile.read(length)))
                except (BadLabel, ValueError) as error:  # ValueError covers bad JSON and bytes that aren't UTF-8
                    return self._send(HTTPStatus.BAD_REQUEST, "text/plain", str(error))
                self._send(HTTPStatus.OK, "application/json", json.dumps(label.to_row()))

            def _send(self, status: HTTPStatus, content_type: str, body: str):
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                pass  # one line per request would bury the CLI's own output

        return HTTPServer(("127.0.0.1", port), Handler)
