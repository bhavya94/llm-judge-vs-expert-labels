from types import SimpleNamespace


class FakeClient:
    """Replies with the queued texts in order and records every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=50))
