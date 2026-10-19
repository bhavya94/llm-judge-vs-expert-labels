# llm-judge-vs-expert-labels

How well an automated judge reproduces one expert's Pass/Fail labels on grounded-QA answers, how few labels it needs, and whether a small trained judge beats a prompted one.

## Evidence

## What is here

## Run it
```console
uv sync
uv run pytest
uv run ljvel --help
```

## Charts
`docs/views/index.html` draws the results above from the same seeded runs. Rebuild it with
`uv run python scripts/build_views.py`.

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py` enforces both.

## Limits
