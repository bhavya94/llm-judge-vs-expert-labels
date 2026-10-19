"""The `ljvel` command: run the experiments and print the results."""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ljvel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    args = parser.parse_args(argv)
    args.run(args)
