"""``memex`` command-line entry point: dispatches to one module per command."""

from __future__ import annotations

import importlib
import sys

from dotenv import load_dotenv

COMMANDS: dict[str, tuple[str, str]] = {
    "build": ("memex.cli.build", "grow a knowledge graph from a seed query"),
    "chat": ("memex.cli.chat", "ask a grounded question of a graph"),
    "hypothesize": ("memex.cli.hypothesize", "generate ranked, evidence-backed hypotheses"),
    "fetch-demo": ("memex.cli.fetch_demo", "download the prebuilt demo graph"),
    "inspect": ("memex.cli.inspect_build", "print a build's trace in the terminal"),
    "ui": ("memex.cli.ui", "browse build and query traces in the browser"),
    "serve": ("memex.cli.serve", "run the HTTP API"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: memex <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += ["", "Run `memex <command> --help` for a command's options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"memex: unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    load_dotenv()
    module = importlib.import_module(COMMANDS[command][0])
    return module.main(rest) or 0
