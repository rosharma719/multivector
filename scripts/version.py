#!/usr/bin/env python3
"""Read, validate, or bump the Cargo package Semantic Version."""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Cargo.toml"
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$")
VERSION_LINE = re.compile(r'^(version\s*=\s*")([^"]+)("\s*)$', re.MULTILINE)


def current() -> str:
    match = VERSION_LINE.search(MANIFEST.read_text())
    if match is None or not SEMVER.fullmatch(match.group(2)):
        raise SystemExit("Cargo.toml has no valid Semantic Version package version")
    return match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--bump", choices=("major", "minor", "patch"))
    group.add_argument("--set")
    args = parser.parse_args()
    old = current()
    if args.print:
        print(old)
        return
    if args.check:
        print(f"valid SemVer: {old}")
        return
    if args.set:
        new = args.set
        if not SEMVER.fullmatch(new):
            raise SystemExit(f"not valid Semantic Version: {new}")
    else:
        major, minor, patch = map(int, old.split("-", 1)[0].split("."))
        if args.bump == "major":
            new = f"{major + 1}.0.0"
        elif args.bump == "minor":
            new = f"{major}.{minor + 1}.0"
        else:
            new = f"{major}.{minor}.{patch + 1}"
    MANIFEST.write_text(VERSION_LINE.sub(rf"\g<1>{new}\g<3>", MANIFEST.read_text(), count=1))
    print(f"{old} -> {new}")
