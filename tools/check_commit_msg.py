#!/usr/bin/env python3
"""
Check commit messages against COMMIT_POLICY.md (standard library only).

    python tools/check_commit_msg.py .git/COMMIT_EDITMSG      # one message file (commit-msg hook)
    python tools/check_commit_msg.py --range origin/master..HEAD   # every commit in a range (CI)
    echo "fix(ptf): clamp p" | python tools/check_commit_msg.py -  # stdin
    echo "$PR_TITLE" | python tools/check_commit_msg.py --header-only -   # a PR title

Errors fail the check; warnings are printed but do not.
"""

import argparse
import re
import subprocess
import sys

RESEARCH_TYPES = {"math", "arch", "data", "hparam", "exp"}
ENGINEERING_TYPES = {"feat", "fix", "perf", "refactor", "docs", "test", "build", "ci", "style", "chore"}
TYPES = RESEARCH_TYPES | ENGINEERING_TYPES

HEADER_RE = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[a-z0-9][a-z0-9._/-]*)\))?"
    r"(?P<breaking>!)?"
    r": (?P<subject>\S.*)$"
)
# Messages git or GitHub generate; not written by hand, not checked.
SKIP_RE = re.compile(r"^(Merge |Revert \"|fixup! |squash! |amend! )")
SCISSORS = "# ------------------------ >8 ------------------------"
NON_IMPERATIVE_RE = re.compile(r"^[a-z]+(ed|ing)$")
NON_IMPERATIVE_OK = {"embed", "feed", "seed", "need", "speed", "shred", "bring", "string", "spring"}
SUMMARY_SOFT, SUMMARY_HARD = 50, 72


def clean(message: str) -> list:
    """Strip git comment lines (and everything below the --verbose scissors)."""
    lines = []
    for line in message.splitlines():
        if line.startswith(SCISSORS):
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def check(message: str, header_only: bool = False):
    """Return ``(errors, warnings)`` for one commit message (or only its header)."""
    errors, warnings = [], []
    lines = clean(message)
    if not lines:
        return ["empty commit message"], warnings
    header = lines[0]
    if SKIP_RE.match(header):
        return errors, warnings

    m = HEADER_RE.match(header)
    if not m:
        errors.append(
            "header must look like '<type>(<scope>): <summary>' "
            "(lowercase type, optional lowercase scope, colon, one space)"
        )
        return errors, warnings

    ctype, scope, subject = m["type"], m["scope"], m["subject"]
    if ctype not in TYPES:
        errors.append(f"unknown type '{ctype}'; allowed: {', '.join(sorted(TYPES))}")
    if len(header) > SUMMARY_HARD:
        errors.append(f"header is {len(header)} chars; the hard limit is {SUMMARY_HARD}")
    elif len(header) > SUMMARY_SOFT:
        warnings.append(f"header is {len(header)} chars; aim for {SUMMARY_SOFT} or fewer")
    if subject.endswith("."):
        errors.append("summary must not end with a period")
    first = subject.split()[0]
    if first[0].isupper() and first[1:].islower():
        warnings.append(f"start the summary in lowercase ('{first.lower()}', not '{first}')")
    if NON_IMPERATIVE_RE.match(first.lower()) and first.lower() not in NON_IMPERATIVE_OK:
        warnings.append(f"use the imperative mood ('{first}' -> e.g. 'add', 'fix', 'remove')")
    if ctype in RESEARCH_TYPES and not scope:
        warnings.append(f"'{ctype}' commits should name a scope (architecture, system or module)")
    if header_only:
        return errors, warnings

    if len(lines) > 1 and lines[1]:
        errors.append("leave a blank line between the header and the body")
    body = [line for line in lines[2:] if line]
    if ctype in RESEARCH_TYPES and not body:
        errors.append(f"'{ctype}' commits need a body explaining what changed and why")
    if m["breaking"] and not any(line.startswith("BREAKING CHANGE: ") for line in lines):
        warnings.append("'!' marks a breaking change; add a 'BREAKING CHANGE: <what breaks>' footer")
    return errors, warnings


def _messages_in_range(rev_range: str):
    shas = subprocess.run(
        ["git", "rev-list", "--no-merges", rev_range],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    for sha in reversed(shas):
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%B", sha],
            check=True, capture_output=True, text=True,
        ).stdout
        yield sha[:10], msg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="commit message file, or '-' for stdin")
    group.add_argument("--range", dest="rev_range", help="git revision range, e.g. origin/master..HEAD")
    parser.add_argument("--header-only", action="store_true",
                        help="check only the first line (e.g. a pull-request title)")
    args = parser.parse_args(argv)

    if args.rev_range:
        items = list(_messages_in_range(args.rev_range))
    elif args.file == "-":
        items = [("stdin", sys.stdin.read())]
    else:
        with open(args.file, encoding="utf-8") as fh:
            items = [(args.file, fh.read())]

    failed = 0
    for label, msg in items:
        errors, warnings = check(msg, header_only=args.header_only)
        header = (clean(msg) or [""])[0]
        for w in warnings:
            print(f"warning [{label}] {header!r}: {w}", file=sys.stderr)
        for e in errors:
            print(f"error   [{label}] {header!r}: {e}", file=sys.stderr)
        failed += bool(errors)

    if failed:
        print(f"\n{failed} commit message(s) do not follow COMMIT_POLICY.md.", file=sys.stderr)
        if args.file and args.file != "-":
            print("Your message was kept in .git/COMMIT_EDITMSG; fix it and commit again "
                  "(git commit -e -F .git/COMMIT_EDITMSG).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
