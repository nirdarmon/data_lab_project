"""Scan authored text files for non-ASCII characters.

Usage:  python .claude/tools/check_ascii.py [path ...]

Exits 1 and prints file:line:col plus the offending code point for every character
outside 0x20-0x7E (tab, newline and carriage return allowed). With no arguments it
scans the repository, skipping the documented exceptions in .claude/rules/ascii-only.md.

For .ipynb files only the authored text is checked - markdown source, code source and
stream output - so that base64 image payloads do not produce false positives.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # .claude/tools/ -> .claude/ -> repo

# Files that are allowed to contain non-ASCII (exception 1 in the rule).
EXEMPT_FILES = {"project_guideline.md", "teacher_review.md"}

SCAN_SUFFIXES = {".md", ".py", ".txt", ".ipynb", ".json", ".yml", ".yaml", ".cfg"}

SKIP_DIRS = {
    ".git", "data", "outputs", "__pycache__", ".ipynb_checkpoints",
    ".venv", "venv", "node_modules", "scratch",
    # Not our text: other students' notebooks and vendored skills fall under
    # exception 1 in .claude/rules/ascii-only.md (preserve existing content).
    "examples", "skills",
}

ALLOWED_CONTROL = {"\t", "\n", "\r"}


def offenders(text):
    """Yield (line, col, char) for each non-ASCII character in text."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        for col, ch in enumerate(line, start=1):
            if ch in ALLOWED_CONTROL:
                continue
            if not (0x20 <= ord(ch) <= 0x7E):
                yield lineno, col, ch


def notebook_segments(path):
    """Yield (label, text) for the authored parts of a notebook."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        yield "parse-error", str(exc)
        return
    for i, cell in enumerate(nb.get("cells", [])):
        yield f"cell[{i}].source", "".join(cell.get("source", []))
        for j, out in enumerate(cell.get("outputs", [])):
            # Stream text and plain-text results are authored-ish; images are not.
            if "text" in out:
                yield f"cell[{i}].out[{j}]", "".join(out["text"])
            plain = out.get("data", {}).get("text/plain")
            if plain:
                yield f"cell[{i}].out[{j}].text", "".join(plain)


def check(path):
    """Return a list of violation strings for one file."""
    rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
    if path.name in EXEMPT_FILES:
        return []

    found = []
    if path.suffix == ".ipynb":
        for label, text in notebook_segments(path):
            for line, col, ch in offenders(text):
                found.append(f"{rel}:{label}:{line}:{col}: U+{ord(ch):04X} {ch!r}")
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            return [f"{rel}: cannot read as UTF-8: {exc}"]
        for line, col, ch in offenders(text):
            found.append(f"{rel}:{line}:{col}: U+{ord(ch):04X} {ch!r}")
    return found


def iter_files(roots):
    for root in roots:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            yield path


def main(argv):
    roots = [Path(a).resolve() for a in argv[1:]] or [REPO]
    violations = []
    scanned = 0
    for path in iter_files(roots):
        scanned += 1
        violations.extend(check(path))

    if violations:
        print(f"Non-ASCII found in {scanned} scanned files:\n")
        for v in violations:
            print("  " + v)
        print(f"\n{len(violations)} violation(s). See .claude/rules/ascii-only.md")
        return 1

    print(f"OK - {scanned} files scanned, all ASCII.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
