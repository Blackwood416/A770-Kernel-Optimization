#!/usr/bin/env python3
"""Validate relative markdown links and heading anchors across the skill docs.

Checks every ``[text](target)`` link in SKILL.md and references/**/*.md:
the target file must exist, and a ``#anchor`` fragment must match a heading
slug in the target file (GitHub-style slug: lowercase, punctuation removed,
spaces to hyphens). External URLs are skipped.

Run with the other harness tests:

    python -m unittest discover -s scripts/tests -p "test_*.py"
"""

from __future__ import annotations

import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILL_MD = ROOT / "SKILL.md"
REFERENCES_DIR = ROOT / "references"

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIPPED_PREFIXES = ("http://", "https://", "mailto:", "#")


def slugify(heading: str) -> str:
    """GitHub-style heading slug: lowercase, drop non-word chars except
    spaces and hyphens, then turn every space into a hyphen. A removed
    punctuation mark between spaces therefore yields a double hyphen, exactly
    like GitHub (e.g. ``Vector + Float`` -> ``vector--float``)."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return s.replace(" ", "-")


def headings_of(path: pathlib.Path) -> set[str]:
    slugs: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            slugs.add(slugify(line.lstrip("#").strip()))
    return slugs


def check_file(path: pathlib.Path, failures: list[str]) -> None:
    headings = headings_of(path)
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue  # code blocks are not link targets (e.g. C++ lambdas)
        for match in LINK_RE.finditer(line):
            target = match.group(1).strip()
            if target.startswith(SKIPPED_PREFIXES):
                continue
            target = target.split(" ", 1)[0]  # drop an optional title
            file_part, _, anchor = target.partition("#")
            dest = path if not file_part else (path.parent / file_part).resolve()
            rel = path.relative_to(ROOT)
            if not dest.exists():
                failures.append(
                    f"{rel}:{lineno}: missing file {file_part!r} for link {target!r}"
                )
                continue
            if anchor:
                slugs = headings if dest == path else headings_of(dest)
                if slugify(anchor) not in slugs:
                    failures.append(
                        f"{rel}:{lineno}: missing anchor #{anchor} in "
                        f"{dest.relative_to(ROOT)} for link {target!r}"
                    )


def collect_markdown() -> list[pathlib.Path]:
    files = [SKILL_MD]
    files.extend(sorted(REFERENCES_DIR.rglob("*.md")))
    return files


class LinkIntegrityTest(unittest.TestCase):
    def test_all_relative_links_and_anchors_resolve(self) -> None:
        failures: list[str] = []
        for path in collect_markdown():
            check_file(path, failures)
        self.assertEqual(
            failures,
            [],
            "broken markdown links/anchors:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
