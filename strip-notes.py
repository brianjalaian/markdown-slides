#!/usr/bin/env python3
"""
Strip presenter Note: blocks from slides.local.md → slides.md

Usage:
    python3 strip-notes.py <deck-directory>
    python3 strip-notes.py ensemble-methods
    python3 strip-notes.py          # processes all decks that have slides.local.md

Run this before committing to ensure slides.md on GitHub has no presenter notes.
"""

import re
import sys
from pathlib import Path


def strip_notes(content: str) -> str:
    # Remove everything from a Note: line up to (but not including) the next --- or end of file
    stripped = re.sub(r'\nNote:.*?(?=\n---|\Z)', '', content, flags=re.DOTALL)
    # Clean up extra blank lines left behind
    stripped = re.sub(r'\n{3,}', '\n\n', stripped)
    return stripped.rstrip() + '\n'


def process_deck(deck_dir: Path):
    src = deck_dir / 'slides.local.md'
    dst = deck_dir / 'slides.md'
    if not src.exists():
        print(f'  Skipping {deck_dir.name}: no slides.local.md found')
        return
    content = src.read_text(encoding='utf-8')
    stripped = strip_notes(content)
    dst.write_text(stripped, encoding='utf-8')
    note_count = content.count('\nNote:')
    print(f'  {deck_dir.name}: stripped {note_count} Note: blocks → slides.md')


def main():
    repo_root = Path(__file__).parent

    if len(sys.argv) > 1:
        decks = [repo_root / sys.argv[1]]
    else:
        # Find all directories that contain slides.local.md
        decks = [p.parent for p in repo_root.glob('*/slides.local.md')]

    if not decks:
        print('No slides.local.md files found.')
        return

    print('Stripping presenter notes...')
    for deck in sorted(decks):
        process_deck(deck)
    print('Done. Commit slides.md — slides.local.md stays local.')


if __name__ == '__main__':
    main()
