"""
Checks of the markdown documentation (README.md, UserGuide.md, AnalysisGallery.md, docs/, visualization/): the images are hosted as assets of the
release `docs-assets` of FlexTRKR/PyFLEXTRKR, not in the repository (the clone stays the size of the code), so the only remote image URLs allowed
start with that release URL; every relative link target exists; and every table row has as many cells as its header (markdownlint MD056: a `|` inside
a code span splits a cell in GitHub markdown, write the alternatives in words or escape the pipe). No network is used.
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_URL = 'https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/'
SKIP = ('ExtraCode/', 'Analysis/ToBeDeleted/', 'tmp/', 'deleteme/', 'benchmarks/')


def markdown_files():
    """The tracked markdown files (git ls-files) except the legacy folders; CLAUDE.md is a local note."""
    try:
        out = subprocess.run(['git', 'ls-files', '*.md'], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('not a git checkout')
    return sorted(f for f in out if not f.startswith(SKIP) and f != 'CLAUDE.md')


def text_without_code_blocks(path):
    """The lines of a markdown file with the fenced code blocks left out (replaced by empty lines, so line numbers stay)."""
    lines, in_code = [], False
    for line in open(os.path.join(REPO, path), encoding='utf-8', errors='ignore').read().splitlines():
        if line.lstrip().startswith('```'):
            in_code = not in_code
            lines.append('')
        else:
            lines.append('' if in_code else line)
    return lines


IMAGE = re.compile(r'!\[[^\]]*\]\(([^)\s]+)[^)]*\)|<img[^>]+src=["\']([^"\']+)["\']')
LINK = re.compile(r'(?<!!)\[[^\]]*\]\(([^)\s]+)[^)]*\)')


def test_the_images_are_release_assets_of_the_docs_assets_release():
    bad = []
    for path in markdown_files():
        for number, line in enumerate(text_without_code_blocks(path), 1):
            for m in IMAGE.finditer(line):
                url = m.group(1) or m.group(2)
                if re.match(r'https?://', url):
                    if not url.startswith(RELEASE_URL):
                        bad.append(f'{path}:{number}: {url}')
                elif not os.path.exists(os.path.normpath(os.path.join(REPO, os.path.dirname(path), url))):
                    bad.append(f'{path}:{number}: {url} (no such file)')
    assert not bad, 'images that are neither files of the repository nor assets of the docs-assets release:\n' + '\n'.join(bad)


def test_relative_links_exist():
    bad = []
    for path in markdown_files():
        for number, line in enumerate(text_without_code_blocks(path), 1):
            for m in LINK.finditer(line):
                url = m.group(1)
                if re.match(r'(https?:|mailto:|#)', url):
                    continue
                target = os.path.normpath(os.path.join(REPO, os.path.dirname(path), url.split('#')[0]))
                if not os.path.exists(target):
                    bad.append(f'{path}:{number}: {url}')
    assert not bad, 'links to files that do not exist:\n' + '\n'.join(bad)


def table_cells(line):
    """The cells of a table row: split on `|` that is not escaped, without the outer pipes."""
    s = line.strip()
    s = s[1:] if s.startswith('|') else s
    s = s[:-1] if s.endswith('|') and not s.endswith('\\|') else s
    return re.split(r'(?<!\\)\|', s)


def test_every_table_row_has_the_number_of_cells_of_its_header():
    separator = re.compile(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$')
    bad = []
    for path in markdown_files():
        lines = text_without_code_blocks(path)
        i = 0
        while i < len(lines):
            if '|' in lines[i] and i + 1 < len(lines) and separator.match(lines[i + 1]):
                n = len(table_cells(lines[i]))
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith('|'):
                    if len(table_cells(lines[j])) != n:
                        bad.append(f'{path}:{j + 1}: {len(table_cells(lines[j]))} cells, the header has {n}')
                    j += 1
                i = j
            else:
                i += 1
    assert not bad, 'table rows with the wrong number of cells (an unescaped | in a code span?):\n' + '\n'.join(bad)
