#!/usr/bin/env python3
"""Offline helper: extract readable text from a .docx (no external deps).

A .docx is a zip of XML; we pull word/document.xml and strip tags. Used once to
turn job_description.docx into job_description.txt so the JobSpec parser (and the
Stage-3 reproduction) can read plain text without a docx dependency.

Run:  python scripts/extract_docx.py job_description.docx [out.txt]
"""
from __future__ import annotations
import re
import sys
import zipfile
from pathlib import Path

_UNESCAPE = [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
             ("&quot;", '"'), ("&#39;", "'"), ("&apos;", "'")]


def docx_to_text(path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab\b[^>]*>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    for a, b in _UNESCAPE:
        text = text.replace(a, b)
    out, prev_blank = [], True
    for line in text.split("\n"):
        line = line.rstrip()
        blank = (line.strip() == "")
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    return "\n".join(out).strip() + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/extract_docx.py <file.docx> [out.txt]")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".txt")
    dst.write_text(docx_to_text(src), encoding="utf-8")
    print(f"wrote {dst}  ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
