#!/usr/bin/env python3
"""Parse a JD text file into a JobSpec YAML artifact and print a summary.

Run:  python scripts/build_jobspec.py [job_description.txt] [job_spec.yaml]
The YAML is reviewable/editable before ranking — that's the "parse once per job"
artifact the ranker loads.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from redrob import jobspec as js  # noqa: E402


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "job_description.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "job_spec.yaml"

    text = src.read_text(encoding="utf-8")
    # title = first non-empty line, trimmed of a leading "Job Description:" label
    title = ""
    for line in text.splitlines():
        if line.strip():
            title = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
            break

    spec = js.from_text(text, title=title)
    spec.to_yaml(out)

    print(f"JobSpec written to {out.name}\n")
    print(f"title          : {spec.title}")
    print(f"experience      : {spec.min_years:.0f}-{spec.max_years:.0f} yrs "
          f"(ideal {spec.ideal_years:.0f})")
    print(f"locations       : {', '.join(spec.locations) or '(none found)'}  "
          f"| relocate_ok={spec.relocate_ok} | modes={spec.work_modes}")
    print(f"role_terms ({len(spec.role_terms)})   : {', '.join(spec.role_terms)}")
    print(f"must_have ({len(spec.must_have_terms)})    : {', '.join(spec.must_have_terms)}")
    print(f"nice_to_have ({len(spec.nice_to_have_terms)}) : {', '.join(spec.nice_to_have_terms)}")
    print("disqualifiers:")
    for name, hits in spec.disqualifiers.items():
        desc = js.DISQUALIFIERS[name]["desc"]
        print(f"   - {name:15} {desc}")
        if hits:
            print(f"       triggers seen: {', '.join(hits)}")


if __name__ == "__main__":
    main()
