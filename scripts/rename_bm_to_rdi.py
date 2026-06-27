"""Rename BM benchmark labels to RDI-Test / RDI-Test-Lines across all publication files."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    "scripts/analyze_experiment3_full.py",
    "scripts/generate_pptx_v5.py",
    "docs/experiment_results.md",
    "publication/ieee-conference/IEEE_Conference_Paper/conference_101719.tex",
    "publication/thesis/Cairo_University_Faculty_of_Engineering_Thesis/MainMatter/Chapter5.tex",
    "publication/thesis/Cairo_University_Faculty_of_Engineering_Thesis/MainMatter/Chapter6.tex",
]

# Order matters: more specific first
REPLACEMENTS = [
    # Dataset key labels (display only — actual keys stay BM-* in JSONL)
    ("BM LR-Handwritten",       "RDI-Test-Lines Handwritten"),
    ("BM LR-Manuscripts",       "RDI-Test-Lines Manuscripts"),
    ("BM LR-Typewritten",       "RDI-Test-Lines Typewritten"),
    ("BM LR Overall",           "RDI-Test-Lines Overall"),
    ("BM-Handwritten",          "RDI-Test-Lines Handwritten"),
    ("BM-Manuscripts",          "RDI-Test-Lines Manuscripts"),
    ("BM-Typewritten",          "RDI-Test-Lines Typewritten"),
    # Longer phrases first
    ("BM Line Recognition Benchmark", "RDI-Test-Lines Benchmark"),
    ("BM Line Recognition",     "RDI-Test-Lines"),
    ("BM Line Segmentation",    "RDI-Test Line Segmentation"),
    ("BM Arabic Benchmark",     "RDI-Test Benchmark"),
    ("BM Arabic benchmark",     "RDI-Test benchmark"),
    ("BM Benchmark",            "RDI-Test Benchmark"),
    ("BM benchmark",            "RDI-Test benchmark"),
    ("BM Overall",              "RDI-Test-Lines Overall"),
    # Python dict/variable labels
    ('"BM-Handwritten"',        '"RDI-Test-Lines Handwritten"'),
    ('"BM-Manuscripts"',        '"RDI-Test-Lines Manuscripts"'),
    ('"BM-Typewritten"',        '"RDI-Test-Lines Typewritten"'),
    # Short standalone BM references in text (careful: only where it means the benchmark)
    ("BM LR",                   "RDI-Test-Lines"),
    ("BM LR\n",                 "RDI-Test-Lines\n"),
    # Slide titles and headers
    ("BM Arabic Benchmark",     "RDI-Test Benchmark"),
    # Final: any remaining "BM" that refers to the benchmark (not the museum)
    # -- be careful not to rename "British Museum" or unrelated acronyms
    ("BM-Handwritten",          "RDI-Test-Lines-Handwritten"),
    ("BM-Manuscripts",          "RDI-Test-Lines-Manuscripts"),
    ("BM-Typewritten",          "RDI-Test-Lines-Typewritten"),
]

# Additional specific replacements for the analysis script
ANALYSIS_REPLACEMENTS = [
    ('BM_SETS  = ["BM-Handwritten", "BM-Manuscripts", "BM-Typewritten"]',
     'BM_SETS  = ["BM-Handwritten", "BM-Manuscripts", "BM-Typewritten"]  # internal keys unchanged'),
    ('"BM-Handwritten": "BM LR-Handwritten"',
     '"BM-Handwritten": "RDI-Test-Lines Handwritten"'),
    ('"BM-Manuscripts": "BM LR-Manuscripts"',
     '"BM-Manuscripts": "RDI-Test-Lines Manuscripts"'),
    ('"BM-Typewritten": "BM LR-Typewritten"',
     '"BM-Typewritten": "RDI-Test-Lines Typewritten"'),
    ("BM LR-Handwritten", "RDI-Test-Lines Handwritten"),
    ("BM LR-Manuscripts", "RDI-Test-Lines Manuscripts"),
    ("BM LR-Typewritten", "RDI-Test-Lines Typewritten"),
    ("BM Benchmark",      "RDI-Test Benchmark"),
    ("BM Line Recognition", "RDI-Test-Lines"),
    ("BM Line Segmentation", "RDI-Test Line Segmentation"),
    ("BM LR Overall",     "RDI-Test-Lines Overall"),
    ("BM Overall",        "RDI-Test-Lines Overall"),
    ("'BM Line Segmentation'", "'RDI-Test Line Segmentation'"),
    ("'BM LR'",           "'RDI-Test-Lines'"),
    ("BM LR\n",           "RDI-Test-Lines\n"),
]

def replace_in_file(path: Path, replacements: list[tuple[str, str]]) -> int:
    content = path.read_text(encoding="utf-8")
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        path.write_text(content, encoding="utf-8")
        n = sum(1 for o, n in replacements if o in original)
        return n
    return 0


# Build a consolidated ordered replacement list for text files
TEXT_REPLACEMENTS = [
    # Most specific first
    ("BM LR-Handwritten",            "RDI-Test-Lines Handwritten"),
    ("BM LR-Manuscripts",            "RDI-Test-Lines Manuscripts"),
    ("BM LR-Typewritten",            "RDI-Test-Lines Typewritten"),
    ("BM LR Overall",                "RDI-Test-Lines Overall"),
    ("BM LR\x20",                    "RDI-Test-Lines "),
    ("BM Line Recognition Benchmark","RDI-Test-Lines Benchmark"),
    ("BM Line Recognition",          "RDI-Test-Lines"),
    ("BM Line Segmentation",         "RDI-Test Line Segmentation"),
    ("BM Arabic Benchmark",          "RDI-Test Benchmark"),
    ("BM Arabic benchmark",          "RDI-Test benchmark"),
    ("BM Benchmark",                 "RDI-Test Benchmark"),
    ("BM benchmark",                 "RDI-Test benchmark"),
    ("BM Overall",                   "RDI-Test-Lines Overall"),
    # Python string literals (analysis script)
    ('"BM-Handwritten": "RDI-Test-Lines Handwritten"',
     '"BM-Handwritten": "RDI-Test-Lines Handwritten"'),  # already correct after first pass
    # Slide / prose references
    ("BM LR-Handwritten",            "RDI-Test-Lines Handwritten"),
    ("BM Benchmark",                 "RDI-Test Benchmark"),
    # BM dict values in generate_pptx_v5.py
    ('"BM LR-Handwritten"',          '"RDI-Test-Lines Handwritten"'),
    ('"BM LR-Manuscripts"',          '"RDI-Test-Lines Manuscripts"'),
    ('"BM LR-Typewritten"',          '"RDI-Test-Lines Typewritten"'),
    ("BM LR Benchmark",              "RDI-Test-Lines Benchmark"),
    # LaTeX
    ("BM~Line~Recognition",          "RDI-Test-Lines"),
    ("\\texttt{BM}",                 "RDI-Test"),
    ("BM\\ Line\\ Recognition",      "RDI-Test-Lines"),
    # After prior replacements these need catching
    ("RDI-Test-Lines Handwritten Handwritten", "RDI-Test-Lines Handwritten"),
    ("RDI-Test-Lines Manuscripts Manuscripts", "RDI-Test-Lines Manuscripts"),
    ("RDI-Test-Lines Typewritten Typewritten", "RDI-Test-Lines Typewritten"),
]

# BM_LABELS dict in analyze script needs specific fix
ANALYSIS_LABEL_FIX = {
    '"BM-Handwritten": "BM LR-Handwritten"': '"BM-Handwritten": "RDI-Test-Lines Handwritten"',
    '"BM-Manuscripts": "BM LR-Manuscripts"': '"BM-Manuscripts": "RDI-Test-Lines Manuscripts"',
    '"BM-Typewritten": "BM LR-Typewritten"': '"BM-Typewritten": "RDI-Test-Lines Typewritten"',
}

total_changed = 0
for rel_path in FILES:
    p = ROOT / rel_path
    if not p.exists():
        print(f"MISSING: {rel_path}")
        continue
    content = p.read_text(encoding="utf-8")
    original = content
    for old, new in TEXT_REPLACEMENTS:
        content = content.replace(old, new)
    # Apply analysis-specific dict fixes
    if "analyze_experiment3" in rel_path:
        for old, new in ANALYSIS_LABEL_FIX.items():
            content = content.replace(old, new)
    if content != original:
        p.write_text(content, encoding="utf-8")
        total_changed += 1
        print(f"  UPDATED: {rel_path}")
    else:
        print(f"  unchanged: {rel_path}")

print(f"\nTotal files changed: {total_changed}")
