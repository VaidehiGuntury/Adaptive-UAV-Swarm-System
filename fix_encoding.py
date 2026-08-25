"""Fix non-ASCII characters in Python source files."""
import pathlib

files = [
    "src/search/target_manager.py",
    "src/search/target.py",
    "src/search/tracker.py",
    "src/search/mission_phase.py",
    "src/search/search_controller.py",
    "src/config/loader.py",
    "src/config/search_config.py",
]

replacements = [
    ("\u2014", "--"),   # em dash
    ("\u2013", "-"),    # en dash
    ("\u2019", "'"),    # right single quote
    ("\u2018", "'"),    # left single quote
    ("\u201c", '"'),    # left double quote
    ("\u201d", '"'),    # right double quote
    ("\u00e0", "a"),    # a-grave (mangled em dash)
    ("\u00f9", "u"),    # u-grave
]

for fname in files:
    p = pathlib.Path(fname)
    if not p.exists():
        print(f"SKIP {fname}")
        continue
    raw = p.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    original = text
    for bad, good in replacements:
        text = text.replace(bad, good)
    if text != original:
        p.write_text(text, encoding="ascii", errors="replace")
        print(f"FIXED  {fname}")
    else:
        print(f"ok     {fname}")
