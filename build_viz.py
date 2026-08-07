"""
Build the standalone visualiser page.

Injects viz_data.json into viz_template.html and writes overgraze.html, which
is fully self-contained -- no external scripts, fonts, or data fetches, so it
opens offline and satisfies the artifact CSP.

Regenerate after changing the simulation:

    python export_viz.py && python build_viz.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
TOKEN = "/*__VIZ_DATA__*/"

template = (HERE / "viz_template.html").read_text(encoding="utf-8")
data = (HERE / "viz_data.json").read_text(encoding="utf-8")

if TOKEN not in template:
    raise SystemExit(f"placeholder {TOKEN} not found in viz_template.html")

# The payload sits in a <script type="application/json"> block, so the only
# sequence that could break out of it is a literal "</script"; JSON escapes
# the slash harmlessly and the browser still parses it.
data = data.replace("</", "<\\/")

out = HERE / "overgraze.html"
out.write_text(template.replace(TOKEN, data), encoding="utf-8")
print(f"wrote {out.name}: {out.stat().st_size/1024:.0f} KB")
