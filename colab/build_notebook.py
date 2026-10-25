"""Turn fbpinn_corrected_psf.py (cells marked with '# %%') into a Colab notebook.

    python build_notebook.py    # writes fbpinn_corrected_psf.ipynb with QUICK = False
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "fbpinn_corrected_psf.py")
DST = os.path.join(HERE, "fbpinn_corrected_psf.ipynb")

text = open(SRC, encoding="utf-8").read()
text = re.sub(r"^QUICK = (True|False)", "QUICK = False", text, count=1, flags=re.M)

cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    if header.strip() == "[markdown]":
        lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.strip("\n").splitlines()]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.strip("\n")})

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": cells,
}
with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {DST} ({len(cells)} cells)")
