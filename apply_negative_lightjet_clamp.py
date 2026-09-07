#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).with_name("dy_bkg_estimation.py")
text = path.read_text()
old = '''    for bg in bg_files:\n        h_bg = load_histogram_across_eras(\n            ROOT,\n            args,\n            bg,\n            folder,\n            hist_name,\n            f"subtract_{bg}_{folder}_{hist_name}",\n            required=True,\n        )\n        if h_bg is None:\n            print(f"[WARNING] Background missing during subtraction: {bg}")\n            continue\n        h_sub.Add(h_bg, -1.0)\n\n    return h_sub\n'''
new = '''    for bg in bg_files:\n        h_bg = load_histogram_across_eras(\n            ROOT,\n            args,\n            bg,\n            folder,\n            hist_name,\n            f"subtract_{bg}_{folder}_{hist_name}",\n            required=True,\n        )\n        if h_bg is None:\n            print(f"[WARNING] Background missing during subtraction: {bg}")\n            continue\n        h_sub.Add(h_bg, -1.0)\n\n    # In sparse high-mass light-jet data, Data-QCD-Top-Others can become\n    # slightly negative when the observed data count is zero but the\n    # subtracted MC prediction is nonzero.  A negative event yield is not a\n    # physical DY template, so clamp only the central bin content to zero.\n    # Keep the propagated Sumw2 bin uncertainty unchanged so LightJetStat still\n    # reflects the statistical precision of the data-minus-background source.\n    n_clamped = 0\n    for ibin in range(0, h_sub.GetNbinsX() + 2):\n        if float(h_sub.GetBinContent(ibin)) < 0.0:\n            h_sub.SetBinContent(ibin, 0.0)\n            n_clamped += 1\n    if n_clamped:\n        print(\n            f"[INFO] Clamped {n_clamped} negative bin(s) to zero in "\n            f"LightJetSource ({folder}/{hist_name}); bin errors are unchanged."\n        )\n\n    return h_sub\n'''
if old not in text:
    raise SystemExit("Expected load_subtracted_histogram block not found; refusing to modify file.")
text = text.replace(old, new, 1)
compile(text, str(path), "exec")
path.write_text(text)
print(f"Updated {path}")
