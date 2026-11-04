# Plot the blinded QCD/DY sideband study

The separate `plotter_sideband_fit.py` consumes JSON exported by
[`higgs_combine/NIsoMuon/limit_workflow_sideband_fit.py`](https://github.com/joonblee/higgs_combine/blob/main/NIsoMuon/limit_workflow_sideband_fit.py).
It does not read target-region data or perform another fit. Existing
`plotter.py` behaviour and outputs are unchanged.

Run the fit in the configured CMSSW/Combine environment, from `NIsoMuon`:

```bash
python3 limit_workflow_sideband_fit.py --masses 12,30,70 --target full
```

Then, from SKPlotMaker:

```bash
python3 plotter_sideband_fit.py \
  /path/to/NIsoMuon/sideband_fit_outputs/Run2Run3_exclude5sigma_full_bin2_era/sideband_fit_Run2Run3_M*.json
```

Outputs under `plots/sideband_fit/<configuration>/`:

- Per-era dimuon-mass plots: postfit process stack, nominal background,
  sideband-only postfit total and its uncertainty, background Asimov points,
  SR boundaries and ratio panel. The SR Asimov point is excluded from the fit.
- One combined Run2+Run3 SR-yield plot against mass hypothesis. Eras are summed
  using joint nuisance draws in the fit exporter, retaining their correlations.

Per-era distributions use **Events / GeV**, because SR and sideband bins have
different widths. The combined plot uses **Events in signal window** and is a
sequence of separate mass hypotheses, not independent spectrum bins. The
ratio denominator is the unchanged nominal background Asimov prediction.
Asimov point error bars show sqrt(N); they represent expected counting precision,
not a random fluctuation of the Asimov dataset.

Colours and stack order match `plotter.py` (DY grey, QCD light blue, ttbar orange,
single top violet, Others light green). If PyROOT is available, colours/order
are read directly from that module; otherwise the corresponding palette is used.
The ratio panel occupies approximately one third of the plot height.

Bands are the 16/84 percentiles of fit-covariance propagation conditional on
parameter bounds. They are not profile intervals. Boundary-sensitive results
are marked; invalid fits are rejected. A free normalisation has no finite
prefit-prior uncertainty, so a prefit uncertainty band is not invented here.

Each file corresponds to a specific mass and sideband prescription. Do not mix
5/7/10-sigma configurations in one mass scan. Their plots go into separate
configuration subdirectories. See the
[statistical model and run instructions](https://github.com/joonblee/higgs_combine/blob/main/NIsoMuon/SIDEBAND_FIT.md)
for masking, joint expected limits, correlations and boundary/extrapolation
choices. Same-model Asimov closure alone does not validate background modelling.
