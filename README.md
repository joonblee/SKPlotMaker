# SKPlotMaker

Analysis-specific PyROOT utilities for the CMS `NIsoMuon` workflow.  The repository contains the post-processing, background-estimation, validation, efficiency, signal-fit, and plotting scripts used for the non-isolated dimuon analysis.

This is **not intended to be a general-purpose plotting package**.  Most defaults intentionally encode the current analysis choices, ROOT histogram naming, input directory layout, blinding convention, and background-estimation strategy.

For coding agents, see [`AGENTS.md`](AGENTS.md) before modifying the analysis code.

## Analysis intent

The analysis targets a narrow dimuon resonance reconstructed inside a jet, with a separate b-tagged event category used for the signal region.  The repository is the final post-processing layer after the analyser has produced ROOT histograms.

The main design is:

1. merge per-sample ROOT outputs into analysis-level process files;
2. derive data-driven DY and QCD background templates;
3. make blinded signal-region and unblinded control/validation plots;
4. validate muon ID and trigger efficiencies;
5. fit signal mass shapes and detector mass resolution.

The analysis logic is deliberately explicit.  Physics choices such as fit ranges, control regions, nuisance-template names, and the nominal data-driven methods should not be changed merely for code simplification.

## Current input layout

Most scripts use

```text
/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon
```

with era directories

```text
2016preVFP
2016postVFP
2017
2018
2022
2022EE
2023
2023BPix
```

and combined-period aliases where supported:

```text
Run2
Run3
Run2+3
full        # accepted by some plotters as an alias for all eras
```

The most common nominal merged files are

```text
data.root
NIsoMuon_QCD_Inclusive.root
NIsoMuon_DYJets_Inclusive.root
NIsoMuon_DYJets_MG_Inclusive.root
NIsoMuon_tt.root
NIsoMuon_ST.root
NIsoMuon_Top.root
NIsoMuon_Others.root
```

The data-driven scripts additionally produce

```text
NIsoMuon_SS_fit.root
NIsoMuon_DYJets_est.root
```

for the QCD and DY estimates, respectively.

## Main workflow

### 1. Merge analyser outputs

`hadd.sh` constructs the standard process files for nominal, `RunSyst`, and `RunXSecSyst` collections.

Examples:

```bash
source hadd.sh 2023
source hadd.sh Run3 nominal
source hadd.sh Run2 RunSyst
source hadd.sh all all
```

The script can also be executed with `bash hadd.sh ...`.  Missing inputs are skipped rather than terminating the shell.

### 2. Build the DY data-driven estimate

The current production mode is the constant normalisation-factor method (`--method nf`, which is also the default).

For each period:

```bash
for era in Run2 Run3 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix; do
    python3 dy_bkg_estimation.py --era ${era} --blind
done
```

The current NF workflow is:

- start from background-subtracted light-jet data;
- derive the central DY normalisation factor from the aMC@NLO DY simulation;
- use the MG LO DY sample as the generator-modelling comparison;
- extract the MC NF in `11 < m(mumu) < 80 GeV`;
- write the final DY estimate to `NIsoMuon_DYJets_est.root`;
- keep the primitive light-jet source and NF metadata under `DYAux/` so that `plotter.py` can propagate `LightJetStat`, `NFStat`, and `NFModel` consistently.

The older parameter-dependent transfer-factor method is retained as a cross-check:

```bash
python3 dy_bkg_estimation.py --era 2023 --method tf --blind
```

### 3. Build the QCD data-driven estimate

The production central QCD template is derived from same-sign data:

```bash
for era in Run2 Run3 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix; do  
  python3 qcd_bkg_estimation.py --mode ss-data --year "$era" --ss-binning adaptive --ss-min-effective-count 10 --ss-max-bin-width 5;   
done
```

`ss-data` fits the SS dimuon-mass distribution and produces the nominal QCD template together with normalisation and analytic-function-envelope shape variations.

SS fits use bin-integrated statistical chi-square over 5--30 GeV, retaining
negative background-subtracted bins. The default `--ss-binning auto` keeps the
original fine bins when the total effective count, `(sum y)^2 / sum(error^2)`,
is at least five events per fine fit bin. Below that threshold it uses 1 GeV
bins up to 11 GeV, 2 GeV bins from 11 to 21 GeV, and one 21--30 GeV tail bin.
This stabilises sparse samples such as 2022 without coarsening every era.
Use `--ss-binning regular` or `--ss-binning legacy` to select either binning
explicitly; `--rebin` then merges those bins further. Output ROOT template
binning is unchanged.

For bin edges determined from each sample's statistics, use the explicit
`--ss-binning adaptive` option (`auto` retains the two-scheme selection above):

```bash
python3 qcd_bkg_estimation.py --mode ss-data --era 2022 \
    --ss-binning adaptive --inspect-binning
python3 qcd_bkg_estimation.py --mode ss-data --era 2022 \
    --ss-binning adaptive --ss-min-effective-count 25 --ss-max-bin-width 5
```

Adaptive binning starts at 5 GeV and merges adjacent native ROOT bins (currently
0.02 GeV over 0--150 GeV). It closes each bin at the first edge satisfying
`Neff = max(sum y, 0)^2 / sum(error^2) >= --ss-min-effective-count` (default 25,
equivalent to at most 20% relative statistical error for a positive sum), or
before the next native bin would exceed `--ss-max-bin-width` (default 5 GeV).
It stops at 30 GeV. An under-target final remainder merges backwards if the
width cap allows it. The selected edges are printed and can vary by era;
combined periods use the histogram summed over their eras before selecting edges.

All original positive **and negative** `Data - Top - Others` bin contents enter
the signed merged yield, and all variances add. Only a nonpositive **merged**
yield receives a zero bin-selection score. Clipping individual negative input
bins to zero would artificially increase the yield. Sparse or negative bins
that cannot reach the target within the width cap remain in the chi-square
with their signed contents and original propagated errors. Thus the precision
target is not guaranteed for every bin. Bins with zero error follow the existing
chi-square exclusion. The choice of data-dependent edges can still change the
fit; preserving signed yields does not make adaptive binning unbiased.

The native grid and the exact 5/30 GeV boundaries are respected. Binning outside
the fit range and output ROOT template binning are unchanged. `--rebin` acts
after adaptive selection and can exceed its width cap; use the default
`--rebin 1` to keep the selected bins. `--inspect-binning` prints the final
fit-bin edges, signed yields, errors, effective counts and under-target bins,
then exits without fits or output writes.

The SS minimizer fits `log(A)` within a common positive amplitude domain,
instead of a seed-dependent linear amplitude box. Multi-start seeds transfer
the ERF/logistic turn-on widths and include the nested `n=0` exponential and
`k=0` power limits. Stored anchors still contain the physical amplitude `A`.
The six model families, nominal Power x Exp x Logistic choice, transfer
normalisation, shape-envelope convention, and QCD-MC objectives are unchanged.

**Do not normally run the commands above in parallel with `&`.**  Each SS fit updates the same `NIsoMuon_SS_fit_anchors.json` catalogue.  Sequential execution avoids concurrent read/modify/write races in that shared file.

#### Statistical uncertainty of the QCD prediction

Individual-era `ss-data` production now also writes `QCDStat/metadata` into both
copies of `NIsoMuon_SS_fit.root`. Central fits, transfer factors, Norm/Shape
contents and zero TH1 errors are retained. The metadata contain the SS central
fit covariance in `(log(A), n, k, m0, w)` coordinates and the primitive Sumw2
transfer statistics, including the shared DY NF-stat contribution.

All NF/SS-fit statistical calculations are contained in `qcd_bkg_estimation.py`;
no separate statistical script or execution is required. Alongside the existing
templates, each ROOT file stores `QCDStat/CentralYield`, `FitGradient_0` through
`FitGradient_4`, and `NFGradient_low_transfer`/`NFGradient_mc_double_ratio`.
These histograms contain native-bin yield derivatives, with covariance and
transfer variances in `QCDStat/metadata` (`NPS26009_QCDStat_v2`). Consumers sum
the stored derivatives over their actual window before propagating covariance.
This preserves fitted-bin and low/high-transfer correlations; summing per-bin
statistical errors in quadrature would not. `plotter.py` and the Combine workflow
only read the ROOT information; they do not rebuild or refit the SS function.
Covariance status 2 or 3 is accepted, including parameter-boundary solutions
such as `n = 0`. Where needed, Minuit2 regularises the covariance to be positive
definite. The returned matrix is used directly, and its status, regularisation
flag and boundary parameters are recorded in ROOT and the diagnostic reports.
Missing DY NF metadata or unavailable/non-finite statistical propagation
prevents replacement of the production ROOT output.

NF-stat and SS-fit-stat share SS events. Their cross-covariance has not been
calculated, so one statistical nuisance uses the conservative first-order bound
`sigma_stat = sigma_NFstat + sigma_SSfitStat`. It is not an independent quadrature
of those two terms or a demonstrated confidence-interval coverage prescription.
The existing NF-modelling and functional-form uncertainties remain separate.
`plotter.py` includes this statistical bound per plotting-bin integral, combining
independent eras in quadrature, even in mass `stat-only` mode. It reads only the
nominal QCD file for this purpose. Final `syst+stat --strict` plots also retain the
existing Norm/Shape and other background uncertainties.

For each era, the mass plot audits the stored transfer primitives: the 5--9 GeV
QCD prediction uses `R_data(low)`, and the 11--80 GeV prediction uses
`R_data(low) * R_MC(high) / R_MC(low)`. These factors are already contained in
the SS-fit template; the plotter does not apply another normalisation. In
`syst+stat` mode it checks the QCD Norm pair against the stored log-symmetric
data/MC factor and the DY central contents/errors against
`DYAux/LightJetSource * DYAux/NF_aMC`. `--strict` rejects mismatched inputs.

For **blinded QCD-MC mass validation** (`--qcd-method mc --blind`), the separate
MC scale factor `(Data - selected non-QCD backgrounds) / QCD MC` is calculated
from native 5--9 GeV bins before display rebinning. Changing `--xmin`, `--xmax`,
or plotting bin edges cannot bring high-mass OS data into this calibration.
Asimov/toy displays continue to use factor 1; object-validation and unblinded
validation retain their existing normalisation prescription.

The mass `syst+stat` band follows the current Combine correlations: PU, muon ID
and muon scale are coherent within each run; JES, JER and muon trigger remain
era-specific. Luminosity uses the same multiyear Cholesky components and
coefficients as `higgs_combine/NIsoMuon/limit_workflow.py`.

Regenerate the individual-era QCD ROOT files for the new derivative storage format
before rebuilding Combine cards. Older metadata-only ROOT files are rejected:

```bash
for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix; do
    python3 qcd_bkg_estimation.py --mode ss-data --year "$era" \
        --ss-binning adaptive --ss-min-effective-count 10 --ss-max-bin-width 5 \
        --uncertainty-diagnostics
done
```

The diagnostic report also shows `sigma_current_model = hypot(form, normModel)`
and its ratio to SS-fit-stat, matching the comparison of existing modelling
uncertainties with the central-fit statistical error. The report's default
12/30/70 GeV windows have a +/-1% half-width; production cards use their actual
signal-resolution counting windows. Existing cards/limits are not updated by
running this producer alone.

QCD simulation is used as a modelling/cross-check fit.  For the current workflow use the log-chi-square objective explicitly:

```bash
for era in Run2 Run3 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix; do
    python3 qcd_bkg_estimation.py --mode qcd-mc --year "${era}" --fit-objective log-chi2
done
```

The QCD-MC fit excludes `9 < m(mumu) < 11 GeV`.  The OS/SS transfer diagnostics use `5 < m(mumu) < 9 GeV` as the low-mass region and `11 < m(mumu) < 80 GeV` as the high-mass region.

### 4. Make the final dimuon-mass plots

A typical full production loop is

```bash
for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do
    python3 plotter.py --era ${era} --blind --variable dimuon_mass \
        --uncertainty syst+stat --draw-signal --xmax 80 \
        --signal-scale 0.01 --ymin 0.01
done
```

For the dimuon mass, the intended final plot uses the data-driven QCD and DY inputs unless a cross-check explicitly requests MC backgrounds.

### 5. Make validation plots

Nominal MC-background validation in the b-jet OS category:

```bash
for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do
    python3 plotter.py --qcd-method mc --dy-method mc --era ${era} \
        --blind --variable all --ymin 0.5
done
```

Same-sign b-jet validation:

```bash
for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do
    python3 plotter.py --qcd-method mc --dy-method mc --era ${era} \
        --unblind --variable all --ymin 0.5 --dimuon-sign ss
done
```

Light-jet validation:

```bash
for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix Run2 Run3 Run2+3; do
    python3 plotter.py --qcd-method mc --dy-method mc --era ${era} \
        --unblind --variable all --ymin 0.5 --jet-flavour light-jet
done
```

### 6. Convener-review diagnostic studies

The dedicated Run-2 SKFlatAnalyzer modes `ConvenerStudy` and
`DYValidationDRStudy` write the histograms used for the current convener
cross-checks.  For the first-pass 2018 study, merge the analyzer outputs with

```bash
source hadd.sh 2018 ConvenerStudy
source hadd.sh 2018 DYValidationDRStudy
```

Then make all current diagnostic plots with

```bash
python3 convener_studies.py --era 2018 --study all
```

or run one study at a time:

```bash
python3 convener_studies.py --era 2018 --study lepton-veto
python3 convener_studies.py --era 2018 --study jet-composition
python3 convener_studies.py --era 2018 --study dijet-mass
python3 convener_studies.py --era 2018 --study dy-vr
```

The default merged input directories are

```text
/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon/ConvenerStudy/2018
/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon/DYValidationDRStudy/2018
```

and can be overridden with `--convener-dir` and `--dy-vr-dir`.

The diagnostics are intentionally separate from the nominal background
workflow:

- **lepton-veto:** compares the selected yield before and after the additional
  electron veto, tau veto, and combined veto, and writes
  `lepton_veto_yields.csv`;
- **jet-composition:** compares the shape-normalised dimuon-jet constituent
  multiplicity and charged-hadron energy fraction for QCD, DY, Top, and chosen
  signal masses;
- **dijet-mass:** compares the shape-normalised
  `m(j_mumu,j_tag)` distribution;
- **dy-vr:** performs the orthogonal
  `DeltaR(mu, reference jet)>0.4` data closure test.  It subtracts QCD, top,
  and minor MC backgrounds in the b-jet and light-jet validation categories,
  derives the aMC@NLO DY transfer factor in `11<m(mumu)<80 GeV`, and compares
  the target data with the scaled light-jet source.  The bin-by-bin values are
  also written to `dy_validation_drgt0p4.csv`.

The ConvenerStudy plots are MC-only and do not display signal-region collision
data.  The DY validation-region plot uses collision data only in the explicitly
orthogonal validation selection.

## File guide

| File | Role | Why it exists |
| --- | --- | --- |
| `hadd.sh` | ROOT-file merger | Converts many analyser outputs into the standard process files expected by the rest of the workflow. Keeps aMC and MG DY samples separate and builds `tt`, `ST`, `Top`, `QCD`, `Others`, and data files. |
| `dy_bkg_estimation.py` | DY data-driven estimate | Builds the light-jet-data-based DY prediction. NF is the production method; TF is retained for closure/model cross-checks. |
| `qcd_bkg_estimation.py` | QCD data-driven and MC fits | Fits SS data for the production QCD shape and uncertainties, and fits QCD MC for validation/model studies. |
| `NIsoMuon_SS_fit_anchors.json` | QCD fit-anchor catalogue | Stores SS fit results used to initialise/anchor later QCD fits across eras and combined periods. This is shared state, so concurrent writers should be avoided. |
| `plotter.py` | Main final/validation plotter | Reads nominal and systematic ROOT templates, combines eras, applies blinding, builds uncertainty bands, overlays signals, and produces the standard CMS-style plots. |
| `plotter_qcdseparate.py` | QCD composition validation | Replaces inclusive QCD with the individual QCD `pT`-binned MuEnriched samples to check which generated `pT` bins dominate a distribution. One common QCD normalisation factor preserves their relative composition. |
| `os_ss_comparison.py` | OS/SS QCD validation | Compares QCD-MC and `Data - nonQCD` OS/SS behaviour. In OS, DY subtraction uses `NIsoMuon_DYJets_est.root`; in SS, only Top and Others are subtracted. |
| `id_eff.py` | Muon-ID tag-and-probe | Self-contained J/psi/Z efficiency and scale-factor measurement. The C++ fit implementation is embedded in the Python script and compiled through ROOT ACLiC. |
| `trig_eff.py` | Trigger-efficiency measurement | Projects the analyser's 2D trigger-efficiency histograms into the required eta regions and pT binning, then makes data/MC efficiency and SF plots. |
| `convener_studies.py` | Convener-review diagnostics | Plots the dedicated 2018 electron/tau-veto, dimuon-jet composition, dijet-mass, and orthogonal DY data-closure studies written by the SKFlatAnalyzer diagnostic modes. |
| `sigFit.py` | Signal mass-shape fit | Fits signal dimuon-mass distributions with a double Crystal Ball or Gaussian and parameterises `sigma_m / m` versus mass. |
| `sigFit_v2.py` | Extended signal fit/interpolation | Extends the signal-fit workflow with interpolation products, closure information, yields, and signal systematic bookkeeping. Use this version when the interpolation outputs are required. |

## Important analysis conventions

### Region naming

The main signal-region histogram directory is based on

```text
OS_POGMedium_tight_BJet_NIsoDimuon
```

with corresponding SS and light-jet control/validation regions constructed by the individual scripts.

### Blinding

Blinding is an analysis constraint, not a plotting preference.  The current plotting/background-estimation code uses an `11–80 GeV` blinded/search interval in the relevant SR workflows.  Do not unblind or change the interval merely to simplify a test.  Use explicitly unblinded control regions for validation.

### Background philosophy

For the final dimuon-mass result:

- QCD: data-driven SS fit;
- DY: data-driven light-jet estimate multiplied by the DY NF;
- ttbar, single top, and Others: simulation.

MC-only QCD or DY modes are validation/cross-check modes unless explicitly requested otherwise.

### Systematic uncertainties

`plotter.py` contains the current template naming and correlation model used for the final plotting/statistical workflow.  It checks experimental, b-tagging, QCD, DY, PDF, alphaS, and scale inputs by process/era.  When changing systematic handling, update producers and consumers together and use `--strict` to expose missing required templates.

Example:

```bash
python3 plotter.py --era Run2 --variable dimuon_mass --blind \
    --uncertainty syst+stat --strict
```

## Object-efficiency and signal-fit utilities

Muon ID:

```bash
python3 id_eff.py --year 2023
python3 id_eff.py --year all
python3 id_eff.py --year 2023 --inspect-hists
```

Trigger:

```bash
python3 trig_eff.py --year 2023
python3 trig_eff.py --year Run2
python3 trig_eff.py --year Run3
python3 trig_eff.py --year 2023 --inspect-hists
```

Signal resolution/interpolation:

```bash
python3 sigFit.py --era Run2
python3 sigFit.py --era Run3
python3 sigFit_v2.py --era Run2 --build-interpolation --interpolation-step 1
```

## QCD pT-bin and OS/SS diagnostic examples

```bash
python3 plotter_qcdseparate.py --era 2017 --variable all
python3 plotter_qcdseparate.py --era Run2 --variable dimuon_mass
python3 os_ss_comparison.py --era Run2 --blind
```

## Getting script-specific help

Most Python scripts intentionally print their detailed usage when run with no arguments or with `--help`.

```bash
python3 dy_bkg_estimation.py
python3 qcd_bkg_estimation.py
python3 plotter.py
python3 id_eff.py
python3 trig_eff.py
python3 sigFit_v2.py --help
```

Run these scripts inside an environment with PyROOT/ROOT available and with access to the analysis ROOT files under the configured server paths.

## Development rule of thumb

When modifying this repository, preserve the physics behaviour first and refactor second.  A small explicit change that keeps histogram names, region definitions, normalisations, fit ranges, blinding, and output conventions intact is preferred to a broad cleanup that silently changes the analysis.
