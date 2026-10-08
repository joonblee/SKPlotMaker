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

In both B-jet and light-jet regions, subtraction is `Data - QCD MC - Top MC -
Others MC`. Signed bin contents and propagated errors are retained through
rebinning and NF scaling, including in the saved `DYAux/LightJetSource` and
central NF prediction. Do not clip negative fine bins before merging them:
that would discard background contributions in empty data bins and increase
the merged yield. Negative values cannot be shown on the default log-y axis;
the stored histogram contents are nevertheless preserved.

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

#### 2016postVFP SingleMuon F/G/H diagnostics

To inspect and fit each period separately, without changing the full-era QCD
estimate:

```bash
for period in F G H; do
    python3 qcd_bkg_estimation.py --mode ss-data --year 2016postVFP \
        --data-period "$period" --ss-binning adaptive \
        --ss-min-effective-count 10 --ss-max-bin-width 5 || break
done
```

This reads `Skim_NIsoMuon_SingleMuon_<PERIOD>.root` in the usual
`NIsoMuon/2016postVFP` input directory instead of `data.root`. It fits
`SS Data(period) - Top(full era)*L_period/L_MC - Others(full era)*L_period/L_MC`
over the usual 5--30 GeV range. Data are not scaled. MC contents and errors
receive the same luminosity factor, so their variances receive its square;
signed residual bins remain in the fit. MC shapes/SFs are the existing full-era
ones, with luminosity scaling for this period comparison.

The default recorded luminosities are derived by summing the certified runs in
the [CMS 2016 per-run luminosity table](https://opendata.cern.ch/record/1059/files/2016lumi.txt)
(`23v1`, composite normtag; values are in fb^-1):

| Period | Certified runs used | Luminosity [fb^-1] |
| --- | --- | ---: |
| F, postVFP only | 278769, 278801, 278802, 278803, 278804, 278805, 278808 | 0.418771191 |
| G | 278820--280385 | 7.653261226 |
| H | 281613--284044 | 8.740119303 |

Their sum, 16.812151720 fb^-1, agrees with the actual full-era MC normalisation
16.812151722482 fb^-1 in SKFlat `Event::GetTriggerLumi("Full")` to the precision
of the published table. The denominator uses that MC normalisation, rather
than the rounded 16.8 fb^-1 plot label. F must not use the luminosity of the
entire Run2016F period, which also includes preVFP data.
Use `--period-lumi-fb VALUE --mc-lumi-fb VALUE` to supply the actual values if
your certification/trigger selection or MC production normalisation differs.
These overrides require `--data-period`.

Each period prints SS Data, luminosity-scaled Top/Others and their residual in
5--9, 11--15, 5--30 and 11--80 GeV. Fit figures have their own period label and
luminosity, and filenames end in
`2016postVFP-SingleMuon_<PERIOD>-AllFits.pdf/.png` in the usual plot directory.
`--inspect-binning` prints the period inputs/yields and fit-bin diagnostics
without fitting. This option is restricted to individual 2016postVFP SS-data
diagnostics: it reads no OS observations, applies no OS transfer factor, and
does not write production anchors or `NIsoMuon_SS_fit.root` templates.
It cannot be combined with `--validate-qcd-double-ratio` or
`--uncertainty-diagnostics`, which are full-era operations.

`plotter.py` also accepts `--data-period F/G/H`, with the same luminosities and
`--period-lumi-fb` / `--mc-lumi-fb` overrides. For SS mass comparisons:

```bash
for period in F G H; do
    python3 plotter.py --era 2016postVFP --data-period "$period" \
        --variable dimuon_mass --dimuon-sign ss --unblind \
        --qcd-method mc --dy-method mc --uncertainty stat --strict || break
done
```

For OS comparisons with the existing data-driven estimates:

```bash
for period in F G H; do
    python3 plotter.py --era 2016postVFP --data-period "$period" \
        --variable dimuon_mass --blind --uncertainty syst+stat --strict || break
done
```

Use `--unblind` only when deliberately inspecting OS observations in the search
window. The existing blinding and QCD-MC normalisation prescriptions are retained;
QCD-MC normalisation uses the selected period data after all predictions are
luminosity-scaled. `--no-qcd-normalise` preserves absolute MC normalisation for
comparisons of rates per unit luminosity.

Period data counts/errors are not luminosity-scaled in event plots. Full-era
MC, signal and DD template contents/errors, and their systematic responses,
are multiplied by `L_period/L_MC`. For `xsec` plots the usual additional division
by the selected period luminosity applies. DD statistical uncertainties remain
those of the full-era estimate times this scale; they are not statistics of a
new period-specific fit. QCD transfer and DY NF metadata remain full-era, and
their consistency checks are retained. This tests whether the existing era
prediction describes each period, assuming full-era shapes/SFs and rates per
unit luminosity. It does **not** derive F/G/H-specific QCD or DY estimates or
consume the SS-only diagnostic fits above. The log and DD plots state this
assumption. No production inputs are written.

Plots use the period luminosity and `SingleMuon F/G/H` label, and filenames start
with `2016postVFP_SingleMuon_<PERIOD>_`, so they do not overwrite full-era figures.
A missing period data file/histogram is fatal, without falling back to merged
data. `--qcd-normalisation-diagnostics` is restricted to full-era producer-input
audits and cannot be combined with `--data-period`.

#### Jet kinematics of previously selected QCD events

For the previously traced `QCD_Pt-120To170_MuEnriched` selected SS events,
`qcd_selected_jet_diagnostics.py` reads only their recorded file-local entries:

```bash
python3 -u qcd_selected_jet_diagnostics.py \
    --source-report plots/qcd_event_sources_2016postVFP.json \
    --sign ss --mass-range 11 80 2>&1 | tee plots/qcd_selected_jets_postVFP.log
```

Use `--input original` to inspect the uniquely matched original files instead
of the skim (default). The existing source-trace JSON is required. Event IDs
are checked exactly, including the 64-bit event number. All AK4 jets are
printed with their vector index, stored `jet_pt`, nominal JER factor,
`jet_pt * jet_smearedRes`, eta/phi, DeepJet (`jet_DeepFlavour`) and tight jet ID
when present. This follows the nominal MC `AnalyzerCore::GetAllJets()` pT
convention. Stored `jet_pt` is not an uncorrected raw jet pT. The output contains
all jets without selection cuts; the old audit did not record selected dimuon
jet/tag jet indices. Do not interpret the leading stored jet as the selected
dimuon jet automatically. Inputs, weights and production outputs are unchanged.

#### Statistical uncertainty of the QCD prediction

Individual-era `ss-data` production now also writes `QCDStat/metadata` into both
copies of `NIsoMuon_SS_fit.root`. Central fits, transfer factors, Norm/Shape
contents and zero TH1 errors are retained. The metadata contain the SS central
fit covariance in `(log(A), n, k, m0, w)` coordinates and the primitive Sumw2
transfer statistics, including the shared DY NF-stat contribution.

SS-fit yield statistics are calculated in `qcd_bkg_estimation.py`; the ROOT-free
`qcd_common_transfer.py` contains common-transport fitting/validation and
cross-era statistical rules. No separate statistical execution is required. Alongside the existing
templates, each ROOT file stores `QCDStat/CentralYield`, `FitGradient_0` through
`FitGradient_4`, and `NFGradient_low_transfer`/`NFGradient_mc_double_ratio`.
These histograms contain native-bin yield derivatives, with covariance and
transfer variances in `QCDStat/metadata` (`NPS26009_QCDStat_v3` for common
transport, `NPS26009_QCDStat_v2` for legacy methods). Consumers sum
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

Low-data NF-stat and SS-fit-stat share SS events. Their cross-covariance has not been
calculated, so run-common transport uses the era-local conservative bound
`sigma_local = sigma_lowNFstat + sigma_SSfitStat`, plus the independent MC
transport component. The MC component is shared within each Run; the local
bounds combine in quadrature across eras. Legacy methods retain the full
`sigma_stat = sigma_NFstat + sigma_SSfitStat` bound. It is not an independent quadrature
of those two terms or a demonstrated confidence-interval coverage prescription.
The existing NF-modelling and functional-form uncertainties remain separate.
`plotter.py` includes these statistics per plotting-bin integral, summing common
MC responses before squaring, even in mass `stat-only` mode. It reads only the
nominal QCD file for this purpose. Final `syst+stat --strict` plots also retain the
existing Norm/Shape and other background uncertainties.

For each era, the mass plot audits the stored transfer primitives: the 5--9 GeV
QCD prediction uses `R_data(low)`, and the 11--80 GeV prediction uses
`R_data(low) * D_common(Run2 or Run3)` by default. Legacy methods remain explicit
options below. These factors are already contained in
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

`QCD_TRANSFER_LOW_WINDOW` and `QCD_TRANSFER_HIGH_WINDOW` in
`qcd_bkg_estimation.py` govern both the transfer-factor calibration windows and
the output-template support. Nominal, Norm/Shape and statistical derivative
integrals all use these constants. Changing the low window to `(6.0, 9.0)`
also makes the QCD prediction below 6 GeV zero; it is not a calibration-only
change. Regenerate the affected ROOT templates after changing these windows.
The separate `plotter.py` low window governs blinded QCD-MC validation
normalisation, not the SS-data estimator's factor calculation.

Individual-era SS templates accept `--transfer-method`:
`run-common` (default) uses era-local `R_data(low)` in the low region and
`R_data(low) * D_common` in the high region. One MC double ratio is fitted for
the four Run-2 eras and another for the four Run-3 eras. Every era's SS shape
fit and low-data calibration remain local. `era-specific` retains the
previous same-era `R_data(low) * R_MC(high) / R_MC(low)` high factor; `data-low`
uses `R_data(low)` in both regions.

The old option `--qcd-transfer-method` remains an alias, and the old value
`mc-double-ratio` is an alias for `era-specific`.

```bash
python3 qcd_bkg_estimation.py --mode ss-data --year 2016postVFP \
    --ss-binning adaptive --ss-min-effective-count 10 --ss-max-bin-width 5 \
    --transfer-method data-low
python3 plotter.py --era 2016postVFP --unblind --variable dimuon_mass \
    --uncertainty syst+stat --qcd-normalisation-diagnostics --strict
```

The method is recorded under `QCDStat/metadata`; the plotter reads it from the
ROOT file without another method option. Switching methods regenerates the
same `NIsoMuon_SS_fit.root` output, including nominal, Norm/Shape and statistical
basis histograms. `qcd_yield_diagnostics.py` also reports the selected method and
distinguishes the measured MC double ratio from the applied transport.

To verify that a regenerated template reaches the final plot, use the existing
`plotter.py ... --qcd-normalisation-diagnostics --strict` command above and check
`[qcd-transfer-check]` for `method=run-common` and
`T_high=R_data(low)*D_common`. The individual-era producer log must also end in
`[SAVE] .../<ERA>/NIsoMuon_SS_fit.root` and `[DONE]` for each era. A validation
report alone does not replace a template, and regenerating a template does not
redraw an existing final plot. The `FitPlot_FnVariation-SS_fit_...` figures show
SS shapes before OS transport, so their normalisation does not acquire the
common factor. The low window retains its local data factor; only the high
window receives the common MC double ratio.

A compatible common MC double ratio tests era dependence of MC transport, not
closure with observed OS residuals. Relative to `data-low`, the high-mass QCD
yield is multiplied by `D_common` for identical SS shapes and low-data inputs.
Relative to `era-specific`, it is multiplied by `D_common/D_era`. The common
method can therefore change the central yield substantially even when the
compatibility test passes. Use the read-only yield audit below to check closure
and stored/current input agreement separately.

`data-low` retains the current low-mass data/MC `QCD_norm`
modelling comparison in both regions and propagates the common data-ratio
statistics, without MC transport statistics. Shape fits and mass windows are
unchanged. QCD MC inputs remain required for the modelling comparison and
diagnostics. Older files without a method field are treated as
`era-specific`. The per-era method continues to store `mc-double-ratio` in
ROOT metadata so existing plotter and Combine consumers remain compatible;
User-facing logs display `era-specific`. For compatibility with derivative-basis
consumers, the legacy `mc_double_ratio`/`double_ratio_variance` metadata keys contain the
applied transport (1/0 in `data-low`); measured MC values are retained separately
as `measured_mc_double_ratio` and `measured_double_ratio_variance`.
Run individual eras sequentially; combined periods write only SS anchors.
Pass `--transfer-method era-specific` explicitly to restore the previous
per-era MC prescription by regenerating that era's template.

#### Common MC transport validation

Run this before regenerating the templates:

```bash
python3 qcd_bkg_estimation.py --mode ss-data --year Run2+3 \
    --validate-qcd-double-ratio
```

This reads only `NIsoMuon_QCD_Inclusive.root` OS/SS histograms in each era,
using `QCD_TRANSFER_LOW_WINDOW` and `QCD_TRANSFER_HIGH_WINDOW`. It writes
`plots/QCDDoubleRatioValidation_Run2.{json,csv,pdf,png}` and the corresponding
Run3 files. It never reads observed OS data or changes SS fits, anchors or
production ROOT templates. `--year Run2`, `Run3` or one individual era selects
its Run; `Run2+3` validates the two Runs separately.

The estimator is a constant GLS fit to era `log(D)` values, with first-order
variance `sum(Sumw2 / yield**2)` over the four disjoint MC primitives. It is
not the double ratio of merged era histograms, whose low/high era mixture can
change. Reports contain each ratio and log-normal 1-sigma interval, all four
primitive yields/Sumw2/effective counts, leave-one-out pulls, global chi-square,
ndf and p-value. Pulls include the covariance between an era and the fit to the
other eras. Being statistically compatible does not establish equality; the
test includes no detector/generator modelling uncertainty. Sparse effective
counts trigger an approximation warning. Validation-only exits 2 on a failed
compatibility test; it does not inflate errors or tune the central factor.

All four era files and stored Sumw2 are required: missing inputs are errors,
not grounds to silently omit an era. MC statistical independence across eras
is the default assumption. If event samples are reused/correlated, supply a
statistical covariance JSON with `--qcd-transfer-log-covariance PATH`. Its format
is `{ "Run2": { "eras": [...], "log_covariance": [[...], ...] }, "Run3": ... }`,
in the era order recorded in the reports; diagonal entries must match the
primitive Sumw2 propagation. The covariance must be positive definite and must
come from an actual shared-sample calculation, not arbitrary error inflation.

Individual-era default production automatically repeats this validation before
fitting. A p-value below `--qcd-transfer-validation-pmin` (default 0.05) stops
production before changing fits/anchors/templates. An explicit
`--allow-incompatible-qcd-transfer` overrides the check and retains the failure
in the report/metadata. Validation-only still returns 2 on failure.

After validation, regenerate all affected era templates sequentially with the
usual adaptive-binning commands. A common-fit input fingerprint is stored in
each ROOT file; plotting/card generation rejects mixed common/legacy templates
or different common fits within one Run. The plotter reads the method directly.
`QCD_norm` still uses each era's low-data/low-MC comparison; no modelling
uncertainty is inferred from noisy era-to-era scatter. Update the
`higgs_combine` and `combine_review` consumers and rebuild cards for v3: their
era-local statistical Gaussian and shared `QCD_MCTransferStat_Run2/Run3`
Gaussian preserve the common MC covariance. Older consumers reject v3 instead
of silently treating the common factor as independent across eras.

The QCD-MC fit excludes `9 < m(mumu) < 11 GeV`.  The OS/SS transfer diagnostics use `5 < m(mumu) < 9 GeV` as the low-mass region and `11 < m(mumu) < 80 GeV` as the high-mass region.

To diagnose a low QCD data-driven yield using existing ROOT files:

```bash
python3 plotter.py --era 2016postVFP --unblind --variable dimuon_mass \
    --uncertainty syst+stat --qcd-normalisation-diagnostics
```

This prints native-bin event counts and writes
`plots/2016postVFP_qcd_normalisation_diagnostics_unblind_data-driven.json`.
It reports the low/high windows plus high-mass subwindows split at 20 and 30 GeV:
SS-fit/background-subtracted SS data, SS-fit/SS MC, the applied transfer,
local MC OS/SS, DD/MC, and OS-residual/DD. The latter is a required closure
scale for diagnosis only; no factor is applied to the data-driven prediction.
SS-fit, SS-residual and OS-residual yields and the residual OS/SS ratio are
printed directly. Undefined ratios with a non-positive denominator are labelled
separately from blinded OS observations.
The yield decomposition is
`DD/MC = (SS-fit/SS-MC) * (applied transfer/local MC OS/SS)`.
In the low window, DD/OS residual equals SS-fit/SS residual if the calibration
inputs match; the OS/SS anchor alone does not force the fitted integral to match
the observed SS integral.

The report uses the producer's subtraction (OS: Top + DY DD + Others;
SS: Top + Others), also compares Top with the plotter's tt + ST, and compares
stored transfer primitives with current inputs. New producer files record actual
window endpoints; older files infer template support from NF derivatives and
label it explicitly. Existing matching files can be diagnosed without a refit.
With `--blind`, OS-data closure is omitted for windows intersecting the blinded
interval. MC-mode reports also record the MC scale used by the plotter: an
unblinded MC plot is normalised over its displayed range, whereas DD QCD is
transported from the low-mass anchor without another normalisation.

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
| `qcd_common_transfer.py` | Shared QCD transport and statistics | Imported by the estimator and main plotter; fits the common Run-2/Run-3 MC double ratio and preserves its correlation across eras. Required for production. |
| `qcd_yield_diagnostics.py` | QCD yield and closure audit | Reads component yields, SS-fit closure and stored/current transfer inputs without changing templates. |
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

## QCD yield and OS/SS diagnostic examples

For a read-only native-yield audit of the four Run-2 eras:

```bash
python3 qcd_yield_diagnostics.py --unblind
python3 qcd_yield_diagnostics.py --era 2016preVFP 2016postVFP --unblind
```

This prints OS/SS Data, tt/ST/Top/Others, DY MC, OS DY DD, raw QCD MC and
OS QCD DD yields in the stored low/high transfer windows, 11--15 GeV and
each 1-GeV bin there. It reports residual/MC and OS/SS ratios, SS-fit closure,
current/stored transfer primitives and separate global validation scale factors.
`--plot-range 5 120` reproduces the default unblinded QCDMC_norm integration
range; set it to the actual validation plot range when different. SS estimator
subtraction excludes DY; SS validation-plot factors include DY MC, as in the
supplied validation plots. Without `--unblind`, high-mass OS observations and
OS global factors remain hidden. The script opens ROOT files in READ mode,
does not refit or change templates, and writes no files.

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
