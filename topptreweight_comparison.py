#!/usr/bin/env python3
"""Compare nominal and top-pT-reweighted ttbar using the canonical plotter.

The upper panel contains the nominal stack with ttbar at the top, its black
outline, data points subject to canonical blinding, and blue open triangles
for the reweighted total background. Reweighted markers have no displayed
horizontal or vertical errors, in both panels and for all variables.
Mass plots also show nominal-background Asimov open black circles with
Poisson errors sqrt(N_before), including the blinded mass interval.
Their ratio is centred at one with errors 1/sqrt(N_before). Observed data
remain blinded according to the existing policy. Blinded kinematics retain
the canonical nominal-Asimov/toy point mode without an additional duplicate.
Nominal stat/syst bands and all non-ttbar predictions remain unchanged.
Saved reweighted graphs retain their calculated Poisson errors for inspection;
only their display is suppressed. Additional nominal mass graphs are saved as
nominal_asimov_poisson_stat and nominal_asimov_poisson_stat_ratio.

Accepts plotter.py options plus --regions selected|sr|bjet-ss|lightjet-os|all.
Defaults: selected region (OS BJet), all variables, SR blinded internally.
Mass binning defaults to 1/2/5/10 GeV widths over 5--20/20--40/40--60/60--80
GeV, identical to dimuon_mass_systematics.py. Explicit --bin-edges or --bin-width
override this default; a different --xmin/--xmax uses regular binning.
--rebin merges the selected bins. Kinematic binning is unchanged.
Dimuon mass defaults to stat+syst in SR; other variables stay stat-only.
Ratio limits default to 0.5--1.5. Explicit CLI options override these.
For --regions all, SR retains the requested background methods;
SS BJet and OS LightJet use QCD/DY MC and stat-only bands.
No fits, templates or production ROOT files are updated.

Examples:
  python3 topptreweight_comparison.py --era Run2 --regions all
  python3 topptreweight_comparison.py --era 2018 --variable dimuon_mass --xmax 80 --blind
  python3 topptreweight_comparison.py --era Run3 --regions bjet-ss --variable all
  python3 topptreweight_comparison.py --era 2018 --variable dimuon_mass --blind --uncertainty syst+stat --strict
"""

from dataclasses import replace
import sys

import plotter as p


def build_reweighted_total(ROOT, cfg, norm, edges, nominal):
    """Load only reweighted ttbar and reuse every non-ttbar prediction verbatim."""
    if cfg.top_pt_reweight:
        raise ValueError("Comparison denominator must be the unreweighted prediction.")
    study_cfg = replace(cfg, top_pt_reweight=True)
    scale = p.plot_scale(cfg, norm, nominal.lumi_pb)
    parts = []
    for year in nominal.years:
        if nominal.bkg_by_year.get(year, {}).get("tt") is None:
            raise RuntimeError(f"Missing nominal ttbar for {year}; cannot compare incomplete era sets.")
        parts.append(p.prepare_year_hist(ROOT, study_cfg, year, "tt", -1.0,
                                         edges, scale, nominal.warnings))
    varied_tt = p.sum_hists(parts, "tt_toppt_comparison")
    varied_bkg = dict(nominal.bkg)
    varied_bkg["tt"] = varied_tt
    total = p.sum_background_hists(varied_bkg, "bkg_toppt_comparison")
    print("[TopPt comparison] Only ttbar changes; QCD normalisation, DD estimates "
          "and all other backgrounds are fixed to nominal. Bands describe nominal.")
    return total


def region_configs(cfg, regions):
    if regions == "selected":
        return [cfg]
    out = []
    for region in (["sr", "bjet-ss", "lightjet-os"] if regions == "all" else [regions]):
        if region == "sr":
            out.append(replace(cfg, jet_mode="bjet", dilepton_sign="OS"))
        else:
            out.append(replace(cfg, jet_mode="bjet" if region == "bjet-ss" else "lightjet",
                               dilepton_sign="SS" if region == "bjet-ss" else "OS",
                               blind=False, blind_point_mode="data", qcd_method="mc", dy_method="mc",
                               uncertainty="stat-only", draw_systematics=False))
    return out


def main(argv=None):
    parser = p.make_parser()
    parser.description = __doc__
    parser.set_defaults(data_mode="blind", variable="all", output_dir="plots/toppt_comparison",
                        uncertainty=None, ratio_min=0.5, ratio_max=1.5)
    parser.add_argument("--regions", choices=("selected", "sr", "bjet-ss", "lightjet-os", "all"),
                        default="selected", help="all = SR plus SS-BJet and OS-LightJet controls; reweighted Asimov points in all regions")
    argv = p.normalise_cli_hyphen_typos(sys.argv[1:] if argv is None else argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.top_pt_reweight:
        parser.error("This script always compares both predictions; omit --top-pt-reweight.")
    requested_uncertainty = args.uncertainty
    args.uncertainty = requested_uncertainty or "stat-only"
    cfg = replace(p.config_from_args(args), tt_on_top=True)
    for region_cfg in region_configs(cfg, args.regions):
        for variable in p.variables_for_request(region_cfg):
            this_cfg = replace(region_cfg, variable=variable)
            if p.is_mass_variable(this_cfg):
                this_cfg = replace(
                    this_cfg,
                    xmin=5.0 if this_cfg.xmin is None else this_cfg.xmin,
                    xmax=80.0 if this_cfg.xmax is None else this_cfg.xmax,
                )
                if requested_uncertainty is None and this_cfg.jet_mode == "bjet" and this_cfg.dilepton_sign == "OS":
                    this_cfg = replace(this_cfg, uncertainty="syst+stat", draw_systematics=True)
            this_cfg = p.apply_comparison_mass_binning(this_cfg)
            p.run(this_cfg, comparison_builder=build_reweighted_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


