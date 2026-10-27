#!/usr/bin/env python3
"""Compare nominal and top-pT-reweighted ttbar using the canonical plotter.

The upper panel contains the nominal stack with ttbar at the top, its black
total outline, and a blue dashed reweighted total. All non-ttbar histograms,
including the nominal QCD-MC normalisation and DD estimates, are identical.
The lower panel shows Data/Bkg(nominal) and Bkg(reweighted)/Bkg(nominal).
Bands refer to the nominal prediction; the two correlated predictions' ratio
is shown without an independent-sample statistical uncertainty.
Red open boxes show expected Asimov Poisson counting precision sqrt(N),
with N taken from nominal total-background event yields after rebinning.
The ratio boxes show 1 +/- 1/sqrt(N); non-positive bins are omitted.
This is separate from background-template statistics and is not added to
stat+syst bands. Observed data blinding and Asimov point modes are unchanged.

Accepts plotter.py options plus --regions selected|sr|bjet-ss|lightjet-os|all.
Defaults: selected region (OS BJet), all variables, SR blinded.
Dimuon mass defaults to 5--80 GeV and stat+syst in SR; other variables stay
stat-only. Ratio limits default to 0.5--1.5. Explicit CLI options override these.
For --regions all, SR retains the requested blinding and background methods;
SS BJet and OS LightJet use observed control data, QCD/DY MC, and stat-only.
Blinded kinematics use nominal-background Asimov points, exactly as plotter.py.
No fits, templates or production ROOT files are updated.

Examples:
  python3 plotter_toppt_comparison.py --era Run2 --regions all
  python3 plotter_toppt_comparison.py --era 2018 --variable dimuon_mass --xmax 80 --blind
  python3 plotter_toppt_comparison.py --era Run3 --regions bjet-ss --variable all
  python3 plotter_toppt_comparison.py --era 2018 --variable dimuon_mass --blind --uncertainty syst+stat --strict
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
                        default="selected", help="all = blinded/requested SR plus observed SS-BJet and OS-LightJet controls")
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
            p.run(this_cfg, comparison_builder=build_reweighted_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
