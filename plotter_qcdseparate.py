#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIsoMuon validation plotter with QCD pT-binned samples shown separately.

This is intentionally separate from NIsoMuon_validation_plotter.py.
It keeps Data/DY/tt/ST/Others in the usual Data-vs-MC stack, but replaces
NIsoMuon_QCD_Inclusive.root by the individual QCD MuEnriched pT-binned files.

The default QCD normalisation is one common factor

    NF_QCD = (Data - non-QCD MC) / sum(QCD pT-bin MC)

computed after the selected eras are combined.  The same NF is applied to every
QCD pT-bin, so the relative pT-bin composition remains unchanged.

Examples
--------
  python3 plotter_qcdseparate.py --era 2017 --variable all
  python3 plotter_qcdseparate.py --era Run2 --variable dimuon_mass
  python3 plotter_qcdseparate.py --era Run3 --variable jet0_pt --no-qcd-normalise
  python3 plotter_qcdseparate.py --era 2018 --variable all --jet-flavour light-jet
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from array import array
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BASE_DIR_DEFAULT = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
RUN2_ERAS: Tuple[str, ...] = ("2016preVFP", "2016postVFP", "2017", "2018")
RUN3_ERAS: Tuple[str, ...] = ("2022", "2022EE", "2023", "2023BPix")
ALL_ERAS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS
ERA_GROUPS: Dict[str, Tuple[str, ...]] = {
    **{era: (era,) for era in ALL_ERAS},
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
    "full": ALL_ERAS,
    "Run2+3": ALL_ERAS,
}

# pb^-1.  Used only for the CMS luminosity label; histograms are already weighted.
LUMI_PB = {
    "2016preVFP": 19500.0,
    "2016postVFP": 16800.0,
    "2017": 41480.0,
    "2018": 59830.0,
    "2022": 7980.0,
    "2022EE": 26670.0,
    "2023": 17700.0,
    "2023BPix": 9500.0,
}


@dataclass(frozen=True)
class VariableSpec:
    key: str
    hist_name: str
    x_title: str
    xmin: float
    xmax: float
    bin_width: float
    divide_by_bin_width: bool


VARIABLE_SPECS: Dict[str, VariableSpec] = {
    "dimuon_mass": VariableSpec("dimuon_mass", "Dilepton_Mass", "m_{#mu^{+}#mu^{-}} [GeV]", 5.0, 75.0, 1.0, True),
    "jet0_pt": VariableSpec("jet0_pt", "Jet_0_Pt", "p_{T}(j_{#mu#mu}) [GeV]", 30.0, 500.0, 10.0, True),
    "jet0_eta": VariableSpec("jet0_eta", "Jet_0_Eta", "#eta(j_{#mu#mu})", -2.4, 2.4, 0.1, False),
    "jet0_phi": VariableSpec("jet0_phi", "Jet_0_Phi", "#phi(j_{#mu#mu})", -3.0, 3.0, 0.2, False),
    "jet1_pt": VariableSpec("jet1_pt", "Jet_1_Pt", "p_{T}(j_{b-tag}) [GeV]", 30.0, 500.0, 10.0, True),
    "jet1_eta": VariableSpec("jet1_eta", "Jet_1_Eta", "#eta(j_{b-tag})", -2.4, 2.4, 0.1, False),
    "jet1_phi": VariableSpec("jet1_phi", "Jet_1_Phi", "#phi(j_{b-tag})", -3.0, 3.0, 0.2, False),
    "mu_lead_pt": VariableSpec("mu_lead_pt", "Lepton_0_Pt", "p_{T}(#mu_{lead}) [GeV]", 50.0, 500.0, 10.0, True),
    "mu_lead_eta": VariableSpec("mu_lead_eta", "Lepton_0_Eta", "#eta(#mu_{lead})", -2.4, 2.4, 0.1, False),
    "mu_lead_phi": VariableSpec("mu_lead_phi", "Lepton_0_Phi", "#phi(#mu_{lead})", -3.2, 3.2, 0.2, False),
    "mu_sub_pt": VariableSpec("mu_sub_pt", "Lepton_1_Pt", "p_{T}(#mu_{sub}) [GeV]", 10.0, 200.0, 10.0, True),
    "mu_sub_eta": VariableSpec("mu_sub_eta", "Lepton_1_Eta", "#eta(#mu_{sub})", -2.4, 2.4, 0.1, False),
    "mu_sub_phi": VariableSpec("mu_sub_phi", "Lepton_1_Phi", "#phi(#mu_{sub})", -3.2, 3.2, 0.2, False),
}

VARIABLE_ALIASES = {
    "mass": "dimuon_mass",
    "lepton0_pt": "mu_lead_pt",
    "lepton0_eta": "mu_lead_eta",
    "lepton0_phi": "mu_lead_phi",
    "lepton1_pt": "mu_sub_pt",
    "lepton1_eta": "mu_sub_eta",
    "lepton1_phi": "mu_sub_phi",
}


@dataclass(frozen=True)
class QCDBinSpec:
    key: str
    label: str
    filenames: Tuple[str, ...]


# Exactly one file is selected per pT bin and era.  The second spelling is only
# a fallback for old productions; both are never added together.
QCD_BINS: Tuple[QCDBinSpec, ...] = (
    QCDBinSpec("50to80", "QCD 50-80 GeV", (
        "Skim_NIsoMuon_QCD_Pt-50To80_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-50to80_MuEnriched.root",
    )),
    QCDBinSpec("80to120", "QCD 80-120 GeV", (
        "Skim_NIsoMuon_QCD_Pt-80To120_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-80to120_MuEnriched.root",
    )),
    QCDBinSpec("120to170", "QCD 120-170 GeV", (
        "Skim_NIsoMuon_QCD_Pt-120To170_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-120to170_MuEnriched.root",
    )),
    QCDBinSpec("170to300", "QCD 170-300 GeV", (
        "Skim_NIsoMuon_QCD_Pt-170To300_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-170to300_MuEnriched.root",
    )),
    QCDBinSpec("300to470", "QCD 300-470 GeV", (
        "Skim_NIsoMuon_QCD_Pt-300To470_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-300to470_MuEnriched.root",
    )),
    QCDBinSpec("470to600", "QCD 470-600 GeV", (
        "Skim_NIsoMuon_QCD_Pt-470To600_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-470to600_MuEnriched.root",
    )),
    QCDBinSpec("600to800", "QCD 600-800 GeV", (
        "Skim_NIsoMuon_QCD_Pt-600To800_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-600to800_MuEnriched.root",
    )),
    QCDBinSpec("800to1000", "QCD 800-1000 GeV", (
        "Skim_NIsoMuon_QCD_Pt-800To1000_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-800to1000_MuEnriched.root",
    )),
    QCDBinSpec("1000toinf", "QCD >1000 GeV", (
        "Skim_NIsoMuon_QCD_Pt-1000_MuEnriched.root",
        "Skim_NIsoMuon_QCD_Pt-1000toInf_MuEnriched.root",
    )),
)

NONQCD_FILES = {
    "tt": "NIsoMuon_tt.root",
    "ST": "NIsoMuon_ST.root",
    "DY": "NIsoMuon_DYJets_Inclusive.root",
    "Others": "NIsoMuon_Others.root",
}
NONQCD_LABELS = {
    "tt": "t#bar{t}",
    "ST": "single top",
    "DY": "DY",
    "Others": "Others",
}


class NameFactory:
    def __init__(self) -> None:
        self.n = 0

    def unique(self, prefix: str) -> str:
        self.n += 1
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(prefix))
        return f"{clean}_{self.n}"


NAMES = NameFactory()


def import_root():
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError("Could not import PyROOT. Run inside your ROOT/CMSSW environment.") from exc
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2(True)
    return ROOT


def canonical_era(value: str) -> str:
    aliases = {"run2": "Run2", "run3": "Run3", "run2+3": "Run2+3", "run23": "Run2+3", "all": "full"}
    key = aliases.get(value, value)
    if key not in ERA_GROUPS:
        raise ValueError(f"Unknown era: {value}")
    return key


def canonical_variable(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    key = VARIABLE_ALIASES.get(key, key)
    if key != "all" and key not in VARIABLE_SPECS:
        raise ValueError(f"Unknown variable: {value}")
    return key


def canonical_jet(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    if key in {"bjet", "b-jet", "b"}:
        return "bjet"
    if key in {"lightjet", "light-jet", "light"}:
        return "lightjet"
    raise ValueError(f"Unknown jet flavour: {value}")


def canonical_sign(value: str) -> str:
    key = value.strip().upper()
    if key in {"OS", "OPPOSITE", "OPPOSITE-SIGN"}:
        return "OS"
    if key in {"SS", "SAME", "SAME-SIGN"}:
        return "SS"
    raise ValueError(f"Unknown dimuon sign: {value}")


def years_for_era(era: str) -> Tuple[str, ...]:
    return ERA_GROUPS[canonical_era(era)]


def lumi_pb(era: str) -> float:
    return sum(LUMI_PB[y] for y in years_for_era(era))


def lumi_label(era: str) -> str:
    if era in {"full", "Run2+3"}:
        l2 = sum(LUMI_PB[y] for y in RUN2_ERAS) / 1000.0
        l3 = sum(LUMI_PB[y] for y in RUN3_ERAS) / 1000.0
        return f"{l2:.1f} fb^{{-1}} (13 TeV) + {l3:.2f} fb^{{-1}} (13.6 TeV)"
    energy = "13 TeV" if all(y in RUN2_ERAS for y in years_for_era(era)) else "13.6 TeV"
    return f"{lumi_pb(era)/1000.0:.2f} fb^{{-1}} ({energy})"


def region_name(jet_flavour: str, sign: str) -> str:
    category = "BJet" if jet_flavour == "bjet" else "LightJet"
    return f"{sign}_POGMedium_tight_{category}_NIsoDimuon"


def hist_path(spec: VariableSpec, jet_flavour: str, sign: str) -> str:
    region = region_name(jet_flavour, sign)
    return f"{region}/{spec.hist_name}___{region}"


def open_hist(ROOT, filename: str, path: str, *, strict: bool = False):
    if not os.path.isfile(filename):
        if strict:
            raise RuntimeError(f"Missing ROOT file: {filename}")
        return None
    f = ROOT.TFile.Open(filename, "READ")
    if not f or f.IsZombie():
        if strict:
            raise RuntimeError(f"Could not open ROOT file: {filename}")
        return None
    h = f.Get(path)
    if not h:
        f.Close()
        if strict:
            raise RuntimeError(f"Missing histogram: {filename}:{path}")
        return None
    out = h.Clone(NAMES.unique(h.GetName()))
    out.SetDirectory(0)
    f.Close()
    return out


def sum_hists(hists: Iterable[object], name: str):
    out = None
    for h in hists:
        if not h:
            continue
        if out is None:
            out = h.Clone(NAMES.unique(name))
            out.SetDirectory(0)
        else:
            out.Add(h)
    return out


def qcd_file_for_bin(base_dir: str, year: str, spec: QCDBinSpec) -> str:
    era_dir = os.path.join(base_dir, year)
    for name in spec.filenames:
        path = os.path.join(era_dir, name)
        if os.path.isfile(path):
            return path
    return ""


def make_edges(spec: VariableSpec, xmin: Optional[float], xmax: Optional[float], bin_width: Optional[float]) -> List[float]:
    lo = spec.xmin if xmin is None else xmin
    hi = spec.xmax if xmax is None else xmax
    width = spec.bin_width if bin_width is None else bin_width
    if hi <= lo or width <= 0:
        raise ValueError("Require xmax > xmin and bin width > 0")
    n = int(round((hi - lo) / width))
    if n <= 0:
        raise ValueError("No plotting bins")
    # Keep the requested upper edge exact even when floating point round-off is present.
    edges = [lo + i * width for i in range(n + 1)]
    if abs(edges[-1] - hi) > 1.0e-8:
        edges.append(hi)
    else:
        edges[-1] = hi
    return edges


def rebin_hist(ROOT, h, edges: Sequence[float], name: str):
    if not h:
        return None
    out = h.Rebin(len(edges) - 1, NAMES.unique(name), array("d", [float(x) for x in edges]))
    out.SetDirectory(0)
    return out


def divide_by_width(h) -> None:
    if not h:
        return
    for ib in range(1, h.GetNbinsX() + 1):
        w = h.GetXaxis().GetBinWidth(ib)
        if w <= 0:
            continue
        h.SetBinContent(ib, h.GetBinContent(ib) / w)
        h.SetBinError(ib, h.GetBinError(ib) / w)


def prepare_hist(ROOT, h, spec: VariableSpec, edges: Sequence[float], divide_width: bool):
    out = rebin_hist(ROOT, h, edges, spec.key)
    if out and divide_width and spec.divide_by_bin_width:
        divide_by_width(out)
    return out


def load_combined_data(ROOT, args, spec: VariableSpec, edges: Sequence[float]):
    parts = []
    path = hist_path(spec, args.jet_flavour, args.dimuon_sign)
    for year in years_for_era(args.era):
        h = open_hist(ROOT, os.path.join(args.base_dir, year, "data.root"), path, strict=args.strict)
        h = prepare_hist(ROOT, h, spec, edges, args.divide_by_bin_width)
        if h:
            parts.append(h)
    return sum_hists(parts, "data")


def load_combined_nonqcd(ROOT, args, spec: VariableSpec, edges: Sequence[float]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    hpath = hist_path(spec, args.jet_flavour, args.dimuon_sign)
    for proc, filename in NONQCD_FILES.items():
        parts = []
        for year in years_for_era(args.era):
            h = open_hist(ROOT, os.path.join(args.base_dir, year, filename), hpath, strict=args.strict)
            h = prepare_hist(ROOT, h, spec, edges, args.divide_by_bin_width)
            if h:
                parts.append(h)
        hsum = sum_hists(parts, proc)
        if hsum:
            out[proc] = hsum
        elif args.strict:
            raise RuntimeError(f"No histogram loaded for {proc}")
    return out


def load_combined_qcd_bins(ROOT, args, spec: VariableSpec, edges: Sequence[float]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    hpath = hist_path(spec, args.jet_flavour, args.dimuon_sign)
    for qspec in QCD_BINS:
        parts = []
        missing_years = []
        for year in years_for_era(args.era):
            filename = qcd_file_for_bin(args.base_dir, year, qspec)
            if not filename:
                missing_years.append(year)
                continue
            h = open_hist(ROOT, filename, hpath, strict=args.strict)
            h = prepare_hist(ROOT, h, spec, edges, args.divide_by_bin_width)
            if h:
                parts.append(h)
        hsum = sum_hists(parts, f"QCD_{qspec.key}")
        if hsum:
            out[qspec.key] = hsum
        if missing_years:
            msg = f"[WARN] QCD {qspec.key}: missing file in {', '.join(missing_years)}"
            if args.strict:
                raise RuntimeError(msg)
            print(msg, file=sys.stderr)
    return out


def integral_yield(h, *, divided_by_width: bool) -> float:
    if not h:
        return 0.0
    total = 0.0
    for ib in range(1, h.GetNbinsX() + 1):
        value = h.GetBinContent(ib)
        if divided_by_width:
            value *= h.GetXaxis().GetBinWidth(ib)
        total += value
    return total


def compute_qcd_nf(data, nonqcd: Dict[str, object], qcd_bins: Dict[str, object], *, divided_by_width: bool) -> float:
    n_data = integral_yield(data, divided_by_width=divided_by_width)
    n_nonqcd = sum(integral_yield(h, divided_by_width=divided_by_width) for h in nonqcd.values())
    n_qcd = sum(integral_yield(h, divided_by_width=divided_by_width) for h in qcd_bins.values())
    if n_qcd <= 0:
        print("[WARN] QCD normalization denominator <= 0; using NF=1", file=sys.stderr)
        return 1.0
    nf = max(0.0, (n_data - n_nonqcd) / n_qcd)
    print(
        f"[INFO] QCD common NF = (Data - non-QCD MC) / QCD MC = {nf:.6g} "
        f"(Data={n_data:.6g}, non-QCD={n_nonqcd:.6g}, QCD={n_qcd:.6g})"
    )
    return nf


def apply_nf(qcd_bins: Dict[str, object], nf: float) -> None:
    for h in qcd_bins.values():
        h.Scale(nf)


def stat_error_band(total, components: Sequence[object]):
    # total already contains the sum.  ROOT Add propagates uncorrelated bin errors;
    # keep this explicit function for clarity and future extension.
    return total.Clone(NAMES.unique("stat_band"))


def style_root(ROOT) -> None:
    s = ROOT.TStyle("NIsoMuonQCDPt", "NIsoMuon QCD pT bins")
    s.SetOptStat(0)
    s.SetOptTitle(0)
    s.SetPadTickX(1)
    s.SetPadTickY(1)
    s.SetEndErrorSize(0)
    s.SetTextFont(42)
    s.SetLabelFont(42, "XYZ")
    s.SetTitleFont(42, "XYZ")
    s.SetLabelSize(0.040, "XYZ")
    s.SetTitleSize(0.045, "XYZ")
    ROOT.gROOT.SetStyle("NIsoMuonQCDPt")
    ROOT.gROOT.ForceStyle()
    ROOT.TGaxis.SetMaxDigits(3)


def qcd_colors(ROOT) -> Dict[str, int]:
    # Deliberately broad palette so neighbouring pT bins are visually distinct.
    palette = [
        ROOT.kAzure - 9,
        ROOT.kBlue - 7,
        ROOT.kCyan - 9,
        ROOT.kTeal - 7,
        ROOT.kGreen - 7,
        ROOT.kSpring - 7,
        ROOT.kYellow - 7,
        ROOT.kOrange - 4,
        ROOT.kRed - 7,
    ]
    return {spec.key: palette[i] for i, spec in enumerate(QCD_BINS)}


def nonqcd_colors(ROOT) -> Dict[str, int]:
    return {
        "tt": ROOT.kOrange - 2,
        "ST": ROOT.kViolet - 4,
        "DY": ROOT.kGray + 1,
        "Others": ROOT.kSpring - 9,
    }


def style_component(ROOT, h, color: int) -> None:
    h.SetFillColor(color)
    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(1)


def total_hist(nonqcd: Dict[str, object], qcd_bins: Dict[str, object]):
    return sum_hists(list(nonqcd.values()) + list(qcd_bins.values()), "total_mc")


def build_manual_stack(ROOT, nonqcd: Dict[str, object], qcd_bins: Dict[str, object]):
    # bottom -> top.  Draw cumulative copies in reverse order.
    order: List[Tuple[str, object, int]] = []
    ncolors = nonqcd_colors(ROOT)
    qcolors = qcd_colors(ROOT)
    for proc in ("Others", "DY", "ST", "tt"):
        if proc in nonqcd:
            order.append((proc, nonqcd[proc], ncolors[proc]))
    for qspec in QCD_BINS:
        if qspec.key in qcd_bins:
            order.append((qspec.key, qcd_bins[qspec.key], qcolors[qspec.key]))

    cumulative = []
    running = None
    for key, source, color in order:
        if running is None:
            hcum = source.Clone(NAMES.unique(f"stack_{key}"))
        else:
            hcum = running.Clone(NAMES.unique(f"stack_{key}"))
            hcum.Add(source)
        hcum.SetDirectory(0)
        style_component(ROOT, hcum, color)
        cumulative.append((key, hcum))
        running = hcum
    return cumulative


def use_blind_asimov(args) -> bool:
    """Use background-only Asimov points for the blinded OS b-jet signal region."""
    return bool(
        args.blind
        and args.jet_flavour == "bjet"
        and args.dimuon_sign == "OS"
    )


def make_asimov_data(total):
    """Return a background-only Asimov histogram with zero point uncertainty."""
    if not total:
        return None
    out = total.Clone(NAMES.unique("asimov_data"))
    out.SetDirectory(0)
    for ib in range(1, out.GetNbinsX() + 1):
        out.SetBinError(ib, 0.0)
    return out


def mask_blind_data(h, spec: VariableSpec, blind: bool):
    if not h or not blind or spec.key != "dimuon_mass":
        return h
    out = h.Clone(NAMES.unique("blind_data"))
    out.SetDirectory(0)
    for ib in range(1, out.GetNbinsX() + 1):
        xlo = out.GetXaxis().GetBinLowEdge(ib)
        xhi = out.GetXaxis().GetBinUpEdge(ib)
        if xhi > 10.4 and xlo < 70.0:
            out.SetBinContent(ib, 0.0)
            out.SetBinError(ib, 0.0)
    return out


def hist_max(h) -> float:
    if not h:
        return 0.0
    out = 0.0
    for ib in range(1, h.GetNbinsX() + 1):
        out = max(out, h.GetBinContent(ib) + h.GetBinError(ib))
    return out


def make_ratio(ROOT, data, total):
    ratio = data.Clone(NAMES.unique("ratio"))
    ratio.SetDirectory(0)
    for ib in range(1, ratio.GetNbinsX() + 1):
        d = data.GetBinContent(ib)
        de = data.GetBinError(ib)
        m = total.GetBinContent(ib)
        if m > 0:
            ratio.SetBinContent(ib, d / m)
            ratio.SetBinError(ib, de / m)
        else:
            ratio.SetBinContent(ib, 0.0)
            ratio.SetBinError(ib, 0.0)
    return ratio


def make_ratio_band(ROOT, total):
    band = total.Clone(NAMES.unique("ratio_band"))
    band.SetDirectory(0)
    for ib in range(1, band.GetNbinsX() + 1):
        m = total.GetBinContent(ib)
        e = total.GetBinError(ib)
        band.SetBinContent(ib, 1.0)
        band.SetBinError(ib, e / m if m > 0 else 0.0)
    return band


def draw_labels(ROOT, args) -> None:
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextFont(42)
    latex.SetTextSize(0.050)
    latex.SetTextAlign(13)
    latex.DrawLatex(0.125, 0.925, "#bf{CMS} #it{Preliminary}")
    latex.SetTextAlign(31)
    latex.SetTextSize(0.037)
    latex.DrawLatex(0.95, 0.925, lumi_label(args.era))
    latex.SetTextAlign(13)
    latex.SetTextSize(0.035)
    jtxt = "b-jet" if args.jet_flavour == "bjet" else "light-jet"
    latex.DrawLatex(0.145, 0.855, f"{args.era}, {args.dimuon_sign}, {jtxt}")


def draw_plot(ROOT, args, variable: str) -> List[str]:
    spec = VARIABLE_SPECS[variable]
    edges = make_edges(spec, args.xmin, args.xmax, args.bin_width)

    blind_asimov = use_blind_asimov(args)

    # In the blinded OS b-jet signal region, do not read the real data
    # histogram at all.  The plotted "data" points are replaced below by
    # the background-only Asimov expectation.
    data = None
    if not blind_asimov:
        data = load_combined_data(ROOT, args, spec, edges)
        if not data:
            raise RuntimeError("No data histogram loaded")

    nonqcd = load_combined_nonqcd(ROOT, args, spec, edges)
    qcd_bins = load_combined_qcd_bins(ROOT, args, spec, edges)
    if not qcd_bins:
        raise RuntimeError("No QCD pT-bin histograms loaded")

    divided = bool(args.divide_by_bin_width and spec.divide_by_bin_width)
    nf = 1.0
    effective_qcd_normalise = bool(args.qcd_normalise and not blind_asimov)

    if effective_qcd_normalise:
        nf = compute_qcd_nf(data, nonqcd, qcd_bins, divided_by_width=divided)
        apply_nf(qcd_bins, nf)
    elif args.qcd_normalise and blind_asimov:
        print(
            "[INFO] Blinded OS b-jet region: QCD data normalisation is disabled "
            "to avoid using blinded data; using QCD NF = 1."
        )

    total = total_hist(nonqcd, qcd_bins)
    if not total:
        raise RuntimeError("No total MC histogram")

    qcolors = qcd_colors(ROOT)
    ncolors = nonqcd_colors(ROOT)
    for key, h in qcd_bins.items():
        style_component(ROOT, h, qcolors[key])
    for proc, h in nonqcd.items():
        style_component(ROOT, h, ncolors[proc])

    stack = build_manual_stack(ROOT, nonqcd, qcd_bins)
    stat_band = stat_error_band(total, list(nonqcd.values()) + list(qcd_bins.values()))
    stat_band.SetFillColorAlpha(ROOT.kGray + 1, 0.30)
    stat_band.SetFillStyle(1001)
    stat_band.SetLineColor(ROOT.kGray + 2)

    if blind_asimov:
        draw_data = make_asimov_data(total)
    else:
        draw_data = mask_blind_data(data, spec, args.blind)

    draw_data.SetMarkerStyle(20)
    draw_data.SetMarkerSize(0.9)
    draw_data.SetMarkerColor(ROOT.kBlack)
    draw_data.SetLineColor(ROOT.kBlack)

    c = ROOT.TCanvas(NAMES.unique("c"), "", 900, 850)
    p1 = ROOT.TPad(NAMES.unique("p1"), "", 0.0, 0.30, 1.0, 1.0)
    p2 = ROOT.TPad(NAMES.unique("p2"), "", 0.0, 0.00, 1.0, 0.30)
    p1.SetLeftMargin(0.12)
    p1.SetRightMargin(0.05)
    p1.SetTopMargin(0.10)
    p1.SetBottomMargin(0.03)
    p2.SetLeftMargin(0.12)
    p2.SetRightMargin(0.05)
    p2.SetTopMargin(0.04)
    p2.SetBottomMargin(0.35)
    p1.Draw()
    p2.Draw()

    p1.cd()
    if args.logy:
        p1.SetLogy(True)

    ymax = max(hist_max(total), hist_max(draw_data), 1.0)
    ymin = args.ymin
    ymax_plot = args.ymax if args.ymax is not None else (ymax * 80.0 if args.logy else ymax * 1.55)
    if args.logy and ymin <= 0:
        raise ValueError("--ymin must be > 0 for log-y plots")
    if ymax_plot <= ymin:
        ymax_plot = ymin * 10.0 if args.logy else ymin + 1.0

    frame = ROOT.TH1D(NAMES.unique("frame"), "", len(edges) - 1, array("d", edges))
    frame.SetDirectory(0)
    frame.SetMinimum(ymin)
    frame.SetMaximum(ymax_plot)
    frame.GetXaxis().SetLabelSize(0.0)
    frame.GetXaxis().SetTitleSize(0.0)
    frame.GetYaxis().SetTitle("Events / bin" if not divided else "Events / unit")
    frame.GetYaxis().SetTitleSize(0.050)
    frame.GetYaxis().SetTitleOffset(1.10)
    frame.Draw("AXIS")

    for _, h in reversed(stack):
        h.Draw("hist same")
    stat_band.Draw("E2 same")
    draw_data.Draw("E1 same")
    frame.Draw("AXIS SAME")

    draw_labels(ROOT, args)

    # Many entries: use two columns and a compact legend.
    leg = ROOT.TLegend(0.43, 0.45, 0.94, 0.86)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.027)
    leg.SetNColumns(2)
    leg.AddEntry(draw_data, "Bkg Asimov" if blind_asimov else "Data", "pe")

    for qspec in reversed(QCD_BINS):
        h = qcd_bins.get(qspec.key)
        if not h:
            continue
        label = qspec.label
        if effective_qcd_normalise:
            label += f" (x {nf:.3g})"
        leg.AddEntry(h, label, "f")

    for proc in ("tt", "ST", "DY", "Others"):
        if proc in nonqcd:
            leg.AddEntry(nonqcd[proc], NONQCD_LABELS[proc], "f")
    leg.AddEntry(stat_band, "Bkg stat unc.", "f")
    leg.Draw()

    p2.cd()
    ratio = make_ratio(ROOT, draw_data, total)
    ratio_band = make_ratio_band(ROOT, total)
    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.85)
    ratio.SetLineColor(ROOT.kBlack)
    ratio.SetMarkerColor(ROOT.kBlack)
    ratio_band.SetFillColorAlpha(ROOT.kGray + 1, 0.30)
    ratio_band.SetFillStyle(1001)
    ratio_band.SetLineColor(ROOT.kGray + 2)

    rframe = ROOT.TH1D(NAMES.unique("rframe"), "", len(edges) - 1, array("d", edges))
    rframe.SetDirectory(0)
    rframe.SetMinimum(args.ratio_min)
    rframe.SetMaximum(args.ratio_max)
    rframe.GetXaxis().SetTitle(spec.x_title)
    rframe.GetXaxis().SetTitleSize(0.13)
    rframe.GetXaxis().SetLabelSize(0.11)
    rframe.GetXaxis().SetTitleOffset(1.02)
    rframe.GetYaxis().SetTitle("Data / MC")
    rframe.GetYaxis().SetTitleSize(0.11)
    rframe.GetYaxis().SetLabelSize(0.10)
    rframe.GetYaxis().SetTitleOffset(0.48)
    rframe.GetYaxis().SetNdivisions(505)
    rframe.Draw("AXIS")
    ratio_band.Draw("E2 same")
    ratio.Draw("E1 same")
    line = ROOT.TLine(edges[0], 1.0, edges[-1], 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kBlack)
    line.Draw("same")
    rframe.Draw("AXIS SAME")

    os.makedirs(args.output_dir, exist_ok=True)
    data_tag = "blind_" if args.blind else "unblind_"
    jet_tag = "BJet_" if args.jet_flavour == "bjet" else "LightJet_"
    sign_tag = f"{args.dimuon_sign}_"
    qcd_tag = "QCDPtBins_norm_" if effective_qcd_normalise else "QCDPtBins_raw_"
    log_tag = "log" if args.logy else "linear"
    base = os.path.join(
        args.output_dir,
        f"{args.era}_{spec.key}_{data_tag}{jet_tag}{sign_tag}{qcd_tag}{log_tag}",
    )

    outputs = []
    for ext in args.extensions:
        path = f"{base}.{ext}"
        c.SaveAs(path)
        outputs.append(path)
    print(f"[done] {', '.join(outputs)}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plotter_qcdseparate.py",
        description="Data/MC validation plotter with individual QCD pT-bin samples shown separately.",
    )
    p.add_argument("--base-dir", default=BASE_DIR_DEFAULT)
    p.add_argument("--era", default="Run2", help="single era, Run2, Run3, full, or Run2+3")
    p.add_argument("--variable", default="dimuon_mass", help="one variable or all")
    p.add_argument("--jet-flavour", "--jet-flavor", "--jet-mode", dest="jet_flavour", default="bjet")
    p.add_argument("--dimuon-sign", "--dilepton-sign", "--sign", dest="dimuon_sign", default="OS")
    p.add_argument(
        "--blind",
        action="store_true",
        help=(
            "blind data; for OS b-jet plots use background-only Asimov points "
            "for every variable, otherwise mask 10.4 < m_mumu < 70 GeV for dimuon_mass"
        ),
    )
    p.add_argument("--unblind", dest="blind", action="store_false")
    p.set_defaults(blind=False)

    p.add_argument("--qcd-normalise", "--qcd-normalize", dest="qcd_normalise", action="store_true")
    p.add_argument("--no-qcd-normalise", "--no-qcd-normalize", dest="qcd_normalise", action="store_false")
    p.set_defaults(qcd_normalise=True)

    p.add_argument("--xmin", type=float, default=None)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--bin-width", type=float, default=None)
    p.add_argument("--divide-by-bin-width", dest="divide_by_bin_width", action="store_true")
    p.add_argument("--no-divide-by-bin-width", dest="divide_by_bin_width", action="store_false")
    p.set_defaults(divide_by_bin_width=True)

    p.add_argument("--logy", dest="logy", action="store_true")
    p.add_argument("--linear", dest="logy", action="store_false")
    p.set_defaults(logy=True)
    p.add_argument("--ymin", type=float, default=0.5)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--ratio-min", type=float, default=0.5)
    p.add_argument("--ratio-max", type=float, default=1.5)

    p.add_argument("--output-dir", default="plots")
    p.add_argument("--extensions", default="pdf,png")
    p.add_argument("--strict", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.era = canonical_era(args.era)
    args.variable = canonical_variable(args.variable)
    args.jet_flavour = canonical_jet(args.jet_flavour)
    args.dimuon_sign = canonical_sign(args.dimuon_sign)
    args.extensions = [x.strip().lstrip(".") for x in args.extensions.split(",") if x.strip()]
    if not args.extensions:
        args.extensions = ["pdf"]

    ROOT = import_root()
    style_root(ROOT)

    if use_blind_asimov(args):
        print(
            "[INFO] --blind with OS b-jet region: replacing real data by "
            "background-only Asimov points for all variables."
        )

    variables = list(VARIABLE_SPECS) if args.variable == "all" else [args.variable]
    for variable in variables:
        # Jet_1 does not exist in the LightJet region in the current NIsoMuon selection.
        if args.jet_flavour == "lightjet" and variable.startswith("jet1_"):
            print(f"[skip] {variable}: Jet_1 is not defined in the LightJet region")
            continue
        print(f"[INFO] Drawing {variable} for {args.era}")
        draw_plot(ROOT, args, variable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


