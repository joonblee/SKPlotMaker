#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BJet OS/SS overlay: QCD MC vs Data - nonQCD BKG.

The console QCD MC OS/SS summaries use the integrated 5--9 GeV ratio and
the equal mean of the integrated 11--30 and 30--80 GeV ratios. No red horizontal
reference lines are drawn in the ratio panel. Statistical errors for the
mean are combined in quadrature for the disjoint regions. These summaries use
the original weighted yields before rebinning or bin-width normalisation. Data
summaries retain the integrated ratios and the existing blinding. No summaries
text box is drawn on the plot.

--ratio-fits {none,const,linear,both} adds conditional diagonal-statistical
fits y=c (dashed) and y=ax+b (solid) to QCD MC only, by default in 11--80 GeV.
Green lines show the fits. Two green Dashed/Solid numerical equations appear
near m=10 GeV at the top of the ratio panel, with coefficient errors in
parentheses and decimal chi-square/ndf after a comma; no sample/range header.
A JSON report is also saved. Plot binning and 5--80 GeV range are retained.
--ratio-fit-range LOW HIGH selects whole bins. Fits exclude 9--11 GeV;
data points/summaries and their blinding are unchanged. No data ratio is fitted.
No closure p-value is inferred from these ratio fits.

--bin-edges EDGE ... sets custom mass edges, e.g. 5 7 9 11 15 20 30 40 60 80.
Their endpoints define xmin/xmax unless explicitly given (then they must match).
Edges must be finite, strictly increasing and coincide with native bin edges.
Default binning is unchanged. Custom data bins overlapping the blinded interval
are hidden in full. Custom output names include _customBins.

--mc-only reads only the QCD sample selected by --qcd-file and reports
integrated OS/SS ratios, the high/low double ratio and effective counts.
With a grouped --era, --per-era draws and fits each era separately and
reports constant fits across eras for the integrated ratios.
"""

import argparse, json, math, os, sys
from array import array
from bisect import bisect_left
from pathlib import Path

import ratio_fits
from qcd_bkg_estimation import QCD_TRANSFER_LOW_WINDOW, QCD_TRANSFER_HIGH_WINDOW
from qcd_common_transfer import mc_double_ratio

RUN2 = ("2016preVFP", "2016postVFP", "2017", "2018")
RUN3 = ("2022", "2022EE", "2023", "2023BPix")
YEARS = RUN2 + RUN3
ERA_GROUPS = {**{x: (x,) for x in YEARS}, "Run2": RUN2, "Run3": RUN3, "Run2+3": YEARS, "full": YEARS}
BASE = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
OUT = "/data6/Users/joonblee/PlotMaker/plots"
LUMI = {"2016preVFP":19.52,"2016postVFP":16.81,"2017":41.48,"2018":59.83,"2022":7.9804,"2022EE":26.6717,"2023":18.064,"2023BPix":9.693}
#MASS_BINS = [0.,.5,1.,1.5,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.1,3.2,3.3,3.4,3.5,3.7,4.,4.5,5.,7.,9.,10.,11.,13.,15.,20.,25.,30.,35.,40.,50.,60.,70,80.,100.]
#MASS_BINS = [0.,.5,1.,1.5,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.1,3.2,3.3,3.4,3.5,3.7,4.,4.5,5.,7.,9.,10.,11.,14.,17.,23.,29.,35.,42.,50.,60.,70,80.,100.]
MASS_BINS = [0.,5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,11.,11.5,12.,12.5,13.,13.5,14.,14.5,15.,16.,17.,18.,19.,20.,21.,23.,25.,30.,35.,40.,50.,60.,70.,80.,90.,100.]
#MASS_BINS = [0.,.5,1.,1.5,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.1,3.2,3.3,3.4,3.5,3.7,4.,4.5,5.,9.,10.,11.,80.,100.]
NONQCD_OS = ("NIsoMuon_Top.root", "NIsoMuon_DYJets_est.root", "NIsoMuon_Others.root")
NONQCD_SS = ("NIsoMuon_Top.root", "NIsoMuon_Others.root")

class PlotError(RuntimeError): pass

def root_module():
    try: import ROOT
    except Exception as e: raise PlotError(f"Could not import PyROOT: {e}")
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetOptTitle(0)
    try: ROOT.TH1.AddDirectory(False)
    except Exception: pass
    return ROOT

def era_name(x):
    return {"run2":"Run2","run3":"Run3","run2+3":"Run2+3","run23":"Run2+3","full":"full"}.get(x.lower().replace(" ",""), x)

def years(x):
    x = era_name(x)
    if x not in ERA_GROUPS: raise PlotError(f"Unknown era: {x}")
    return ERA_GROUPS[x]

def region(sign, args): return f"{sign}_{args.muon_id}_{args.jet_id}_BJet_NIsoDimuon"
def hpath(reg, args): return f"{reg}/{args.hist_name}___{reg}"
def edir(args, y): return os.path.join(args.base_dir, y, args.trigger) if args.trigger else os.path.join(args.base_dir, y)

def open_hist(ROOT, fn, path, name, require_sumw2=False):
    if not os.path.isfile(fn): raise PlotError(f"Missing ROOT file: {fn}")
    f = ROOT.TFile.Open(fn, "READ")
    if not f or f.IsZombie(): raise PlotError(f"Could not open ROOT file: {fn}")
    h = f.Get(path)
    if not h: f.Close(); raise PlotError(f"Missing histogram: {fn}:{path}")
    if require_sumw2 and h.GetSumw2N() == 0:
        f.Close()
        raise PlotError(f"Missing stored Sumw2: {fn}:{path}")
    out = h.Clone(name); out.SetDirectory(0)
    if int(out.GetSumw2N()) == 0: out.Sumw2()
    f.Close(); return out

def load(ROOT, args, relfile, sign, prefix):
    reg = region(sign, args); p = hpath(reg, args); hs = []
    for y in years(args.era):
        hs.append(open_hist(ROOT, os.path.join(edir(args,y),relfile), p,
                            f"{prefix}_{sign}_{y}", require_sumw2=args.mc_only))
    out = hs[0].Clone(f"{prefix}_{sign}_{era_name(args.era)}"); out.SetDirectory(0)
    if int(out.GetSumw2N()) == 0: out.Sumw2()
    for h in hs[1:]: out.Add(h)
    return out

def qcd(ROOT,args,sign): return load(ROOT,args,args.qcd_file,sign,"qcd")
def data_minus_nonqcd(ROOT,args,sign):
    out = load(ROOT,args,"data.root",sign,"data")
    for f in (NONQCD_OS if sign == "OS" else NONQCD_SS): out.Add(load(ROOT,args,f,sign,"sub"), -1.)
    return out

def edges(args):
    if getattr(args, "bin_edges", None) is not None:
        return list(args.bin_edges)
    if not args.variable_binning:
        n = round((args.xmax-args.xmin)/args.bin_width)
        if n <= 0 or not math.isclose(args.xmin+n*args.bin_width,args.xmax,abs_tol=1e-8): raise PlotError("Bad uniform binning")
        return [args.xmin+i*args.bin_width for i in range(n+1)]
    e = [x for x in MASS_BINS if args.xmin <= x <= args.xmax]
    if not e or not math.isclose(e[0],args.xmin,abs_tol=1e-9): e.insert(0,args.xmin)
    if not math.isclose(e[-1],args.xmax,abs_tol=1e-9): e.append(args.xmax)
    return e

def resolve_binning(args):
    custom = args.bin_edges
    if custom is not None:
        if (len(custom) < 2 or not all(map(math.isfinite, custom))
                or any(left >= right for left, right in zip(custom, custom[1:]))):
            raise PlotError("--bin-edges requires at least two finite, strictly increasing edges")
    if args.xmin is None: args.xmin = custom[0] if custom is not None else 5.
    if args.xmax is None: args.xmax = custom[-1] if custom is not None else 80.
    if not all(map(math.isfinite, (args.xmin, args.xmax))) or args.xmin >= args.xmax:
        raise PlotError("Require finite xmin < xmax")
    if custom is not None and (not math.isclose(args.xmin, custom[0], rel_tol=0., abs_tol=1e-9)
                               or not math.isclose(args.xmax, custom[-1], rel_tol=0., abs_tol=1e-9)):
        raise PlotError("--xmin/--xmax must match the first/last custom bin edges")
    if not args.variable_binning and (not math.isfinite(args.bin_width) or args.bin_width <= 0.):
        raise PlotError("--bin-width must be finite and positive")
    return edges(args)

def rebin(h,e,name):
    native = [float(h.GetXaxis().GetBinLowEdge(i)) for i in range(1,h.GetNbinsX()+2)]
    aligned = []
    for edge in e:
        index = bisect_left(native, edge)
        matches = [i for i in (index-1,index) if 0 <= i < len(native)
                   and math.isclose(native[i],edge,rel_tol=0.,abs_tol=1e-8)]
        if not matches:
            raise PlotError(f"{h.GetName()}: edge {edge:g} GeV is not a native input bin edge")
        aligned.append(native[matches[0]])
    if any(left >= right for left,right in zip(aligned,aligned[1:])):
        raise PlotError("Requested edges do not select distinct input bin boundaries")
    out = h.Rebin(len(aligned)-1,name,array("d",aligned)); out.SetDirectory(0)
    if int(out.GetSumw2N()) == 0: out.Sumw2()
    return out

def integ(h,a,b):
    v=var=0.
    for i in range(1,h.GetNbinsX()+1):
        x=h.GetXaxis().GetBinCenter(i)
        if a < x < b: v += h.GetBinContent(i); var += h.GetBinError(i)**2
    return v, math.sqrt(max(0.,var))

def ratio_val(hos,hss,a,b):
    n,en=integ(hos,a,b); d,ed=integ(hss,a,b)
    if d <= 0: return float("nan"),float("nan")
    r=n/d; er=math.sqrt((en/d)**2+(n*ed/d**2)**2); return r,er

def equal_high_ratio(hos, hss):
    """Equal mean of two disjoint mass-region ratios and its statistical error."""
    r1, e1 = ratio_val(hos, hss, 11, 30)
    r2, e2 = ratio_val(hos, hss, 30, 80)
    if not all(math.isfinite(x) for x in (r1, e1, r2, e2)):
        return float("nan"), float("nan")
    return (r1 + r2) / 2., math.hypot(e1, e2) / 2.

def mc_integrals(hos, hss):
    windows = {"low": QCD_TRANSFER_LOW_WINDOW, "high": QCD_TRANSFER_HIGH_WINDOW}
    primitive = {}
    results = {}
    for label, (low, high) in windows.items():
        counts = {}
        for sign, hist in (("os", hos), ("ss", hss)):
            axis = hist.GetXaxis()
            for edge in (low, high):
                bin_index = axis.FindFixBin(edge + 1e-9)
                if not math.isclose(axis.GetBinLowEdge(bin_index), edge, rel_tol=0., abs_tol=1e-8):
                    raise PlotError(f"{hist.GetName()}: {edge:g} GeV is not a native bin edge")
            value, error = integ(hist, low, high)
            variance = error**2
            if not all(map(math.isfinite, (value, variance))):
                raise PlotError(f"{hist.GetName()}: non-finite yield or variance in {low:g}--{high:g} GeV")
            counts[sign] = dict(value=value, variance=variance,
                                effective_count=value**2 / variance if variance > 0 else None)
            primitive[f"mc_{sign}_{label}"] = counts[sign]
        value, error = ratio_val(hos, hss, low, high)
        measurement = dict(value=value, error=error) if math.isfinite(value) and math.isfinite(error) else None
        results[label] = dict(window=[low, high], counts=counts, ratio=measurement)
        formatted = f"{value:.7g} +/- {error:.3g}" if measurement else "n/a"
        print(f"[MC TF] R_MC({low:g}--{high:g})={formatted}; "
              + ", ".join(f"N_eff({sign.upper()})={count['effective_count']:.4g}"
                          if count["effective_count"] is not None else f"N_eff({sign.upper()})=n/a"
                          for sign, count in counts.items()))
    try:
        value, log_variance = mc_double_ratio(primitive)
    except ValueError as exc:
        results["double_ratio"] = None
        print(f"[MC TF] D=R_MC(high)/R_MC(low) unavailable ({exc})")
    else:
        error = value * math.sqrt(log_variance)
        results["double_ratio"] = dict(value=value, error=error)
        print(f"[MC TF] D=R_MC(high)/R_MC(low)={value:.7g} +/- {error:.3g}")
    if any((count["effective_count"] or 0.) < 10 for count in primitive.values()):
        print("[MC TF] Sparse weighted MC in at least one window; interpret ratio errors and fits with care.")
    return results


def write_era_summary(args, measurements):
    fits = {}
    for label in ("low", "high", "double_ratio"):
        points, used_eras = [], []
        for index, (era, summary) in enumerate(measurements.items()):
            measurement = summary[label] if label == "double_ratio" else summary[label]["ratio"]
            if measurement and measurement["error"] > 0:
                points.append((index, 0., measurement["value"], measurement["error"]))
                used_eras.append(era)
        result = ratio_fits.fit_points(points, "const")
        fits[label] = dict(result, eras=used_eras)
        if result["status"] == "ok":
            print(f"[cross-era] {label}: {result['parameters']['c']:.6g} +/- {result['errors']['c']:.3g}; "
                  f"chi2/ndf={result['chi2']:.3g}/{result['ndf']} ({len(points)}/{len(measurements)} eras)")
        else:
            print(f"[cross-era] {label}: {result['status']}")
    output = Path(args.output_dir) / f"OS_SS_EraSummary_{args.era}_{Path(args.qcd_file).stem}.json"
    payload = dict(qcd_file=args.qcd_file, eras=measurements, constant_fits=fits,
                   error_model="first-order diagonal MC statistical errors from stored Sumw2")
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"[SAVED] {output}")

def mask(h,a,b):
    for i in range(1,h.GetNbinsX()+1):
        if a < h.GetXaxis().GetBinCenter(i) < b: h.SetBinContent(i,0.); h.SetBinError(i,0.)

def mask_overlapping_bins(h,a,b):
    """Custom bins crossing a blind boundary cannot represent a full data bin."""
    for i in range(1,h.GetNbinsX()+1):
        if h.GetXaxis().GetBinLowEdge(i) < b-1e-8 and h.GetXaxis().GetBinUpEdge(i) > a+1e-8:
            h.SetBinContent(i,0.); h.SetBinError(i,0.)

def per_width(h):
    for i in range(1,h.GetNbinsX()+1):
        w=h.GetXaxis().GetBinWidth(i)
        if w>0: h.SetBinContent(i,h.GetBinContent(i)/w); h.SetBinError(i,h.GetBinError(i)/w)

def ratio_hist(num,den,name):
    out=num.Clone(name); out.SetDirectory(0); out.Reset("ICES")
    if int(out.GetSumw2N()) == 0: out.Sumw2()
    for i in range(1,num.GetNbinsX()+1):
        n,en=num.GetBinContent(i),num.GetBinError(i); d,ed=den.GetBinContent(i),den.GetBinError(i)
        if d <= 0: out.SetBinContent(i,-999.); continue
        r=n/d; er=math.sqrt((en/d)**2+(n*ed/d**2)**2); out.SetBinContent(i,r); out.SetBinError(i,er)
    return out

def lumi_label(e):
    e=era_name(e)
    if e in ("Run2+3","full"): return "138 fb^{-1} (13 TeV) + 62 fb^{-1} (13.6 TeV)"
    if e=="Run2": return "138 fb^{-1} (13 TeV)"
    if e=="Run3": return "62 fb^{-1} (13.6 TeV)"
    return f"{LUMI[e]:.1f} fb^{{-1}} ({'13.6 TeV' if e in RUN3 else '13 TeV'})"

def fmt(x):
    r,e=x
    return "n/a" if not math.isfinite(r) else f"{r:.3g} #pm {e:.2g}"

def main(argv=None):
    p=argparse.ArgumentParser(description="BJet OS/SS: QCD MC overlaid with Data - nonQCD BKG")
    p.add_argument("--era",required=True,choices=list(ERA_GROUPS)); p.add_argument("--base-dir",default=BASE); p.add_argument("--output-dir",default=OUT); p.add_argument("--trigger",default="")
    p.add_argument("--qcd-file",default="NIsoMuon_QCD_Inclusive.root",help="QCD ROOT filename in each era directory")
    p.add_argument("--mc-only",action="store_true",help="Read and plot QCD MC only")
    p.add_argument("--per-era",action="store_true",help="With --mc-only, plot group members separately and fit integrated ratios across eras")
    p.add_argument("--muon-id",default="POGMedium"); p.add_argument("--jet-id",default="tight"); p.add_argument("--hist-name",default="Dilepton_Mass")
    p.add_argument("--xmin",type=float,help="Default: 5 GeV or first custom edge"); p.add_argument("--xmax",type=float,help="Default: 80 GeV or last custom edge"); p.add_argument("--bin-width",type=float,default=1.)
    binning=p.add_mutually_exclusive_group()
    binning.add_argument("--bin-edges",type=float,nargs="+",metavar="EDGE",help="Custom mass edges; endpoints set the plot range")
    binning.add_argument("--no-variable-binning",dest="variable_binning",action="store_false"); p.set_defaults(variable_binning=True)
    p.add_argument("--logx",action="store_true"); p.add_argument("--logy",action="store_true"); p.set_defaults(logy=True); p.add_argument("--ymin",type=float); p.add_argument("--ymax",type=float); p.add_argument("--ratio-min",type=float,default=0.); p.add_argument("--ratio-max",type=float,default=5.); p.add_argument("--no-bin-width",dest="divide_by_width",action="store_false"); p.set_defaults(divide_by_width=True)
    b=p.add_mutually_exclusive_group(); b.add_argument("--blind",dest="blind",action="store_true",help="Blind data in 11<m<80 GeV (default)"); b.add_argument("--unblind",dest="blind",action="store_false",help="Show data in 11<m<80 GeV"); p.set_defaults(blind=True)
    p.add_argument("--extensions",default="pdf,png"); ratio_fits.add_arguments(p, default_range=(11.,80.)); args=p.parse_args(argv)
    if args.per_era and (not args.mc_only or len(years(args.era)) < 2):
        p.error("--per-era requires --mc-only and a grouped --era")
    if args.mc_only and not args.blind:
        p.error("--unblind applies to data overlays only")
    if args.mc_only and (Path(args.qcd_file).is_absolute() or ".." in Path(args.qcd_file).parts):
        p.error("--mc-only requires --qcd-file relative to each era directory")
    e=resolve_binning(args)
    if args.logx and args.xmin<=0: raise PlotError("--logx requires --xmin > 0")
    try: ratio_fits.fit_range(args)
    except ValueError as exc: raise PlotError(str(exc)) from exc
    ROOT=root_module()
    if args.per_era:
        measurements = {}
        for era in years(args.era):
            measurements[era] = draw_comparison(ROOT, argparse.Namespace(**dict(vars(args), era=era)), e)
        write_era_summary(args, measurements)
    else:
        draw_comparison(ROOT, args, e)


def draw_comparison(ROOT, args, e):
    qos,qss=qcd(ROOT,args,"OS"),qcd(ROOT,args,"SS")
    qhigh=equal_high_ratio(qos,qss)
    if args.mc_only:
        print(f"[input] {args.era}: QCD MC only, {args.qcd_file}")
        summary = mc_integrals(qos, qss)
        summary["sources"] = [os.path.join(edir(args, era), args.qcd_file) for era in years(args.era)]
        summary["high_equal_mean"] = (dict(value=qhigh[0], error=qhigh[1])
                                      if all(map(math.isfinite, qhigh)) else None)
        print(f"[11<m<80] QCD MC equal mean (11-30, 30-80)={fmt(qhigh).replace('#pm','+/-')}")
    else:
        dos,dss=data_minus_nonqcd(ROOT,args,"OS"),data_minus_nonqcd(ROOT,args,"SS")
        qlow=ratio_val(qos,qss,5,9); dlow=ratio_val(dos,dss,5,9); dhigh=None if args.blind else ratio_val(dos,dss,11,80)
        print(f"[5<m<9] QCD MC={fmt(qlow).replace('#pm','+/-')}, Data-nonQCD={fmt(dlow).replace('#pm','+/-')}")
        print(f"[11<m<80] QCD MC equal mean (11-30, 30-80)={fmt(qhigh).replace('#pm','+/-')}, Data-nonQCD={'blinded' if args.blind else fmt(dhigh).replace('#pm','+/-')}")
        if args.blind: mask(dos,11,80); mask(dss,11,80)
    print("[BINNING]",", ".join(f"{x:g}" for x in e))
    qos,qss=rebin(qos,e,"qos"),rebin(qss,e,"qss")
    histograms = [qos, qss]
    if not args.mc_only:
        dos,dss=rebin(dos,e,"dos"),rebin(dss,e,"dss")
        histograms.extend((dos, dss))
        if args.blind and args.bin_edges is not None:
            mask_overlapping_bins(dos,11.,80.); mask_overlapping_bins(dss,11.,80.)
        dos.SetMarkerStyle(20); dos.SetMarkerColor(ROOT.kBlack); dos.SetLineColor(ROOT.kBlack); dss.SetMarkerStyle(24); dss.SetMarkerColor(ROOT.kBlue+1); dss.SetLineColor(ROOT.kBlue+1)
    if args.divide_by_width:
        for h in histograms: per_width(h)
    qos.SetLineColor(ROOT.kRed+1); qos.SetLineWidth(3); qss.SetLineColor(ROOT.kRed+1); qss.SetLineWidth(3); qss.SetLineStyle(2)
    c=ROOT.TCanvas("c","",950,900); up=ROOT.TPad("up","",0,.30,1,1); lo=ROOT.TPad("lo","",0,0,1,.30)
    for pad in (up,lo): pad.SetLeftMargin(.12); pad.SetRightMargin(.05)
    up.SetTopMargin(.10); up.SetBottomMargin(.03); lo.SetTopMargin(.04); lo.SetBottomMargin(.35)
    if args.logx: up.SetLogx(); lo.SetLogx()
    if args.logy: up.SetLogy()
    up.Draw(); lo.Draw(); up.cd()
    maxy=max(h.GetMaximum() for h in histograms); pos=[h.GetBinContent(i) for h in histograms for i in range(1,h.GetNbinsX()+1) if h.GetBinContent(i)>0]
    min_positive = .4 * min(pos) if pos else 1e-6
    if not args.mc_only:
        min_positive = max(1e-6, min_positive) if pos else 0.
    ymin=args.ymin if args.ymin is not None else (min_positive if args.logy else 0.); ymax=args.ymax if args.ymax is not None else maxy*(100 if args.logy else 1.55)
    qos.SetMinimum(ymin); qos.SetMaximum(max(ymax,ymin*10 if args.logy else ymax)); qos.GetYaxis().SetTitle("Events / GeV" if args.divide_by_width else "Events / bin"); qos.GetXaxis().SetRangeUser(args.xmin,args.xmax); qos.GetXaxis().SetLabelSize(0)
    qos.Draw("HIST"); qss.Draw("HIST SAME")
    leg=ROOT.TLegend(.56,.78 if args.mc_only else .71,.94,.88); leg.SetBorderSize(0); leg.SetFillStyle(0); leg.AddEntry(qos,"QCD MC, OS","l"); leg.AddEntry(qss,"QCD MC, SS","l")
    if not args.mc_only:
        dos.Draw("E1P SAME"); dss.Draw("E1P SAME")
        leg.AddEntry(dos,"Data - nonQCD BKG, OS","lep"); leg.AddEntry(dss,"Data - nonQCD BKG, SS","lep")
    leg.Draw()
    tx=ROOT.TLatex(); tx.SetNDC(); tx.SetTextFont(42); tx.SetTextSize(.047); tx.DrawLatex(.12,.925,"#bf{CMS} #it{Preliminary}"); tx.SetTextAlign(31); tx.SetTextSize(.038); tx.DrawLatex(.95,.925,lumi_label(args.era)); tx.SetTextAlign(11); tx.SetTextSize(.035); tx.DrawLatex(.15,.8,"BJet category: OS vs SS")
    if args.mc_only:
        sample = Path(args.qcd_file).stem.replace("Skim_NIsoMuon_", "")
        tx.DrawLatex(.15, .74, sample.replace("_", " "))
    lo.cd(); qr=ratio_hist(qos,qss,"qr"); qr.SetLineColor(ROOT.kRed+1); qr.SetMarkerColor(ROOT.kRed+1); qr.SetMarkerStyle(20)
    if not args.mc_only:
        dr=ratio_hist(dos,dss,"dr"); dr.SetMarkerColor(ROOT.kBlack); dr.SetLineColor(ROOT.kBlack); dr.SetMarkerStyle(20)
    qr.GetYaxis().SetRangeUser(args.ratio_min,args.ratio_max); qr.GetYaxis().SetTitle("OS / SS"); qr.GetYaxis().SetTitleSize(.11); qr.GetYaxis().SetTitleOffset(.50); qr.GetYaxis().SetLabelSize(.09); qr.GetYaxis().SetNdivisions(505); qr.GetXaxis().SetTitle("m_{#mu#mu} [GeV]"); qr.GetXaxis().SetTitleSize(.13); qr.GetXaxis().SetLabelSize(.10); qr.GetXaxis().SetRangeUser(args.xmin,args.xmax)
    qr.Draw("E1P")
    if not args.mc_only: dr.Draw("E1P SAME")
    line=ROOT.TLine(args.xmin,1,args.xmax,1); line.SetLineStyle(2); line.SetLineColor(ROOT.kGray+2); line.Draw("SAME")
    qr.Draw("E1P SAME")
    if not args.mc_only: dr.Draw("E1P SAME")
    lo.SetGridy()
    fit_series = [
        ("QCD MC", ROOT.kGreen+2, ratio_fits.analyse(
            ratio_fits.histogram_points(qr), args, excluded=((9., 11.),))),
    ]
    # Anchor the left edge at m=10 GeV in either logarithmic or linear x.
    fraction = (math.log(10./args.xmin)/math.log(args.xmax/args.xmin)
                if args.logx else (10.-args.xmin)/(args.xmax-args.xmin))
    result_x = max(.14, min(.40, .12 + .83*fraction))
    fit_objects = ratio_fits.draw(ROOT, fit_series,
                                 label_box=(result_x, .74, .945, .95))
    lo.cd()
    #rleg=ROOT.TLegend(.58,.74,.94,.94); rleg.SetBorderSize(0); rleg.SetFillStyle(0); rleg.SetTextSize(.075); rleg.AddEntry(qr,"QCD MC","lep"); rleg.AddEntry(dr,"Data - nonQCD MC","lep"); rleg.Draw()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tag = f"MCOnly_{Path(args.qcd_file).stem}" if args.mc_only else ("Blind" if args.blind else "Unblind")
    tag += "_LogX" if args.logx else ""; tag += "_LogY" if args.logy else ""
    category = "QCDMC" if args.mc_only else "QCDMC_DataMinusNonQCDMC"
    base=f"OS_SS_Comparison_{era_name(args.era)}_BJet_{category}_{tag}"
    if args.bin_edges is not None: base += "_customBins"
    for ext in [x.strip().lstrip('.') for x in args.extensions.split(',') if x.strip()]: c.SaveAs(str(out/f"{base}.{ext}")); print("[SAVED]",out/f"{base}.{ext}")
    if args.ratio_fits != "none" or args.mc_only:
        extra_metadata = {"bin_edges": e} if args.bin_edges is not None else {}
        if args.mc_only:
            extra_metadata.update(qcd_file=args.qcd_file, integrated=summary)
        ratio_fits.save(out/f"{base}.json", fit_series, era=args.era, blinded=None if args.mc_only else args.blind,
                        fit_sample="QCD MC only", **extra_metadata)
    c.Close()
    return summary if args.mc_only else None

if __name__=="__main__":
    try: main()
    except PlotError as e: print(f"[ERROR] {e}",file=sys.stderr); sys.exit(2)
