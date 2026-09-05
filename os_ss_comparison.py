#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BJet OS/SS overlay: QCD MC vs Data - nonQCD MC."""

import argparse, math, os, sys
from array import array
from pathlib import Path

RUN2 = ("2016preVFP", "2016postVFP", "2017", "2018")
RUN3 = ("2022", "2022EE", "2023", "2023BPix")
YEARS = RUN2 + RUN3
ERA_GROUPS = {**{x: (x,) for x in YEARS}, "Run2": RUN2, "Run3": RUN3, "Run2+3": YEARS, "full": YEARS}
BASE = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
OUT = "/data6/Users/joonblee/PlotMaker/plots"
LUMI = {"2016preVFP":19.52,"2016postVFP":16.81,"2017":41.48,"2018":59.83,"2022":7.9804,"2022EE":26.6717,"2023":18.064,"2023BPix":9.693}
MASS_BINS = [0.,.5,1.,1.5,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.1,3.2,3.3,3.4,3.5,3.7,4.,4.5,5.,6.,7.,8.,9.,10.,11.,15.,20.,30.,40.,50.,60.,80.,100.]
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

def open_hist(ROOT, fn, path, name):
    if not os.path.isfile(fn): raise PlotError(f"Missing ROOT file: {fn}")
    f = ROOT.TFile.Open(fn, "READ")
    if not f or f.IsZombie(): raise PlotError(f"Could not open ROOT file: {fn}")
    h = f.Get(path)
    if not h: f.Close(); raise PlotError(f"Missing histogram: {fn}:{path}")
    out = h.Clone(name); out.SetDirectory(0); out.Sumw2(); f.Close(); return out

def load(ROOT, args, relfile, sign, prefix):
    reg = region(sign, args); p = hpath(reg, args); hs = []
    for y in years(args.era): hs.append(open_hist(ROOT, os.path.join(edir(args,y),relfile), p, f"{prefix}_{sign}_{y}"))
    out = hs[0].Clone(f"{prefix}_{sign}_{era_name(args.era)}"); out.SetDirectory(0); out.Sumw2()
    for h in hs[1:]: out.Add(h)
    return out

def qcd(ROOT,args,sign): return load(ROOT,args,args.qcd_file,sign,"qcd")
def data_minus_nonqcd(ROOT,args,sign):
    out = load(ROOT,args,"data.root",sign,"data")
    for f in (NONQCD_OS if sign == "OS" else NONQCD_SS): out.Add(load(ROOT,args,f,sign,"sub"), -1.)
    return out

def edges(args):
    if not args.variable_binning:
        n = round((args.xmax-args.xmin)/args.bin_width)
        if n <= 0 or not math.isclose(args.xmin+n*args.bin_width,args.xmax,abs_tol=1e-8): raise PlotError("Bad uniform binning")
        return [args.xmin+i*args.bin_width for i in range(n+1)]
    e = [x for x in MASS_BINS if args.xmin <= x <= args.xmax]
    if not e or not math.isclose(e[0],args.xmin,abs_tol=1e-9): e.insert(0,args.xmin)
    if not math.isclose(e[-1],args.xmax,abs_tol=1e-9): e.append(args.xmax)
    return e

def rebin(h,e,name):
    out = h.Rebin(len(e)-1,name,array("d",e)); out.SetDirectory(0); out.Sumw2(); return out

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

def mask(h,a,b):
    for i in range(1,h.GetNbinsX()+1):
        if a < h.GetXaxis().GetBinCenter(i) < b: h.SetBinContent(i,0.); h.SetBinError(i,0.)

def per_width(h):
    for i in range(1,h.GetNbinsX()+1):
        w=h.GetXaxis().GetBinWidth(i)
        if w>0: h.SetBinContent(i,h.GetBinContent(i)/w); h.SetBinError(i,h.GetBinError(i)/w)

def ratio_hist(num,den,name):
    out=num.Clone(name); out.SetDirectory(0); out.Reset("ICES"); out.Sumw2()
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

def main():
    p=argparse.ArgumentParser(description="BJet OS/SS: QCD MC overlaid with Data - nonQCD MC")
    p.add_argument("--era",required=True,choices=list(ERA_GROUPS)); p.add_argument("--base-dir",default=BASE); p.add_argument("--output-dir",default=OUT); p.add_argument("--trigger",default="")
    p.add_argument("--qcd-file",default="NIsoMuon_QCD_Inclusive.root"); p.add_argument("--muon-id",default="POGMedium"); p.add_argument("--jet-id",default="tight"); p.add_argument("--hist-name",default="Dilepton_Mass")
    p.add_argument("--xmin",type=float,default=2.); p.add_argument("--xmax",type=float,default=100.); p.add_argument("--bin-width",type=float,default=1.); p.add_argument("--no-variable-binning",dest="variable_binning",action="store_false"); p.set_defaults(variable_binning=True)
    p.add_argument("--logx",action="store_true"); p.add_argument("--logy",action="store_true"); p.add_argument("--ymin",type=float); p.add_argument("--ymax",type=float); p.add_argument("--ratio-min",type=float,default=0.); p.add_argument("--ratio-max",type=float,default=4.); p.add_argument("--no-bin-width",dest="divide_by_width",action="store_false"); p.set_defaults(divide_by_width=True)
    b=p.add_mutually_exclusive_group(); b.add_argument("--blind",dest="blind",action="store_true",help="Blind data in 11<m<80 GeV (default)"); b.add_argument("--unblind",dest="blind",action="store_false",help="Show data in 11<m<80 GeV"); p.set_defaults(blind=True)
    p.add_argument("--extensions",default="pdf,png"); args=p.parse_args()
    if args.xmax<=args.xmin: raise PlotError("--xmax must be larger than --xmin")
    if args.logx and args.xmin<=0: raise PlotError("--logx requires --xmin > 0")
    ROOT=root_module()
    qos,qss=qcd(ROOT,args,"OS"),qcd(ROOT,args,"SS"); dos,dss=data_minus_nonqcd(ROOT,args,"OS"),data_minus_nonqcd(ROOT,args,"SS")
    qlow,qhigh=ratio_val(qos,qss,6,9),ratio_val(qos,qss,11,80); dlow=ratio_val(dos,dss,6,9); dhigh=None if args.blind else ratio_val(dos,dss,11,80)
    print(f"[6<m<9] QCD MC={fmt(qlow).replace('#pm','+/-')}, Data-nonQCD={fmt(dlow).replace('#pm','+/-')}")
    print(f"[11<m<80] QCD MC={fmt(qhigh).replace('#pm','+/-')}, Data-nonQCD={'blinded' if args.blind else fmt(dhigh).replace('#pm','+/-')}")
    if args.blind: mask(dos,11,80); mask(dss,11,80)
    e=edges(args); print("[BINNING]",", ".join(f"{x:g}" for x in e))
    qos,qss,dos,dss=(rebin(qos,e,"qos"),rebin(qss,e,"qss"),rebin(dos,e,"dos"),rebin(dss,e,"dss"))
    if args.divide_by_width:
        for h in (qos,qss,dos,dss): per_width(h)
    qos.SetLineColor(ROOT.kRed+1); qos.SetLineWidth(3); qss.SetLineColor(ROOT.kRed+1); qss.SetLineWidth(3); qss.SetLineStyle(2)
    dos.SetMarkerStyle(20); dos.SetMarkerColor(ROOT.kBlack); dos.SetLineColor(ROOT.kBlack); dss.SetMarkerStyle(24); dss.SetMarkerColor(ROOT.kBlue+1); dss.SetLineColor(ROOT.kBlue+1)
    c=ROOT.TCanvas("c","",950,900); up=ROOT.TPad("up","",0,.30,1,1); lo=ROOT.TPad("lo","",0,0,1,.30)
    for pad in (up,lo): pad.SetLeftMargin(.12); pad.SetRightMargin(.05)
    up.SetTopMargin(.10); up.SetBottomMargin(.03); lo.SetTopMargin(.04); lo.SetBottomMargin(.35)
    if args.logx: up.SetLogx(); lo.SetLogx()
    if args.logy: up.SetLogy()
    up.Draw(); lo.Draw(); up.cd()
    maxy=max(h.GetMaximum() for h in (qos,qss,dos,dss)); pos=[h.GetBinContent(i) for h in (qos,qss,dos,dss) for i in range(1,h.GetNbinsX()+1) if h.GetBinContent(i)>0]
    ymin=args.ymin if args.ymin is not None else (max(1e-6,.4*min(pos)) if args.logy and pos else 0.); ymax=args.ymax if args.ymax is not None else maxy*(100 if args.logy else 1.55)
    qos.SetMinimum(ymin); qos.SetMaximum(max(ymax,ymin*10 if args.logy else ymax)); qos.GetYaxis().SetTitle("Events / GeV" if args.divide_by_width else "Events / bin"); qos.GetXaxis().SetRangeUser(args.xmin,args.xmax); qos.GetXaxis().SetLabelSize(0)
    qos.Draw("HIST"); qss.Draw("HIST SAME"); dos.Draw("E1P SAME"); dss.Draw("E1P SAME")
    leg=ROOT.TLegend(.56,.67,.94,.86); leg.SetBorderSize(0); leg.SetFillStyle(0); leg.AddEntry(qos,"QCD MC, OS","l"); leg.AddEntry(qss,"QCD MC, SS","l"); leg.AddEntry(dos,"Data - nonQCD MC, OS","lep"); leg.AddEntry(dss,"Data - nonQCD MC, SS","lep"); leg.Draw()
    tx=ROOT.TLatex(); tx.SetNDC(); tx.SetTextFont(42); tx.SetTextSize(.047); tx.DrawLatex(.12,.925,"#bf{CMS} #it{Preliminary}"); tx.SetTextAlign(31); tx.SetTextSize(.038); tx.DrawLatex(.95,.925,lumi_label(args.era)); tx.SetTextAlign(11); tx.SetTextSize(.035); tx.DrawLatex(.15,.8,"BJet category: OS vs SS")
    box=ROOT.TPaveText(.13,.07,.8,.45,"NDC"); box.SetBorderSize(0); box.SetFillStyle(0); box.SetTextAlign(12); box.SetTextSize(.03); box.AddText("Integrated OS/SS"); box.AddText("6<m_{{#mu#mu}}<9:"); box.AddText(f"  QCD MC = {fmt(qlow)}"); box.AddText(f"  Data = {fmt(dlow)}"); box.AddText("11<m_{{#mu#mu}}<80:"); box.AddText(f"  QCD MC ={fmt(qhigh)}"); box.AddText(f"  Data = blinded" if args.blind else f"  Data = {fmt(dhigh)}"); box.Draw()
    lo.cd(); qr=ratio_hist(qos,qss,"qr"); dr=ratio_hist(dos,dss,"dr"); qr.SetLineColor(ROOT.kRed+1); qr.SetMarkerColor(ROOT.kRed+1); qr.SetMarkerStyle(20); dr.SetMarkerColor(ROOT.kBlack); dr.SetLineColor(ROOT.kBlack); dr.SetMarkerStyle(20)
    qr.GetYaxis().SetRangeUser(args.ratio_min,args.ratio_max); qr.GetYaxis().SetTitle("OS / SS"); qr.GetYaxis().SetTitleSize(.11); qr.GetYaxis().SetTitleOffset(.50); qr.GetYaxis().SetLabelSize(.09); qr.GetYaxis().SetNdivisions(505); qr.GetXaxis().SetTitle("m_{#mu#mu} [GeV]"); qr.GetXaxis().SetTitleSize(.13); qr.GetXaxis().SetLabelSize(.10); qr.GetXaxis().SetRangeUser(args.xmin,args.xmax)
    qr.Draw("E1P"); dr.Draw("E1P SAME"); line=ROOT.TLine(args.xmin,1,args.xmax,1); line.SetLineStyle(2); line.SetLineColor(ROOT.kGray+2); line.Draw("SAME"); qr.Draw("E1P SAME"); dr.Draw("E1P SAME"); lo.SetGridy()
    #rleg=ROOT.TLegend(.58,.74,.94,.94); rleg.SetBorderSize(0); rleg.SetFillStyle(0); rleg.SetTextSize(.075); rleg.AddEntry(qr,"QCD MC","lep"); rleg.AddEntry(dr,"Data - nonQCD MC","lep"); rleg.Draw()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); tag="Blind" if args.blind else "Unblind"; tag += "_LogX" if args.logx else ""; tag += "_LogY" if args.logy else ""; base=f"OS_SS_Comparison_{era_name(args.era)}_BJet_QCDMC_DataMinusNonQCDMC_{tag}"
    for ext in [x.strip().lstrip('.') for x in args.extensions.split(',') if x.strip()]: c.SaveAs(str(out/f"{base}.{ext}")); print("[SAVED]",out/f"{base}.{ext}")

if __name__=="__main__":
    try: main()
    except PlotError as e: print(f"[ERROR] {e}",file=sys.stderr); sys.exit(2)
