#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained NIsoMuon muon-ID efficiency and scale-factor measurement.

This file contains the complete up-to-date id_eff.C implementation embedded
below.  No separately maintained id_eff.C file is required.  ROOT ACLiC is
used automatically through a hidden cache to avoid the tamsa Cling JIT
materialization failure seen with TCanvas/TPad.

Fixed input layout
------------------
  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon/
    MuonIDEfficiency/<era>/data.root
    MuonIDEfficiency/<era>/NIsoMuon_QCD_Inclusive.root
    MuonIDEfficiency/<era>/NIsoMuon_Top.root

Fixed output layout
-------------------
  /data6/Users/joonblee/PlotMaker/plots/MuonIDEfficiency/<era>/

The embedded C++ preserves the full supplied implementation:
  * J/psi and Z resonance modes;
  * CB and DSCB signal models;
  * all Exp/Cheb/Bern and monotonic background models;
  * sideband background prefit and final signal+background fit;
  * integral and fit-normalisation yield modes;
  * eta-dependent pT binning and eta-integrated pT-only binning;
  * reconstruction of coarse bins from analyzer pT/eta inputs;
  * Probe_Pt-based treatment of partially overlapping pT bins;
  * per-bin fit plots, summary CSV, and final efficiency/SF panels;
  * input-key inspection, bin filters, quick checks, and reference selection.

Examples
--------
  python3 id_eff.py --year 2023 --inspect-hists
  python3 id_eff.py --year 2023 --quick-check
  python3 id_eff.py --year 2023
  python3 id_eff.py --year 2023 --reference Top
  python3 id_eff.py --year 2023 --binning pt-only
  python3 id_eff.py --year 2023 --yield-mode fitnorm
  python3 id_eff.py --year all

Run this with python3.  Do not use `source id_eff.py`.
Running with no arguments prints the full option page and performs no fit.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence, Tuple


VALID_ERAS: Tuple[str, ...] = (
    "2016preVFP",
    "2016postVFP",
    "2017",
    "2018",
    "2022",
    "2022EE",
    "2023",
    "2023BPix",
)

INPUT_BASE = Path(
    "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
)
INPUT_COLLECTION = "MuonIDEfficiency"
PLOT_BASE = Path("/data6/Users/joonblee/PlotMaker/plots/MuonIDEfficiency")
CACHE_BASE = Path("/data6/Users/joonblee/PlotMaker/.cache_id_eff")


CPP_SOURCE = r"""
// id_eff.C
// -----------------------------------------------------------------------------
// Robust tag-and-probe fit macro for NIsoMuon muon-ID efficiency/SF using J/psi or Z peaks.
//
// Save as id_eff.C before running if desired:
//   cp Fit_JpsiMuonIDEff_6.C id_eff.C
//
// Default:
//   root -l -b -q 'id_eff.C("2018")'
//
// Useful variations:
//   // stronger rebinning
//   root -l -b -q 'id_eff.C("2018","HighPtMuon","/data6/Users/joonblee/SKFlatOutput/Run2UltraLegacy_v3","NIsoMuon","OS_POGMedium_tight_BJet_MuonIDEfficiency",5)'
//
//   // alternative positive-definite backgrounds
//   root -l -b -q 'id_eff.C("2018","HighPtMuon","/data6/Users/joonblee/SKFlatOutput/Run2UltraLegacy_v3","NIsoMuon","OS_POGMedium_tight_BJet_MuonIDEfficiency",3,"CB","Bern7")'
//   root -l -b -q 'id_eff.C("2018","HighPtMuon","/data6/Users/joonblee/SKFlatOutput/Run2UltraLegacy_v3","NIsoMuon","OS_POGMedium_tight_BJet_MuonIDEfficiency",3,"CB","Exp3")'
//
//   // stronger constraint on the [3.3,3.5] GeV sideband
//   root -l -b -q 'id_eff.C("2018","HighPtMuon","/data6/Users/joonblee/SKFlatOutput/Run2UltraLegacy_v3","NIsoMuon","OS_POGMedium_tight_BJet_MuonIDEfficiency",2,"CB","Bern7",2.70,3.50,true,true,6.0)' 
//
//   // old-style signal-shape cross-check
//   root -l -b -q 'id_eff.C("2018","HighPtMuon","/data6/Users/joonblee/SKFlatOutput/Run2UltraLegacy_v3","NIsoMuon","OS_POGMedium_tight_BJet_MuonIDEfficiency",3,"DSCB","Exp1")'
//
// Main changes relative to the first version:
//   1. Rebin before fitting, default factor = 3.
//   2. Fit density histograms, Events/GeV, so the signal parameter Nsig is directly
//      the J/psi yield.  This avoids unstable TF1::IntegralError calls.
//   3. The continuum background is first fitted only in sidebands:
//        2.0--2.8, 3.3--3.5, and 3.8--5.0 GeV.
//      The J/psi and psi(2S) peak regions are excluded from this bkg-only fit.
//   4. AN-compatible nominal fit: signal+background is fitted over 2.0--5.0 GeV
//      with the psi(2S) region vetoed, using the sideband-fitted background shape.
//   5. The nominal AN model is Crystal Ball + positive Bernstein-5 background.
//      Other selectable backgrounds remain available for cross-checks.
//   6. Pass and Fail are fitted separately, as in the AN.  The fitted J/psi
//      signal normalisations P and F give efficiency = P/(P+F).  The nominal
//      statistical uncertainty is the Gaussian propagation of the fitted P/F
//      yield uncertainties.  Rebin=3 (30 MeV) is central; complete refits with
//      rebin=1,2,4,5 provide a symmetric mass-binning envelope.  Statistical
//      and mass-binning components are added in quadrature.
//   7. data.root in the era directory is the current data-input convention.
//   8. Histograms are read from the current DileptonJPsi_Mass output directly.
//      No automatic Dilepton_Mass fallback is used.
//   9. Final summary plots are drawn as two-panel efficiency/SF canvases vs pT and vs |eta|.
//  10. Output pT binning starts from 10,20,30,50,120 GeV in the barrel and
//      is progressively merged at larger |eta|:
//        |eta|=[0.0,0.9]: 10,20,30,50,120 GeV;
//        |eta|=[0.9,1.2]: 10,30,50,120 GeV;
//        |eta|=[1.2,2.1]: 10,30,120 GeV;
//        |eta|=[2.1,2.4]: 10,120 GeV.
//      All retained edges match analyzer input boundaries, avoiding fractional splitting.
//      If a requested coarse histogram is not present, it is built from the
//      current analyzer pT-binned histograms using the analyzer ptEdges.
//  11. In the final signal+background fit, the background normalisation is constrained
//      to +/-5% around the sideband-prefit value; other bkg-shape parameters can be
//      fixed from the sidebands with FixBkgShapeFromSidebands.
//
// Expected direct output-bin histogram path:
//   <BaseRegion>_<binTag>_<Pass|Fail>/<HistName>___<BaseRegion>_<binTag>_<Pass|Fail>
// HistName is DileptonJPsi_Mass for --resonance Jpsi and Dilepton_Mass for --resonance Z
// unless the user overrides it. If this exact output bin is missing, the macro builds it
// from the analyzer input pT/eta bins with pT edges 10,15,20,25,30,40,50,60,120,200,2000 GeV.
// There is no implicit Dilepton_Mass fallback in Jpsi mode; use --hist-name or --resonance Z explicitly.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <utility>

#include "TArrayD.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TDirectory.h"
#include "TBox.h"
#include "TFile.h"
#include "TF1.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TKey.h"
#include "TLatex.h"
#include "TLine.h"
#include "TMath.h"
#include "TPad.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;

namespace JpsiMuonIDFit {

  enum SignalModel { kCB = 0, kDSCB = 1 };
  enum BackgroundModel {
    kExp1 = 0, kExp2, kExp3, kExp4,
    kMonoExp1, kMonoExp2, kMonoExp3, kMonoExp4,
    kCheb1, kCheb2, kCheb3, kCheb4,
    kMonoCheb1, kMonoCheb2, kMonoCheb3, kMonoCheb4,
    kBern1, kBern2, kBern3, kBern4, kBern5, kBern6, kBern7, kBern8,
    kMonoBern1, kMonoBern2, kMonoBern3, kMonoBern4,
    kMonoBern5, kMonoBern6, kMonoBern7, kMonoBern8
  };

  enum ResonanceMode { kResJpsi = 0, kResZ = 1 };
  enum BinningMode { kBinningEtaPt = 0, kBinningPtOnly = 1 };

  SignalModel gSignalModel = kCB;
  BackgroundModel gBackgroundModel = kBern5;
  ResonanceMode gResonanceMode = kResJpsi;
  BinningMode gBinningMode = kBinningEtaPt;

  TString gResonanceLabel = "J/#psi";
  TString gResonanceShort = "Jpsi";
  TString gDefaultHistName = "DileptonJPsi_Mass";
  double gPeakMass = 3.0969;

  // Final signal+background fit window around the selected resonance peak.
  double gFitMin = 2.00;
  double gFitMax = 5.00;

  // Background-only sideband fit window and sideband definitions.
  // The continuum background is fitted only in:
  //   2.0--2.8, 3.3--3.5, and 3.8--5.0 GeV.
  const double kBkgFitMin   = 2.00;
  const double kBkgFitMax   = 5.00;
  const double kSide1Low    = 2.00;
  const double kSide1High   = 2.80;
  const double kJpsiVetoLow = 2.80;
  const double kJpsiVetoHigh= 3.30;
  const double kSide2Low    = 3.30;
  const double kSide2High   = 3.50;
  const double kPsiPVetoLow = 3.50;
  const double kPsiPVetoHigh= 3.80;

  // J/psi signal yield used for efficiencies/SFs.
  // It is evaluated from the fitted signal function in this fixed core window,
  // reducing dependence on the signal-tail convention of CB vs DSCB.
  const double kYieldIntLow  = 3.00;
  const double kYieldIntHigh = 3.20;

  const double kSide3Low    = 3.80;
  const double kSide3High   = 5.00;

  // Dynamic ranges configured by ConfigureResonance(...).  Defaults are J/psi.
  double gBkgFitMin = kBkgFitMin;
  double gBkgFitMax = kBkgFitMax;
  double gYieldIntLow = kYieldIntLow;
  double gYieldIntHigh = kYieldIntHigh;
  TString gYieldMode = "fitnorm"; // "integral" or "fitnorm"

  bool UseFitNormYield() {
    TString key = gYieldMode;
    key.ToLower();
    key.ReplaceAll("_", "");
    key.ReplaceAll("-", "");
    return (key == "fitnorm" || key == "norm" || key == "normalisation" || key == "normalization");
  }
  double gFinalVetoLow = kPsiPVetoLow;
  double gFinalVetoHigh = kPsiPVetoHigh;
  bool gUseFinalVeto = true;
  double gSignalSigmaInit = 0.045;
  double gSignalSigmaMin = 0.010;
  double gSignalSigmaMax = 0.120;
  double gMeanFitLow = 3.04;
  double gMeanFitHigh = 3.16;
  vector<std::pair<double, double> > gBkgSidebands = {
    std::make_pair(kSide1Low, kSide1High),
    std::make_pair(kSide2Low, kSide2High),
    std::make_pair(kSide3Low, kSide3High)
  };

  // Chebychev backgrounds are always defined over the active sideband-fit domain,
  // so a shape fitted in the sidebands can be reused in the final resonance fit.
  double gBkgShapeMin = kBkgFitMin;
  double gBkgShapeMax = kBkgFitMax;

  // True only while the final signal+background fit is running.
  // For J/psi this excludes the psi(2S) region [3.5,3.8] if it overlaps.
  // For Z it is disabled by ConfigureResonance(...).
  bool gRejectPsiPInFinalFit = false;

  // Pass/Fail simultaneous-fit bookkeeping.  The final fit uses one combined
  // histogram with the Fail spectrum shifted directly after the Pass spectrum.
  // GenericSimultaneousModel maps the shifted Fail coordinate back to dimuon mass.
  double gSimultaneousOffset = 0.;
  double gSimultaneousSplit = 0.;
  double gSimultaneousBinWidth = 1.;

  // The second sideband is the nearest sideband to the J/psi peak on the high-mass side.
  // It is short and otherwise gets diluted by the long sidebands, so it is upweighted
  // in the background-only prefit.
  double gSide2FitWeight = 4.0;
  bool gUseLogBkgFit = true;
  double gMinBkgRelErr = 0.015;

  const double kBkgNormRelConstraint = 0.05;  // final S+B bkg norm is limited to sideband value +/-5%
  bool gIncludeInclusive = false;      // Inclusive is optional because the analyzer often does not fill it

  struct BinDef {
    TString tag;
    double ptLow;
    double ptHigh;
    double etaLow;
    double etaHigh;
    bool inclusive;
  };

  struct BkgOutput {
    bool ok = false;
    double chi2 = 0.;
    int ndf = 0;
    int nPoints = 0;
    int fitStatus = 999;
    int covStatus = -1;
    vector<double> pars;
    vector<double> errs;
  };

  struct FitOutput {
    bool ok = false;
    bool usedCommonShape = false;
    double yield = 0.;      // Integral of fitted signal function in [kYieldIntLow,kYieldIntHigh]
    double yieldErr = 0.;
    double rawNorm = 0.;     // Fitted signal normalisation parameter, kept for diagnostics only
    double rawNormErr = 0.;
    double mean = 0.;
    double sigma = 0.;
    double chi2 = 0.;
    int ndf = 0;
    int fitStatus = 999;
    int covStatus = -1;
    double eventsInFitRange = 0.;
    vector<double> pars;
    vector<double> errs;
  };

  struct EffOutput {
    bool ok = false;
    bool hasMinos = false;
    bool binningOk = false;
    double eff = 0.;
    double err = 0.;          // final symmetric error = sqrt(stat^2 + binning^2)
    double statErr = 0.;      // nominal-fit statistical uncertainty
    double binningErr = 0.;   // envelope from rebin=1,2 relative to nominal rebin=3
    double fitErr = 0.;       // parabolic/Hessian uncertainty, diagnostic/fallback
    double minosLow = 0.;     // signed lower Minos error
    double minosHigh = 0.;    // signed upper Minos error
    double rebin1Eff = 0.;
    double rebin2Eff = 0.;
    double rebin4Eff = 0.;
    double rebin5Eff = 0.;
    bool rebin1Ok = false;
    bool rebin2Ok = false;
    bool rebin4Ok = false;
    bool rebin5Ok = false;
    double wilsonErr = 0.;    // separate-fit emergency fallback only
    double nCount = 0.;       // fitted total signal normalisation
    double wilsonLow = 0.;
    double wilsonHigh = 0.;
    int fitStatus = 999;
    int covStatus = -1;
    double chi2 = 0.;         // 2*NLL for Data, chi2 for weighted MC/reference
    int ndf = 0;
  };

  struct SimultaneousOutput {
    bool ok = false;
    FitOutput pass;
    FitOutput fail;
    EffOutput eff;
  };

  struct SummaryRow {
    BinDef bin;
    BkgOutput dataAllBkg;
    BkgOutput dataPassBkg;
    BkgOutput dataFailBkg;
    BkgOutput qcdAllBkg;
    BkgOutput qcdPassBkg;
    BkgOutput qcdFailBkg;
    FitOutput dataAll;
    FitOutput dataPass;
    FitOutput dataFail;
    FitOutput qcdAll;
    FitOutput qcdPass;
    FitOutput qcdFail;
    EffOutput dataEff;
    EffOutput qcdEff;
    bool sfOk = false;
    double sf = 0.;
    double sfErr = 0.;
    double sfStatErr = 0.;
    double sfBinningErr = 0.;
    double sfRebin1 = 0.;
    double sfRebin2 = 0.;
    double sfRebin4 = 0.;
    double sfRebin5 = 0.;
    bool sfBinningOk = false;
  };

  TString EdgeLabel(double x) {
    TString out = Form("%.1f", x);
    out.ReplaceAll(".", "p");
    out.ReplaceAll("-", "m");
    return out;
  }

  TString Sanitise(TString s) {
    s.ReplaceAll("/", "_");
    s.ReplaceAll(" ", "_");
    s.ReplaceAll("#", "");
    s.ReplaceAll("{", "");
    s.ReplaceAll("}", "");
    s.ReplaceAll("(", "_");
    s.ReplaceAll(")", "_");
    s.ReplaceAll("+", "p");
    s.ReplaceAll("-", "m");
    s.ReplaceAll(".", "p");
    s.ReplaceAll(",", "_");
    return s;
  }

  bool NearlyEqual(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
  }

  vector<double> PtEdgesForEta(double etaLow, double etaHigh) {
    // Start from the 10,20,30,50,120 GeV reference binning in the barrel and
    // merge bins progressively at larger |eta| where the tag-and-probe sample
    // is statistically limited.  All retained edges coincide with exact
    // NIsoMuon analyzer input boundaries, so no fractional pT-bin splitting is
    // needed for the standard output bins.
    if(NearlyEqual(etaLow, 0.0) && NearlyEqual(etaHigh, 0.9)) {
      return {10., 20., 30., 50., 120.};
    }
    if(NearlyEqual(etaLow, 0.9) && NearlyEqual(etaHigh, 1.2)) {
      return {10., 30., 50., 120.};
    }
    if(NearlyEqual(etaLow, 1.2) && NearlyEqual(etaHigh, 2.1)) {
      return {10., 30., 120.};
    }
    if(NearlyEqual(etaLow, 2.1) && NearlyEqual(etaHigh, 2.4)) {
      return {10., 120.};
    }

    cout << "[ERROR] Unsupported eta bin for PtEdgesForEta: [" << etaLow << ", " << etaHigh << "]" << endl;
    return {};
  }

  vector<double> PtOnlyEdges() {
    // pT-only output mode uses the same barrel-base binning.
    return {10., 20., 30., 50., 120.};
  }

  vector<double> DefaultEtaEdges() {
    return {0., 0.9, 1.2, 2.1, 2.4};
  }

  // These are the pT edges actually used by NIsoMuon::MuonIDEfficiency when
  // filling the input ROOT histograms.  They are used only to reconstruct the
  // requested output bins above; they are not the plotted efficiency bins.
  vector<double> AnalyzerInputPtEdges() {
    return {10., 15., 20., 25., 30., 40., 50., 60., 120., 200., 2000.};
  }

  TString BinTag(double ptLow, double ptHigh, double etaLow, double etaHigh) {
    return TString("Pt") + EdgeLabel(ptLow) + TString("to") + EdgeLabel(ptHigh)
         + TString("_AbsEta") + EdgeLabel(etaLow) + TString("to") + EdgeLabel(etaHigh);
  }

  bool IsFullEtaRange(double etaLow, double etaHigh) {
    return NearlyEqual(etaLow, 0.0) && NearlyEqual(etaHigh, 2.4);
  }

  TString NormaliseKey(TString s);

  BinningMode ParseBinningMode(TString s) {
    TString key = NormaliseKey(s);
    if(key == "pt" || key == "ptonly" || key == "ptonlybinning" || key == "noeta" || key == "integratedeta") return kBinningPtOnly;
    return kBinningEtaPt;
  }

  TString BinningModeName(BinningMode mode) {
    return (mode == kBinningPtOnly) ? "pt-only (eta-integrated)" : "eta-pt";
  }

  TString BinningModeShort(BinningMode mode) {
    return (mode == kBinningPtOnly) ? "ptonly" : "etapt";
  }

  vector<BinDef> MakeBins() {
    vector<BinDef> bins;
    if(gIncludeInclusive) bins.push_back({"Inclusive", 10., 120., 0., 2.4, true});

    if(gBinningMode == kBinningPtOnly) {
      const vector<double> ptEdges = PtOnlyEdges();
      for(unsigned int ipt = 0; ipt + 1 < ptEdges.size(); ++ipt) {
        const TString tag = BinTag(ptEdges[ipt], ptEdges[ipt+1], 0., 2.4);
        bins.push_back({tag, ptEdges[ipt], ptEdges[ipt+1], 0., 2.4, false});
      }
      return bins;
    }

    const vector<double> etaEdges = DefaultEtaEdges();
    for(unsigned int ieta = 0; ieta + 1 < etaEdges.size(); ++ieta) {
      const vector<double> ptEdges = PtEdgesForEta(etaEdges[ieta], etaEdges[ieta+1]);
      for(unsigned int ipt = 0; ipt + 1 < ptEdges.size(); ++ipt) {
        const TString tag = BinTag(ptEdges[ipt], ptEdges[ipt+1], etaEdges[ieta], etaEdges[ieta+1]);
        bins.push_back({tag, ptEdges[ipt], ptEdges[ipt+1], etaEdges[ieta], etaEdges[ieta+1], false});
      }
    }
    return bins;
  }

  TString ResonanceKey(TString s) {
    TString key = NormaliseKey(s);
    if(key == "z" || key == "zpeak" || key == "zmumu" || key == "zboson") return "Z";
    return "Jpsi";
  }

  void ConfigureResonance(TString resonanceInput,
                          double &fitMin, double &fitMax,
                          TString &histName,
                          double yieldIntLowInput,
                          double yieldIntHighInput,
                          double bkgFitMinInput,
                          double bkgFitMaxInput) {
    const TString key = ResonanceKey(resonanceInput);
    gBkgSidebands.clear();

    if(key == "Z") {
      gResonanceMode = kResZ;
      gResonanceLabel = "Z";
      gResonanceShort = "Z";
      gDefaultHistName = "Dilepton_Mass";
      gPeakMass = 91.1876;

      // If the user did not override the J/psi defaults, switch to Z defaults.
      if((NearlyEqual(fitMin, 2.70) && NearlyEqual(fitMax, 3.50)) ||
         (NearlyEqual(fitMin, 2.00) && NearlyEqual(fitMax, 5.00))) {
        fitMin = 70.0;
        fitMax = 110.0;
      }
      gBkgFitMin = (bkgFitMinInput < bkgFitMaxInput) ? bkgFitMinInput : 60.0;
      gBkgFitMax = (bkgFitMinInput < bkgFitMaxInput) ? bkgFitMaxInput : 120.0;
      gYieldIntLow = (yieldIntLowInput < yieldIntHighInput) ? yieldIntLowInput : 80.0;
      gYieldIntHigh = (yieldIntLowInput < yieldIntHighInput) ? yieldIntHighInput : 100.0;
      gUseFinalVeto = false;
      gFinalVetoLow = 0.;
      gFinalVetoHigh = 0.;
      gSignalSigmaInit = 2.0;
      gSignalSigmaMin = 0.4;
      gSignalSigmaMax = 8.0;
      gMeanFitLow = 86.0;
      gMeanFitHigh = 96.0;
      gBkgSidebands.push_back(std::make_pair(gBkgFitMin, 80.0));
      gBkgSidebands.push_back(std::make_pair(100.0, gBkgFitMax));
    }
    else {
      gResonanceMode = kResJpsi;
      gResonanceLabel = "J/#psi";
      gResonanceShort = "Jpsi";
      gDefaultHistName = "DileptonJPsi_Mass";
      gPeakMass = 3.0969;
      gBkgFitMin = (bkgFitMinInput < bkgFitMaxInput) ? bkgFitMinInput : kBkgFitMin;
      gBkgFitMax = (bkgFitMinInput < bkgFitMaxInput) ? bkgFitMaxInput : kBkgFitMax;
      gYieldIntLow = (yieldIntLowInput < yieldIntHighInput) ? yieldIntLowInput : kYieldIntLow;
      gYieldIntHigh = (yieldIntLowInput < yieldIntHighInput) ? yieldIntHighInput : kYieldIntHigh;
      gUseFinalVeto = true;
      gFinalVetoLow = kPsiPVetoLow;
      gFinalVetoHigh = kPsiPVetoHigh;
      gSignalSigmaInit = 0.045;
      gSignalSigmaMin = 0.010;
      gSignalSigmaMax = 0.120;
      gMeanFitLow = 3.04;
      gMeanFitHigh = 3.16;
      gBkgSidebands.push_back(std::make_pair(kSide1Low, kSide1High));
      gBkgSidebands.push_back(std::make_pair(kSide2Low, kSide2High));
      gBkgSidebands.push_back(std::make_pair(kSide3Low, kSide3High));
    }

    if(histName == "" || NormaliseKey(histName) == "auto") histName = gDefaultHistName;
    gFitMin = fitMin;
    gFitMax = fitMax;
    gBkgShapeMin = gBkgFitMin;
    gBkgShapeMax = gBkgFitMax;
  }

  TString SidebandRangesText() {
    if(gBkgSidebands.empty()) return "none";
    TString out;
    for(unsigned int i = 0; i < gBkgSidebands.size(); ++i) {
      if(i > 0) out += ", ";
      out += Form("[%.2g,%.2g]", gBkgSidebands[i].first, gBkgSidebands[i].second);
    }
    return out;
  }

  TString LumiText(const TString &year) {
    if      (year == "2016preVFP")  return "19.5 fb^{-1} (13 TeV)";
    else if (year == "2016postVFP") return "16.8 fb^{-1} (13 TeV)";
    else if (year == "2016")        return "36.31 fb^{-1} (13 TeV)";
    else if (year == "2017")        return "42.07 fb^{-1} (13 TeV)";
    else if (year == "2018")        return "59.56 fb^{-1} (13 TeV)";
    else if (year == "2022")        return "7.98 fb^{-1} (13.6 TeV)";
    else if (year == "2022EE")      return "26.67 fb^{-1} (13.6 TeV)";
    else if (year == "2023")        return "17.7 fb^{-1} (13.6 TeV)";
    else if (year == "2023BPix")    return "9.5 fb^{-1} (13.6 TeV)";
    return "";
  }

  int NSignalPars(SignalModel model) { return (model == kDSCB) ? 7 : 5; }

  bool IsMonoExpBackground(BackgroundModel model) {
    return (model == kMonoExp1 || model == kMonoExp2 || model == kMonoExp3 || model == kMonoExp4);
  }

  bool IsExpBackground(BackgroundModel model) {
    return (model == kExp1 || model == kExp2 || model == kExp3 || model == kExp4 || IsMonoExpBackground(model));
  }

  bool IsMonoChebBackground(BackgroundModel model) {
    return (model == kMonoCheb1 || model == kMonoCheb2 || model == kMonoCheb3 || model == kMonoCheb4);
  }

  bool IsChebBackground(BackgroundModel model) {
    return (model == kCheb1 || model == kCheb2 || model == kCheb3 || model == kCheb4 || IsMonoChebBackground(model));
  }

  bool IsMonoBernBackground(BackgroundModel model) {
    return (model == kMonoBern1 || model == kMonoBern2 || model == kMonoBern3 || model == kMonoBern4 ||
            model == kMonoBern5 || model == kMonoBern6 || model == kMonoBern7 || model == kMonoBern8);
  }

  bool IsBernBackground(BackgroundModel model) {
    return (model == kBern1 || model == kBern2 || model == kBern3 || model == kBern4 ||
            model == kBern5 || model == kBern6 || model == kBern7 || model == kBern8 ||
            IsMonoBernBackground(model));
  }

  bool UsesLogNorm(BackgroundModel model) {
    return IsExpBackground(model) || IsBernBackground(model);
  }

  int ExpOrder(BackgroundModel model) {
    if(model == kExp4 || model == kMonoExp4) return 4;
    if(model == kExp3 || model == kMonoExp3) return 3;
    if(model == kExp2 || model == kMonoExp2) return 2;
    return 1;
  }

  int ChebOrder(BackgroundModel model) {
    if(model == kCheb4 || model == kMonoCheb4) return 4;
    if(model == kCheb3 || model == kMonoCheb3) return 3;
    if(model == kCheb2 || model == kMonoCheb2) return 2;
    return 1;
  }

  int BernOrder(BackgroundModel model) {
    if(model == kBern1 || model == kMonoBern1) return 1;
    if(model == kBern2 || model == kMonoBern2) return 2;
    if(model == kBern3 || model == kMonoBern3) return 3;
    if(model == kBern4 || model == kMonoBern4) return 4;
    if(model == kBern6 || model == kMonoBern6) return 6;
    if(model == kBern7 || model == kMonoBern7) return 7;
    if(model == kBern8 || model == kMonoBern8) return 8;
    return 5;
  }

  int NBkgPars(BackgroundModel model) {
    if(IsExpBackground(model))  return 1 + ExpOrder(model);
    if(IsChebBackground(model)) return 1 + ChebOrder(model);
    if(IsBernBackground(model)) return 1 + BernOrder(model);
    return 2;
  }

  SignalModel ParseSignalModel(TString s) {
    s.ToLower();
    if(s == "dscb" || s == "doublecb" || s == "doublecrystalball") return kDSCB;
    return kCB;
  }

  BackgroundModel ParseBackgroundModel(TString s) {
    s.ToLower();
    s.ReplaceAll("_", "");
    s.ReplaceAll("-", "");

    if(s == "exp1" || s == "exppol1" || s == "exppoly1") return kExp1;
    if(s == "exp2" || s == "exppol2" || s == "exppoly2") return kExp2;
    if(s == "exp3" || s == "exppol3" || s == "exppoly3") return kExp3;
    if(s == "exp4" || s == "exppol4" || s == "exppoly4") return kExp4;
    if(s == "monoexp1" || s == "monotonicexp1" || s == "monoexppol1") return kMonoExp1;
    if(s == "monoexp2" || s == "monotonicexp2" || s == "monoexppol2") return kMonoExp2;
    if(s == "monoexp3" || s == "monotonicexp3" || s == "monoexppol3") return kMonoExp3;
    if(s == "monoexp4" || s == "monotonicexp4" || s == "monoexppol4") return kMonoExp4;

    if(s == "cheb1" || s == "cheby1" || s == "chebychev1") return kCheb1;
    if(s == "cheb2" || s == "cheby2" || s == "chebychev2") return kCheb2;
    if(s == "cheb3" || s == "cheby3" || s == "chebychev3") return kCheb3;
    if(s == "cheb4" || s == "cheby4" || s == "chebychev4") return kCheb4;
    if(s == "monocheb1" || s == "monotoniccheb1" || s == "monochebychev1") return kMonoCheb1;
    if(s == "monocheb2" || s == "monotoniccheb2" || s == "monochebychev2") return kMonoCheb2;
    if(s == "monocheb3" || s == "monotoniccheb3" || s == "monochebychev3") return kMonoCheb3;
    if(s == "monocheb4" || s == "monotoniccheb4" || s == "monochebychev4") return kMonoCheb4;

    if(s == "bern1" || s == "bernstein1" || s == "positivebernstein1") return kBern1;
    if(s == "bern2" || s == "bernstein2" || s == "positivebernstein2") return kBern2;
    if(s == "bern3" || s == "bernstein3" || s == "positivebernstein3") return kBern3;
    if(s == "bern4" || s == "bernstein4" || s == "positivebernstein4") return kBern4;
    if(s == "bern5" || s == "bernstein5" || s == "positivebernstein5") return kBern5;
    if(s == "bern6" || s == "bernstein6" || s == "positivebernstein6") return kBern6;
    if(s == "bern7" || s == "bernstein7" || s == "positivebernstein7") return kBern7;
    if(s == "bern8" || s == "bernstein8" || s == "positivebernstein8") return kBern8;

    if(s == "monobern1" || s == "monotonicbernstein1" || s == "monobernstein1") return kMonoBern1;
    if(s == "monobern2" || s == "monotonicbernstein2" || s == "monobernstein2") return kMonoBern2;
    if(s == "monobern3" || s == "monotonicbernstein3" || s == "monobernstein3") return kMonoBern3;
    if(s == "monobern4" || s == "monotonicbernstein4" || s == "monobernstein4") return kMonoBern4;
    if(s == "monobern5" || s == "monotonicbernstein5" || s == "monobernstein5") return kMonoBern5;
    if(s == "monobern6" || s == "monotonicbernstein6" || s == "monobernstein6") return kMonoBern6;
    if(s == "monobern7" || s == "monotonicbernstein7" || s == "monobernstein7") return kMonoBern7;
    if(s == "monobern8" || s == "monotonicbernstein8" || s == "monobernstein8") return kMonoBern8;

    cout << "[WARNING] Unknown background model input '" << s << "'. Use Bern7 as default." << endl;
    return kBern7;
  }

  TString SignalModelName(SignalModel model) { return (model == kDSCB) ? "DSCB" : "CB"; }

  TString BackgroundModelName(BackgroundModel model) {
    if(model == kExp1)      return "exp(pol1)";
    if(model == kExp2)      return "exp(pol2)";
    if(model == kExp3)      return "exp(pol3)";
    if(model == kExp4)      return "exp(pol4)";
    if(model == kMonoExp1)  return "monotonic exp1";
    if(model == kMonoExp2)  return "monotonic exp2";
    if(model == kMonoExp3)  return "monotonic exp3";
    if(model == kMonoExp4)  return "monotonic exp4";
    if(model == kCheb1)     return "Chebychev1";
    if(model == kCheb2)     return "Chebychev2";
    if(model == kCheb3)     return "Chebychev3";
    if(model == kCheb4)     return "Chebychev4";
    if(model == kMonoCheb1) return "monotonic Chebychev1";
    if(model == kMonoCheb2) return "monotonic Chebychev2";
    if(model == kMonoCheb3) return "monotonic Chebychev3";
    if(model == kMonoCheb4) return "monotonic Chebychev4";
    if(model == kBern1)     return "Bernstein1";
    if(model == kBern2)     return "Bernstein2";
    if(model == kBern3)     return "Bernstein3";
    if(model == kBern4)     return "Bernstein4";
    if(model == kBern5)     return "Bernstein5";
    if(model == kBern6)     return "Bernstein6";
    if(model == kBern7)     return "Bernstein7";
    if(model == kBern8)     return "Bernstein8";
    if(model == kMonoBern1) return "monotonic Bernstein1";
    if(model == kMonoBern2) return "monotonic Bernstein2";
    if(model == kMonoBern3) return "monotonic Bernstein3";
    if(model == kMonoBern4) return "monotonic Bernstein4";
    if(model == kMonoBern5) return "monotonic Bernstein5";
    if(model == kMonoBern6) return "monotonic Bernstein6";
    if(model == kMonoBern7) return "monotonic Bernstein7";
    if(model == kMonoBern8) return "monotonic Bernstein8";
    return "unknown";
  }

  TString AvailableBackgroundModelsText() {
    return "Exp1, Exp2, Exp3, Exp4, MonoExp1, MonoExp2, MonoExp3, MonoExp4, "
           "Cheb1, Cheb2, Cheb3, Cheb4, MonoCheb1, MonoCheb2, MonoCheb3, MonoCheb4, "
           "Bern1, Bern2, Bern3, Bern4, Bern5, Bern6, Bern7, Bern8, "
           "MonoBern1, MonoBern2, MonoBern3, MonoBern4, MonoBern5, MonoBern6, MonoBern7, MonoBern8";
  }

  TString RegionName(const TString &baseRegion, const TString &binTag, const TString &status) {
    return baseRegion + "_" + binTag + "_" + status;
  }

  TString HistPath(const TString &baseRegion, const TString &binTag,
                   const TString &status, const TString &histName = "DileptonJPsi_Mass") {
    const TString region = RegionName(baseRegion, binTag, status);
    return region + "/" + histName + "___" + region;
  }

  TString ResolveFile(const TString &inputDir, const vector<TString> &candidates, const TString &label) {
    for(const auto &rel : candidates) {
      TString path = inputDir + rel;
      if(!gSystem->AccessPathName(path)) {
        cout << "[INFO] " << label << " file: " << path << endl;
        return path;
      }
    }
    cout << "[WARNING] No " << label << " file found. Tried:" << endl;
    for(const auto &rel : candidates) cout << "          " << inputDir + rel << endl;
    return inputDir + candidates.front();
  }


  TString NormaliseKey(TString s) {
    s.ToLower();
    s.ReplaceAll(" ", "");
    s.ReplaceAll("_", "");
    s.ReplaceAll("-", "");
    return s;
  }

  TString ResolveReferenceFile(const TString &inputDir, TString refInput) {
    TString key = NormaliseKey(refInput);

    if(refInput.EndsWith(".root") || refInput.Contains("/")) {
      TString path = refInput;
      if(!path.BeginsWith("/")) path = inputDir + path;
      if(!gSystem->AccessPathName(path)) {
        cout << "[INFO] Reference file: " << path << endl;
        return path;
      }
      cout << "[WARNING] Requested reference file does not exist: " << path << endl;
      return path;
    }

    if(key == "qcd" || key == "qcdmc") {
      return ResolveFile(inputDir, {"NIsoMuon_QCD_Inclusive.root"}, "QCD");
    }
    if(key == "top" || key == "tops" || key == "tttw") {
      return ResolveFile(inputDir, {"NIsoMuon_Top.root", "NIsoMuon_tt.root", "NIsoMuon_tW.root"}, "Top");
    }
    if(key == "allmc" || key == "mc") {
      return ResolveFile(inputDir, {"NIsoMuon_AllMC.root"}, "AllMC");
    }
    if(key == "qcdtop" || key == "qcdtops" || key == "qcdtttw") {
      return ResolveFile(inputDir, {"NIsoMuon_QCDTop.root"}, "QCDTop");
    }

    cout << "[WARNING] Unknown reference sample '" << refInput << "'. Use QCD by default." << endl;
    return ResolveFile(inputDir, {"NIsoMuon_QCD_Inclusive.root"}, "QCD");
  }

  TString ReferenceLabel(TString refInput) {
    TString key = NormaliseKey(refInput);
    if(key == "qcd" || key == "qcdmc") return "QCD";
    if(key == "top" || key == "tops" || key == "tttw") return "Top";
    if(key == "allmc" || key == "mc") return "AllMC";
    if(key == "qcdtop" || key == "qcdtops" || key == "qcdtttw") return "QCDTop";

    if(refInput.EndsWith(".root") || refInput.Contains("/")) {
      TString base = gSystem->BaseName(refInput);
      base.ReplaceAll(".root", "");
      base.ReplaceAll("NIsoMuon_", "");
      return base;
    }
    return "QCD";
  }

  bool MatchBinFilter(const BinDef &bin, TString filter) {
    TString key = NormaliseKey(filter);
    if(key == "" || key == "all" || key == "*") return true;
    if(key == "inclusive") return bin.inclusive;
    if(key == "noinclusive" || key == "exclusive") return !bin.inclusive;

    TString tag = NormaliseKey(bin.tag);
    if(tag.Contains(key)) return true;

    TString compact = tag;
    compact.ReplaceAll("pt", "");
    compact.ReplaceAll("abseta", "eta");
    if(compact.Contains(key)) return true;
    return false;
  }

  int CountSelectedBins(const vector<BinDef> &bins, TString filter, int maxBins) {
    int n = 0;
    for(const auto &bin : bins) {
      if(!MatchBinFilter(bin, filter)) continue;
      if(maxBins >= 0 && n >= maxBins) break;
      ++n;
    }
    return n;
  }

  TH1D* CloneAsTH1D(TH1 *src, const TString &cloneName) {
    if(!src) return nullptr;
    const int nBins = src->GetNbinsX();
    const TArrayD *xbins = src->GetXaxis()->GetXbins();
    TH1D *out = nullptr;
    if(xbins && xbins->GetSize() > 0) {
      out = new TH1D(cloneName.Data(), src->GetTitle(), nBins, xbins->GetArray());
    }
    else {
      out = new TH1D(cloneName.Data(), src->GetTitle(), nBins,
                     src->GetXaxis()->GetXmin(), src->GetXaxis()->GetXmax());
    }
    out->SetDirectory(0);
    out->Sumw2();
    for(int i = 0; i <= nBins + 1; ++i) {
      out->SetBinContent(i, src->GetBinContent(i));
      out->SetBinError(i, src->GetBinError(i));
    }
    return out;
  }

  TH1D* LoadHistSingle(const TString &filePath, const TString &histPath,
                       const TString &cloneName, bool verboseMissing = true) {
    if(gSystem->AccessPathName(filePath)) {
      if(verboseMissing) cout << "[WARNING] Missing file: " << filePath << endl;
      return nullptr;
    }
    TFile *f = TFile::Open(filePath);
    if(!f || f->IsZombie()) {
      if(verboseMissing) cout << "[WARNING] Cannot open file: " << filePath << endl;
      return nullptr;
    }
    TH1 *hIn = dynamic_cast<TH1*>(f->Get(histPath));
    if(!hIn) {
      if(verboseMissing) cout << "[WARNING] Missing histogram: " << filePath << ":" << histPath << endl;
      f->Close();
      delete f;
      return nullptr;
    }
    TH1D *h = CloneAsTH1D(hIn, cloneName);
    f->Close();
    delete f;
    return h;
  }

  int PrintKeysMatching(TDirectory *dir, const TString &prefix,
                        const TString &pattern, int &nPrinted, const int maxPrint) {
    if(!dir) return 0;
    int nMatch = 0;
    TIter next(dir->GetListOfKeys());
    TKey *key = nullptr;
    while((key = dynamic_cast<TKey*>(next()))) {
      const TString name = key->GetName();
      const TString path = (prefix == "") ? name : prefix + "/" + name;
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if(cl && cl->InheritsFrom(TDirectory::Class())) {
        TDirectory *subdir = dynamic_cast<TDirectory*>(dir->Get(name));
        nMatch += PrintKeysMatching(subdir, path, pattern, nPrinted, maxPrint);
        continue;
      }
      if(pattern == "" || path.Contains(pattern)) {
        ++nMatch;
        if(nPrinted < maxPrint) {
          cout << "    " << path << endl;
          ++nPrinted;
        }
      }
    }
    return nMatch;
  }

  void InspectInputHistograms(const TString &filePath, const TString &label,
                              const TString &baseRegion, const TString &histName,
                              int maxPrint = 120) {
    cout << "[INSPECT] " << label << " file: " << filePath << endl;
    TFile *f = TFile::Open(filePath);
    if(!f || f->IsZombie()) {
      cout << "[INSPECT]   cannot open file." << endl;
      return;
    }

    vector<TString> patterns = {baseRegion, "MuonIDEfficiency", histName, "Dilepton"};
    for(const auto &pat : patterns) {
      int nPrinted = 0;
      cout << "[INSPECT]   keys containing '" << pat << "':" << endl;
      const int nMatch = PrintKeysMatching(f, "", pat, nPrinted, maxPrint);
      if(nMatch == 0) cout << "    none" << endl;
      else if(nMatch > maxPrint) cout << "    ... (" << (nMatch - maxPrint) << " more not printed)" << endl;
      cout << "[INSPECT]   total matches: " << nMatch << endl;
    }
    f->Close();
    delete f;
  }

  TH1D* LoadHistFromOpenFile(TFile *f, const TString &filePath, const TString &histPath,
                             const TString &cloneName, bool verboseMissing = true) {
    if(!f || f->IsZombie()) return nullptr;
    TH1 *hIn = dynamic_cast<TH1*>(f->Get(histPath));
    if(!hIn) {
      if(verboseMissing) cout << "[WARNING] Missing histogram: " << filePath << ":" << histPath << endl;
      return nullptr;
    }
    return CloneAsTH1D(hIn, cloneName);
  }

  double UniformOverlapFraction(double fineLow, double fineHigh, double targetLow, double targetHigh) {
    const double overlapLow = std::max(fineLow, targetLow);
    const double overlapHigh = std::min(fineHigh, targetHigh);
    if(fineHigh <= fineLow || overlapHigh <= overlapLow) return 0.;
    return (overlapHigh - overlapLow) / (fineHigh - fineLow);
  }

  bool IsFullyInside(double fineLow, double fineHigh, double targetLow, double targetHigh) {
    return (targetLow <= fineLow + 1e-9 && fineHigh <= targetHigh + 1e-9);
  }

  double HistIntegralInRangeByBinOverlap(TH1 *h, double lo, double hi) {
    if(!h || hi <= lo) return 0.;
    double sum = 0.;
    TAxis *ax = h->GetXaxis();
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double binLo = ax->GetBinLowEdge(ibin);
      const double binHi = ax->GetBinUpEdge(ibin);
      const double ovLo = std::max(lo, binLo);
      const double ovHi = std::min(hi, binHi);
      if(ovHi <= ovLo) continue;
      const double bw = std::max(1e-12, binHi - binLo);
      sum += h->GetBinContent(ibin) * (ovHi - ovLo) / bw;
    }
    return sum;
  }

  double FractionFromProbePt(TFile *f, const TString &baseRegion, const TString &subTag,
                             const TString &status, double fineLow, double fineHigh,
                             double targetLow, double targetHigh) {
    if(!f || f->IsZombie()) return -1.;
    const TString dir = baseRegion + "_" + subTag + "_" + status;
    TH1 *hPt = dynamic_cast<TH1*>(f->Get(dir + "/Probe_Pt___" + dir));
    if(!hPt) return -1.;

    // The analyzer currently fills Probe_Pt in [0,200] GeV.  If a source bin extends
    // beyond that range, the visible Probe_Pt histogram cannot split the overflow part.
    // In that case the caller falls back to the geometrical overlap fraction.
    if(fineHigh > hPt->GetXaxis()->GetXmax() + 1e-9) return -1.;

    const double denom = HistIntegralInRangeByBinOverlap(hPt, fineLow, fineHigh);
    if(denom <= 0.) return -1.;
    const double num = HistIntegralInRangeByBinOverlap(hPt,
                                                       std::max(fineLow, targetLow),
                                                       std::min(fineHigh, targetHigh));
    return std::max(0., std::min(1., num / denom));
  }

  TH1D* BuildFromAnalyzerInputBins(const TString &filePath, const TString &baseRegion,
                                   const BinDef &targetBin, const TString &status,
                                   const TString &cloneName, const TString &histName) {
    TFile *f = TFile::Open(filePath);
    if(!f || f->IsZombie()) {
      cout << "[WARNING] Cannot open file for subbin merge: " << filePath << endl;
      return nullptr;
    }

    TH1D *out = nullptr;
    int nUsed = 0;
    int nMissing = 0;
    int nPartial = 0;

    const vector<double> etaEdges = DefaultEtaEdges();
    const vector<double> ptEdges = AnalyzerInputPtEdges();

    for(unsigned int ieta = 0; ieta + 1 < etaEdges.size(); ++ieta) {
      const double etaLow = etaEdges[ieta];
      const double etaHigh = etaEdges[ieta+1];
      const bool integrateAllEta = targetBin.inclusive || IsFullEtaRange(targetBin.etaLow, targetBin.etaHigh);
      if(!integrateAllEta) {
        if(!NearlyEqual(etaLow, targetBin.etaLow) || !NearlyEqual(etaHigh, targetBin.etaHigh)) continue;
      }

      for(unsigned int ipt = 0; ipt + 1 < ptEdges.size(); ++ipt) {
        const double ptLow = ptEdges[ipt];
        const double ptHigh = ptEdges[ipt+1];
        if(!targetBin.inclusive) {
          if(std::min(ptHigh, targetBin.ptHigh) <= std::max(ptLow, targetBin.ptLow)) continue;
        }

        const TString subTag = BinTag(ptLow, ptHigh, etaLow, etaHigh);
        const TString subPath = HistPath(baseRegion, subTag, status, histName);
        TH1D *hSub = LoadHistFromOpenFile(f, filePath, subPath,
                                          cloneName + "_sub_" + subTag, false);
        if(!hSub) {
          ++nMissing;
          continue;
        }

        double frac = 1.;
        // With the standard configured output binning this branch
        // should never be entered.  Keep it only for explicitly non-standard
        // future bin definitions.
        if(!targetBin.inclusive && !IsFullyInside(ptLow, ptHigh, targetBin.ptLow, targetBin.ptHigh)) {
          ++nPartial;
          frac = FractionFromProbePt(f, baseRegion, subTag, status,
                                     ptLow, ptHigh, targetBin.ptLow, targetBin.ptHigh);
          if(frac < 0.) {
            frac = UniformOverlapFraction(ptLow, ptHigh, targetBin.ptLow, targetBin.ptHigh);
            cout << "[WARNING] Partial pT source bin " << subTag << " -> " << targetBin.tag
                 << " in " << status << ": Probe_Pt cannot determine the split; use geometric fraction "
                 << frac << endl;
          }
          else {
            cout << "[INFO] Partial pT source bin " << subTag << " -> " << targetBin.tag
                 << " in " << status << ": use Probe_Pt fraction " << frac << endl;
          }
        }

        if(frac <= 0.) {
          delete hSub;
          continue;
        }
        if(std::fabs(frac - 1.) > 1e-9) hSub->Scale(frac);

        if(!out) {
          out = CloneAsTH1D(hSub, cloneName);
        }
        else {
          out->Add(hSub);
        }
        ++nUsed;
        delete hSub;
      }
    }

    if(out) {
      cout << "[MERGE] " << filePath << ":" << targetBin.tag << "_" << status
           << " built from " << nUsed << " analyzer pT-bin histogram(s)";
      if(nPartial > 0) cout << ", partial bins=" << nPartial;
      if(nMissing > 0) {
        cout << ", missing inputs=" << nMissing
             << " (treated as zero contribution to the merged bin)";
      }
      cout << endl;
    }
    else {
      cout << "[WARNING] Could not build " << targetBin.tag << "_" << status
           << " from analyzer pT-bin histograms in " << filePath << endl;
    }

    f->Close();
    delete f;
    return out;
  }

  TH1D* LoadForBin(const TString &filePath, const TString &baseRegion,
                   const BinDef &bin, const TString &status,
                   const vector<BinDef> &/*bins*/, const TString &cloneName,
                   const TString &histName) {
    // First use an exact output-bin histogram if it exists.  If not, build the
    // requested eta-dependent output bin from the analyzer pT-binned histograms.
    // No Dilepton_Mass fallback is used; histName must match the requested mass histogram.
    TH1D *direct = LoadHistSingle(filePath, HistPath(baseRegion, bin.tag, status, histName),
                                  cloneName, false);
    if(direct) return direct;

    cout << "[INFO] Direct output-bin histogram not found; build from analyzer input bins: "
         << bin.tag << " " << status << endl;
    return BuildFromAnalyzerInputBins(filePath, baseRegion, bin, status, cloneName, histName);
  }

  TH1D* AddHists(TH1D *h1, TH1D *h2, const TString &name) {
    if(!h1 && !h2) return nullptr;
    TH1D *out = h1 ? CloneAsTH1D(h1, name) : CloneAsTH1D(h2, name);
    if(h1 && h2) out->Add(h2);
    return out;
  }

  int ValidRebinFactor(TH1D *h, int requested) {
    if(!h || requested <= 1) return 1;
    int factor = requested;
    const int nBins = h->GetNbinsX();
    while(factor > 1 && (nBins % factor) != 0) --factor;
    if(factor != requested) {
      cout << "[WARNING] Requested rebin factor " << requested
           << " does not divide " << nBins << " bins; use " << factor << " instead." << endl;
    }
    return std::max(1, factor);
  }

  TH1D* RebinAndMakeDensity(TH1D *hIn, int rebinFactor, const TString &name,
                            bool verbose = true) {
    if(!hIn) return nullptr;
    TH1D *h = CloneAsTH1D(hIn, name + "_rawClone");

    const int nBinsBefore = h->GetNbinsX();
    const double widthBefore = h->GetXaxis()->GetBinWidth(1);
    const int factor = ValidRebinFactor(h, rebinFactor);

    if(factor > 1) {
      TH1D *hRebinned = dynamic_cast<TH1D*>(h->Rebin(factor, (name + "_rebinned").Data()));
      if(hRebinned) {
        hRebinned->SetDirectory(0);
        hRebinned->Sumw2();
        h = hRebinned;
      }
    }

    const int nBinsAfter = h->GetNbinsX();
    const double widthAfter = h->GetXaxis()->GetBinWidth(1);
    if(verbose) {
      cout << "[REBIN] " << name
           << ": requested=" << rebinFactor
           << ", applied=" << factor
           << ", nbins " << nBinsBefore << " -> " << nBinsAfter
           << ", first-bin width " << widthBefore << " -> " << widthAfter
           << " GeV" << endl;
    }

    TH1D *density = CloneAsTH1D(h, name + "_density");
    for(int ibin = 1; ibin <= density->GetNbinsX(); ++ibin) {
      const double width = density->GetXaxis()->GetBinWidth(ibin);
      if(width <= 0.) continue;
      density->SetBinContent(ibin, density->GetBinContent(ibin) / width);
      density->SetBinError(ibin, density->GetBinError(ibin) / width);
    }
    density->GetYaxis()->SetTitle("Events / GeV");
    return density;
  }

  bool IsInFinalFitVeto(const double x);

  double HistIntegralDensity(TH1D *h, double xmin, double xmax, bool excludePsiP = false) {
    if(!h) return 0.;
    double sum = 0.;
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(x < xmin || x >= xmax) continue;
      if(excludePsiP && IsInFinalFitVeto(x)) continue;
      sum += h->GetBinContent(ibin) * h->GetXaxis()->GetBinWidth(ibin);
    }
    return sum;
  }

  double HistMaxInRange(TH1D *h, double xmin, double xmax) {
    if(!h) return 0.;
    double out = 0.;
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(x < xmin || x >= xmax) continue;
      out = std::max(out, h->GetBinContent(ibin) + h->GetBinError(ibin));
    }
    return out;
  }

  double Median(vector<double> vals, double fallback) {
    vals.erase(std::remove_if(vals.begin(), vals.end(), [](double x){ return !std::isfinite(x) || x <= 0.; }), vals.end());
    if(vals.empty()) return fallback;
    std::sort(vals.begin(), vals.end());
    const unsigned int n = vals.size();
    if(n % 2) return vals[n/2];
    return 0.5 * (vals[n/2 - 1] + vals[n/2]);
  }

  bool IsInBkgSideband(const double x) {
    for(const auto &r : gBkgSidebands) {
      if(r.first <= x && x < r.second) return true;
    }
    return false;
  }

  bool IsInFinalFitVeto(const double x) {
    return gUseFinalVeto && (gFinalVetoLow <= x && x < gFinalVetoHigh);
  }

  double BkgSidebandWeight(const double x) {
    // Upweight the sideband nearest to the signal peak.  For J/psi this is [3.3,3.5];
    // for Z this is the high-mass sideband if present.
    if(gBkgSidebands.size() >= 2 && gBkgSidebands[1].first <= x && x < gBkgSidebands[1].second) {
      return std::max(1.0, gSide2FitWeight);
    }
    return 1.0;
  }

  double SidebandMedianDensity(TH1D *h, double /*fitMin*/, double /*fitMax*/) {
    vector<double> vals;
    if(!h) return 1e-6;
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(!IsInBkgSideband(x)) continue;
      vals.push_back(h->GetBinContent(ibin));
    }
    return Median(vals, std::max(1e-6, 0.02 * HistMaxInRange(h, gBkgFitMin, gBkgFitMax)));
  }

  double LocalMedianDensity(TH1D *h, double xmin, double xmax, double fallback) {
    vector<double> vals;
    if(!h) return fallback;
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(x < xmin || x >= xmax) continue;
      const double y = h->GetBinContent(ibin);
      if(std::isfinite(y) && y > 0.) vals.push_back(y);
    }
    return Median(vals, fallback);
  }

  double EstimateSidebandDensity(TH1D *h, double x, double fallback) {
    const double y1 = LocalMedianDensity(h, gBkgSidebands.size() > 0 ? gBkgSidebands[0].first : gBkgFitMin, gBkgSidebands.size() > 0 ? gBkgSidebands[0].second : gBkgFitMin, fallback);
    const double y2 = LocalMedianDensity(h, gBkgSidebands.size() > 1 ? gBkgSidebands[1].first : gBkgFitMin, gBkgSidebands.size() > 1 ? gBkgSidebands[1].second : gBkgFitMax, fallback);
    const double y3 = LocalMedianDensity(h, gBkgSidebands.size() > 2 ? gBkgSidebands[2].first : gBkgFitMax, gBkgSidebands.size() > 2 ? gBkgSidebands[2].second : gBkgFitMax, fallback);
    const double x1 = gBkgSidebands.size() > 0 ? 0.5 * (gBkgSidebands[0].first + gBkgSidebands[0].second) : gBkgFitMin;
    const double x2 = gBkgSidebands.size() > 1 ? 0.5 * (gBkgSidebands[1].first + gBkgSidebands[1].second) : gPeakMass;
    const double x3 = gBkgSidebands.size() > 2 ? 0.5 * (gBkgSidebands[2].first + gBkgSidebands[2].second) : gBkgFitMax;
    auto logInterp = [](double xa, double ya, double xb, double yb, double xx, double fb) {
      if(!(ya > 0.) || !(yb > 0.) || std::fabs(xb - xa) < 1e-9) return fb;
      const double w = (xx - xa) / (xb - xa);
      double ly = std::log(ya) + w * (std::log(yb) - std::log(ya));
      if(ly > 700.) ly = 700.;
      if(ly < -700.) ly = -700.;
      return std::exp(ly);
    };
    double out = fallback;
    if(x <= x2) out = logInterp(x1, y1, x2, y2, x, fallback);
    else        out = logInterp(x2, y2, x3, y3, x, fallback);
    return (std::isfinite(out) && out > 0.) ? out : fallback;
  }

  double CBShapeRaw(double t, double alpha, double n) {
    const double a = std::max(std::fabs(alpha), 1e-6);
    const double nn = std::max(n, 1.0001);
    if(t > -a) return std::exp(-0.5 * t * t);
    const double A = std::pow(nn / a, nn) * std::exp(-0.5 * a * a);
    const double B = nn / a - a;
    return A * std::pow(B - t, -nn);
  }

  double CBNorm(double sigma, double alpha, double n) {
    const double sig = std::max(std::fabs(sigma), 1e-9);
    const double a = std::max(std::fabs(alpha), 1e-6);
    const double nn = std::max(n, 1.0001);
    const double tail = (nn / a) * std::exp(-0.5 * a * a) / (nn - 1.0);
    const double core = std::sqrt(TMath::Pi() / 2.0) * (1.0 + TMath::Erf(a / std::sqrt(2.0)));
    return sig * (tail + core);
  }

  double DSCBShapeRaw(double t, double alphaL, double nL, double alphaR, double nR) {
    const double aL = std::max(std::fabs(alphaL), 1e-6);
    const double aR = std::max(std::fabs(alphaR), 1e-6);
    const double nnL = std::max(nL, 1.0001);
    const double nnR = std::max(nR, 1.0001);
    if(t < -aL) {
      const double A = std::pow(nnL / aL, nnL) * std::exp(-0.5 * aL * aL);
      const double B = nnL / aL - aL;
      return A * std::pow(B - t, -nnL);
    }
    if(t > aR) {
      const double A = std::pow(nnR / aR, nnR) * std::exp(-0.5 * aR * aR);
      const double B = nnR / aR - aR;
      return A * std::pow(B + t, -nnR);
    }
    return std::exp(-0.5 * t * t);
  }

  double DSCBNorm(double sigma, double alphaL, double nL, double alphaR, double nR) {
    const double sig = std::max(std::fabs(sigma), 1e-9);
    const double aL = std::max(std::fabs(alphaL), 1e-6);
    const double aR = std::max(std::fabs(alphaR), 1e-6);
    const double nnL = std::max(nL, 1.0001);
    const double nnR = std::max(nR, 1.0001);
    const double left  = (nnL / aL) * std::exp(-0.5 * aL * aL) / (nnL - 1.0);
    const double right = (nnR / aR) * std::exp(-0.5 * aR * aR) / (nnR - 1.0);
    const double core  = std::sqrt(TMath::Pi() / 2.0) *
                         (TMath::Erf(aR / std::sqrt(2.0)) + TMath::Erf(aL / std::sqrt(2.0)));
    return sig * (left + core + right);
  }

  Double_t GenericSignal(Double_t *x, Double_t *p) {
    const double N = std::max(p[0], 0.0);
    const double mean = p[1];
    const double sigma = std::max(std::fabs(p[2]), 1e-9);
    const double t = (x[0] - mean) / sigma;

    if(gSignalModel == kDSCB) {
      const double raw = DSCBShapeRaw(t, p[3], p[4], p[5], p[6]);
      const double norm = DSCBNorm(sigma, p[3], p[4], p[5], p[6]);
      return (norm > 0.) ? N * raw / norm : 0.;
    }

    const double raw = CBShapeRaw(t, p[3], p[4]);
    const double norm = CBNorm(sigma, p[3], p[4]);
    return (norm > 0.) ? N * raw / norm : 0.;
  }

  double SignalIntegralFromPars(const vector<double> &pars, SignalModel sig,
                                double xmin = kYieldIntLow, double xmax = kYieldIntHigh) {
    if(xmax <= xmin) return 0.;
    const int nSig = NSignalPars(sig);
    if((int)pars.size() < nSig) return 0.;

    SignalModel oldSig = gSignalModel;
    gSignalModel = sig;
    static int sigIntCounter = 0;
    TF1 sigFn((TString("sigInt_") + TString::Itoa(sigIntCounter++, 10)).Data(),
              GenericSignal, xmin, xmax, nSig);
    sigFn.SetNpx(1000);
    for(int i = 0; i < nSig; ++i) sigFn.SetParameter(i, pars[i]);
    const double val = sigFn.Integral(xmin, xmax);
    gSignalModel = oldSig;
    return (std::isfinite(val) && val > 0.) ? val : 0.;
  }

  double SignalIntegralFromModel(TF1 *model, SignalModel sig,
                                 double xmin = kYieldIntLow, double xmax = kYieldIntHigh) {
    if(!model) return 0.;
    const int nSig = NSignalPars(sig);
    vector<double> pars(nSig, 0.);
    for(int i = 0; i < nSig; ++i) pars[i] = model->GetParameter(i);
    return SignalIntegralFromPars(pars, sig, xmin, xmax);
  }

  double SignalIntegralErrorFromFit(TF1 *model, const TFitResultPtr &fitRes, SignalModel sig,
                                    double xmin = kYieldIntLow, double xmax = kYieldIntHigh) {
    if(!model) return 0.;
    const int nSig = NSignalPars(sig);

    vector<double> pars(nSig, 0.);
    vector<double> grad(nSig, 0.);
    for(int i = 0; i < nSig; ++i) pars[i] = model->GetParameter(i);

    const double central = SignalIntegralFromPars(pars, sig, xmin, xmax);
    if(central <= 0.) return 0.;

    for(int i = 0; i < nSig; ++i) {
      double step = 0.;
      if(fitRes.Get()) {
        const double err = model->GetParError(i);
        if(std::isfinite(err) && err > 0.) step = 0.25 * err;
      }
      if(!(step > 0.) || !std::isfinite(step)) {
        step = 1e-4 * std::max(1.0, std::fabs(pars[i]));
      }
      if(i == 1) step = std::min(step, 1e-3);  // mean, GeV
      if(i == 2) step = std::min(step, 1e-4);  // sigma, GeV
      if(step <= 0.) continue;

      vector<double> plus = pars;
      vector<double> minus = pars;
      plus[i] += step;
      minus[i] -= step;
      const double ip = SignalIntegralFromPars(plus, sig, xmin, xmax);
      const double im = SignalIntegralFromPars(minus, sig, xmin, xmax);
      if(std::isfinite(ip) && std::isfinite(im)) grad[i] = (ip - im) / (2.0 * step);
    }

    double var = 0.;
    if(fitRes.Get()) {
      for(int i = 0; i < nSig; ++i) {
        for(int j = 0; j < nSig; ++j) {
          var += grad[i] * fitRes->CovMatrix(i, j) * grad[j];
        }
      }
    }

    if(!(std::isfinite(var) && var > 0.)) {
      const double rawNorm = model->GetParameter(0);
      const double rawNormErr = model->GetParError(0);
      if(rawNorm > 0. && std::isfinite(rawNormErr) && rawNormErr > 0.) {
        const double frac = central / rawNorm;
        var = frac * frac * rawNormErr * rawNormErr;
      }
    }

    return (std::isfinite(var) && var > 0.) ? std::sqrt(var) : std::sqrt(std::max(0.0, central));
  }


  double SafeExp(double x) {
    if(x > 700.) x = 700.;
    if(x < -700.) x = -700.;
    return std::exp(x);
  }

  double ScaledMass(const double x) {
    const double mid = 0.5 * (gBkgShapeMin + gBkgShapeMax);
    const double half = std::max(0.5 * (gBkgShapeMax - gBkgShapeMin), 1e-6);
    return (x - mid) / half;
  }

  double UnitMass(const double x) {
    const double den = std::max(gBkgShapeMax - gBkgShapeMin, 1e-6);
    double t = (x - gBkgShapeMin) / den;
    if(t < 0.) t = 0.;
    if(t > 1.) t = 1.;
    return t;
  }

  double BinomialCoeff(int n, int k) {
    if(k < 0 || k > n) return 0.;
    if(k == 0 || k == n) return 1.;
    double out = 1.;
    for(int i = 1; i <= k; ++i) out *= double(n - k + i) / double(i);
    return out;
  }

  double BernsteinBasis(int n, int i, double t) {
    if(t < 0.) t = 0.;
    if(t > 1.) t = 1.;
    return BinomialCoeff(n, i) * std::pow(t, i) * std::pow(1.0 - t, n - i);
  }

  double PositiveBernsteinRaw(int order, const double *shapePars, double t) {
    // shapePars has order entries.  The first coefficient is fixed to one to remove
    // the degeneracy with the overall normalisation p0.
    double raw = BernsteinBasis(order, 0, t);
    for(int i = 1; i <= order; ++i) {
      const double coeff = SafeExp(shapePars[i - 1]);
      raw += coeff * BernsteinBasis(order, i, t);
    }
    return std::max(raw, 1e-300);
  }

  double MonotonicBernsteinRaw(int order, const double *shapePars, double t) {
    // Positive and monotonically decreasing Bernstein coefficients.
    // coeff_0 = 1, coeff_i = coeff_{i-1} * exp(-d_i^2).
    double coeff = 1.0;
    double raw = coeff * BernsteinBasis(order, 0, t);
    for(int i = 1; i <= order; ++i) {
      const double d = shapePars[i - 1];
      coeff *= SafeExp(-d * d);
      raw += coeff * BernsteinBasis(order, i, t);
    }
    return std::max(raw, 1e-300);
  }

  double PositiveBernsteinBackground(const double x, const double *p, int order, bool monotonic) {
    const double norm = SafeExp(p[0]);
    const double t = UnitMass(x);
    const double tref = UnitMass(gPeakMass);
    const double raw = monotonic ? MonotonicBernsteinRaw(order, &p[1], t)
                                 : PositiveBernsteinRaw(order, &p[1], t);
    const double ref = monotonic ? MonotonicBernsteinRaw(order, &p[1], tref)
                                 : PositiveBernsteinRaw(order, &p[1], tref);
    return norm * raw / std::max(ref, 1e-300);
  }

  double PositiveRateBernsteinRaw(int order, const double *shapePars, double t) {
    // Positive derivative basis used by MonoExpN.  For MonoExpN, d(log f)/dt
    // is minus this positive rate, so the function is exactly non-increasing.
    if(order <= 0) return 1.0;
    const int degree = order - 1;
    double rate = 0.;
    for(int i = 0; i <= degree; ++i) {
      rate += SafeExp(shapePars[i]) * BernsteinBasis(degree, i, t);
    }
    return std::max(rate, 1e-300);
  }

  double IntegratePositiveRateBernstein(int order, const double *shapePars, double a, double b) {
    if(a == b) return 0.;
    const double sign = (b > a) ? 1.0 : -1.0;
    const double lo = std::min(a, b);
    const double hi = std::max(a, b);
    const int nStep = 48;
    const double step = (hi - lo) / double(nStep);
    double sum = 0.;
    for(int i = 0; i < nStep; ++i) {
      const double u = lo + (i + 0.5) * step;
      sum += PositiveRateBernsteinRaw(order, shapePars, u);
    }
    return sign * sum * step;
  }

  double MonotonicExpBackground(const double x, const double *p, int order) {
    const double t = UnitMass(x);
    const double tref = UnitMass(gPeakMass);
    const double integral = IntegratePositiveRateBernstein(order, &p[1], tref, t);
    return SafeExp(p[0] - integral);
  }

  double ChebychevRaw(const double x, const double *p, int order) {
    const double t = ScaledMass(x);
    const double T0 = 1.0;
    const double T1 = t;
    const double T2 = 2.0 * t * t - 1.0;
    const double T3 = 4.0 * t * t * t - 3.0 * t;
    const double T4 = 8.0 * t * t * t * t - 8.0 * t * t + 1.0;

    double sum = T0;
    if(order >= 1) sum += p[1] * T1;
    if(order >= 2) sum += p[2] * T2;
    if(order >= 3) sum += p[3] * T3;
    if(order >= 4) sum += p[4] * T4;
    return p[0] * sum;
  }

  double ChebychevBackground(const double x, const double *p, int order) {
    return std::max(ChebychevRaw(x, p, order), 1e-9);
  }

  bool ChebychevPositiveAndDecreasing(const double *p, int order) {
    const int nScan = 80;
    double prev = ChebychevRaw(gBkgShapeMin, p, order);
    if(!std::isfinite(prev) || prev <= 0.) return false;
    for(int i = 1; i <= nScan; ++i) {
      const double x = gBkgShapeMin + (gBkgShapeMax - gBkgShapeMin) * double(i) / double(nScan);
      const double y = ChebychevRaw(x, p, order);
      if(!std::isfinite(y) || y <= 0.) return false;
      if(y > prev * (1.0 + 1e-6) + 1e-9) return false;
      prev = y;
    }
    return true;
  }

  Double_t GenericBackground(Double_t *x, Double_t *p) {
    const double xx = x[0];
    const double x0 = gPeakMass;
    const double dx = xx - x0;

    if(IsExpBackground(gBackgroundModel)) {
      const int order = ExpOrder(gBackgroundModel);
      if(IsMonoExpBackground(gBackgroundModel)) return MonotonicExpBackground(xx, p, order);
      double exponent = p[0];
      if(order >= 1) exponent += p[1] * dx;
      if(order >= 2) exponent += p[2] * dx * dx;
      if(order >= 3) exponent += p[3] * dx * dx * dx;
      if(order >= 4) exponent += p[4] * dx * dx * dx * dx;
      return SafeExp(exponent);
    }

    if(IsChebBackground(gBackgroundModel)) {
      const int order = ChebOrder(gBackgroundModel);
      if(IsMonoChebBackground(gBackgroundModel) && !ChebychevPositiveAndDecreasing(p, order)) return 1e300;
      return ChebychevBackground(xx, p, order);
    }

    if(IsBernBackground(gBackgroundModel)) {
      return PositiveBernsteinBackground(xx, p, BernOrder(gBackgroundModel), IsMonoBernBackground(gBackgroundModel));
    }

    return SafeExp(p[0] + p[1] * dx);
  }

  Double_t GenericBackgroundLog(Double_t *x, Double_t *p) {
    const double y = GenericBackground(x, p);
    return std::log(std::max(y, 1e-300));
  }

  Double_t GenericModel(Double_t *x, Double_t *p) {
    if(gRejectPsiPInFinalFit && IsInFinalFitVeto(x[0])) {
      TF1::RejectPoint();
      return 0.;
    }
    const int nSig = NSignalPars(gSignalModel);
    return GenericSignal(x, p) + GenericBackground(x, &p[nSig]);
  }

  Double_t GenericSimultaneousModel(Double_t *x, Double_t *p) {
    const int nSig = NSignalPars(gSignalModel);
    const int nBkg = NBkgPars(gBackgroundModel);
    const bool isFail = (x[0] >= gSimultaneousSplit);
    Double_t mass[1] = { isFail ? x[0] - gSimultaneousOffset : x[0] };

    const double eff = std::max(0.0, std::min(1.0, p[0]));
    const double nTotal = std::max(0.0, p[1]);
    double sigPars[7] = {0., 0., 0., 0., 0., 0., 0.};
    sigPars[0] = nTotal * (isFail ? (1.0 - eff) : eff);
    for(int i = 1; i < nSig; ++i) sigPars[i] = p[i + 1];

    const int passBkgStart = nSig + 1;
    const int failBkgStart = passBkgStart + nBkg;
    const int bkgStart = isFail ? failBkgStart : passBkgStart;

    return GenericSignal(mass, sigPars) + GenericBackground(mass, &p[bkgStart]);
  }

  Double_t GenericSimultaneousCountModel(Double_t *x, Double_t *p) {
    // The likelihood histogram stores counts (or sum of weights), while the
    // physics model is a density in Events/GeV.  TH1::Fit option "I" replaces
    // the bin-centre value by the bin average of this function.  Multiplying
    // by the common mass-bin width therefore makes that average equal to the
    // expected integral of the density in the bin.
    return gSimultaneousBinWidth * GenericSimultaneousModel(x, p);
  }

  void SetSignalParameterNames(TF1 *f, SignalModel sig) {
    const TString ampName = TString("A_{") + gResonanceLabel + "}";
    const TString massName = TString("m_{") + gResonanceLabel + "}";
    if(sig == kDSCB) {
      f->SetParName(0, ampName.Data());
      f->SetParName(1, massName.Data());
      f->SetParName(2, "#sigma");
      f->SetParName(3, "#alpha_{L}");
      f->SetParName(4, "n_{L}");
      f->SetParName(5, "#alpha_{R}");
      f->SetParName(6, "n_{R}");
    }
    else {
      f->SetParName(0, ampName.Data());
      f->SetParName(1, massName.Data());
      f->SetParName(2, "#sigma");
      f->SetParName(3, "#alpha");
      f->SetParName(4, "n");
    }
  }

  void ConfigureModel(TF1 *model, TH1D *h, const FitOutput *commonShape,
                      bool fixCommonShape, double fitMin, double fitMax,
                      SignalModel sig, BackgroundModel bkg) {
    const int nSig = NSignalPars(sig);
    const int nBkg = NBkgPars(bkg);
    SetSignalParameterNames(model, sig);
    for(int i = 0; i < nBkg; ++i) model->SetParName(nSig + i, Form("b%d", i));

    const double eventsFit = std::max(HistIntegralDensity(h, fitMin, fitMax, true), 1.0);
    const double peakEvents = HistIntegralDensity(h, gYieldIntLow, gYieldIntHigh);
    const double bkgDens = std::max(SidebandMedianDensity(h, fitMin, fitMax), 1e-6);
    double nsigGuess = peakEvents - bkgDens * std::max(gYieldIntHigh - gYieldIntLow, 1e-6);
    if(!std::isfinite(nsigGuess) || nsigGuess <= 0.) nsigGuess = 0.5 * eventsFit;
    nsigGuess = std::max(1e-6, std::min(nsigGuess, 0.95 * eventsFit));

    model->SetParameter(0, nsigGuess);
    model->SetParLimits(0, 0.0, 10.0 * eventsFit + 100.0);

    if(commonShape && commonShape->ok && (int)commonShape->pars.size() >= nSig) {
      for(int i = 1; i < nSig; ++i) model->SetParameter(i, commonShape->pars[i]);
      if(fixCommonShape) {
        for(int i = 1; i < nSig; ++i) model->FixParameter(i, commonShape->pars[i]);
      }
    }
    else {
      model->SetParameter(1, gPeakMass);
      model->SetParameter(2, gSignalSigmaInit);
      model->SetParameter(3, 1.6);
      model->SetParameter(4, 5.0);
      if(sig == kDSCB) {
        model->SetParameter(5, 2.0);
        model->SetParameter(6, 5.0);
      }
    }

    if(!(commonShape && commonShape->ok && fixCommonShape)) {
      model->SetParLimits(1, gMeanFitLow, gMeanFitHigh);
      model->SetParLimits(2, gSignalSigmaMin, gSignalSigmaMax);
      model->SetParLimits(3, 0.4, 6.0);
      model->SetParLimits(4, 1.05, 60.0);
      if(sig == kDSCB) {
        model->SetParLimits(5, 0.4, 6.0);
        model->SetParLimits(6, 1.05, 60.0);
      }
    }

    const double maxInFit = std::max(HistMaxInRange(h, fitMin, fitMax), bkgDens);
    if(UsesLogNorm(bkg)) {
      model->SetParameter(nSig + 0, std::log(std::max(bkgDens, 1e-9)));
      model->SetParLimits(nSig + 0, -30.0, std::log(std::max(maxInFit * 100.0, 1.0)));
    }
    else {
      model->SetParameter(nSig + 0, bkgDens);
      model->SetParLimits(nSig + 0, 0.0, std::max(maxInFit * 100.0, 1.0));
    }

    for(int i = 1; i < nBkg; ++i) {
      model->SetParameter(nSig + i, 0.0);
      if(IsMonoExpBackground(bkg)) {
        model->SetParameter(nSig + i, 0.0);
        model->SetParLimits(nSig + i, -5.0, 5.0);
      }
      else if(IsExpBackground(bkg)) {
        if(i == 1)      model->SetParLimits(nSig + i, -15.0, 15.0);
        else if(i == 2) model->SetParLimits(nSig + i, -50.0, 50.0);
        else            model->SetParLimits(nSig + i, -80.0, 80.0);
      }
      else if(IsMonoChebBackground(bkg)) {
        model->SetParLimits(nSig + i, -1.0, 1.0);
      }
      else if(IsChebBackground(bkg)) {
        model->SetParLimits(nSig + i, -2.0, 2.0);
      }
      else if(IsMonoBernBackground(bkg)) {
        model->SetParameter(nSig + i, 0.3);
        model->SetParLimits(nSig + i, 0.0, 5.0);
      }
      else if(IsBernBackground(bkg)) {
        const int order = BernOrder(bkg);
        const double frac = double(i) / double(std::max(order, 1));
        model->SetParameter(nSig + i, -2.0 * frac);
        model->SetParLimits(nSig + i, -8.0, 8.0);
      }
    }
  }

  TGraphErrors* BuildSidebandGraph(TH1D *h, const TString &name,
                                   bool useLogDensity = false,
                                   bool applyFitWeights = false) {
    if(!h) return nullptr;

    TGraphErrors *gr = new TGraphErrors();
    gr->SetName(name);

    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(!IsInBkgSideband(x)) continue;

      const double y = h->GetBinContent(ibin);
      if(!std::isfinite(y) || y <= 0.) continue;

      double ey = h->GetBinError(ibin);
      if(!std::isfinite(ey) || ey <= 0.) {
        const double width = std::max(h->GetXaxis()->GetBinWidth(ibin), 1e-9);
        ey = std::sqrt(std::max(y * width, 1.0)) / width;
      }

      if(applyFitWeights) {
        ey = std::max(ey, gMinBkgRelErr * y);
        ey /= std::sqrt(BkgSidebandWeight(x));
      }

      const int n = gr->GetN();
      if(useLogDensity) {
        gr->SetPoint(n, x, std::log(y));
        gr->SetPointError(n, 0.0, std::max(ey / y, 1e-6));
      }
      else {
        gr->SetPoint(n, x, y);
        gr->SetPointError(n, 0.0, ey);
      }
    }
    return gr;
  }

  void ConfigureBackgroundOnly(TF1 *bkgFn, TH1D *h, BackgroundModel bkg) {
    if(!bkgFn) return;

    const int nBkg = NBkgPars(bkg);
    for(int i = 0; i < nBkg; ++i) bkgFn->SetParName(i, Form("b%d", i));

    const double bkgDens = std::max(SidebandMedianDensity(h, gBkgFitMin, gBkgFitMax), 1e-9);
    const double maxDens = std::max(HistMaxInRange(h, gBkgFitMin, gBkgFitMax), bkgDens);
    const double refDens = std::max(EstimateSidebandDensity(h, gPeakMass, bkgDens), 1e-9);

    if(UsesLogNorm(bkg)) {
      bkgFn->SetParameter(0, std::log(IsBernBackground(bkg) ? refDens : bkgDens));
      bkgFn->SetParLimits(0, -30.0, std::log(std::max(maxDens * 100.0, 1.0)));
    }
    else {
      bkgFn->SetParameter(0, bkgDens);
      bkgFn->SetParLimits(0, 0.0, std::max(maxDens * 100.0, 1.0));
    }

    for(int i = 1; i < nBkg; ++i) {
      bkgFn->SetParameter(i, 0.0);
      if(IsMonoExpBackground(bkg)) {
        bkgFn->SetParameter(i, 0.0);
        bkgFn->SetParLimits(i, -5.0, 5.0);
      }
      else if(IsExpBackground(bkg)) {
        if(i == 1)      bkgFn->SetParLimits(i, -15.0, 15.0);
        else if(i == 2) bkgFn->SetParLimits(i, -50.0, 50.0);
        else            bkgFn->SetParLimits(i, -80.0, 80.0);
      }
      else if(IsMonoChebBackground(bkg)) {
        bkgFn->SetParLimits(i, -1.0, 1.0);
      }
      else if(IsChebBackground(bkg)) {
        bkgFn->SetParLimits(i, -2.0, 2.0);
      }
      else if(IsMonoBernBackground(bkg)) {
        bkgFn->SetParameter(i, 0.3);
        bkgFn->SetParLimits(i, 0.0, 5.0);
      }
      else if(IsBernBackground(bkg)) {
        const int order = BernOrder(bkg);
        const double tnode = double(i) / double(std::max(order, 1));
        const double xnode = gBkgFitMin + tnode * (gBkgFitMax - gBkgFitMin);
        const double ynode = std::max(EstimateSidebandDensity(h, xnode, bkgDens), 1e-9);
        const double logCoeff = std::max(-5.0, std::min(5.0, std::log(ynode / refDens)));
        bkgFn->SetParameter(i, logCoeff);
        bkgFn->SetParLimits(i, -8.0, 8.0);
      }
    }
  }

  double ComputeGraphChi2(TGraphErrors *gr, TF1 *fn, double xmin, double xmax, int &nPoints) {
    nPoints = 0;
    if(!gr || !fn) return 0.;

    double chi2 = 0.;
    for(int i = 0; i < gr->GetN(); ++i) {
      double x = 0.;
      double y = 0.;
      gr->GetPoint(i, x, y);
      if(x < xmin || x >= xmax) continue;
      const double ey = gr->GetErrorY(i);
      if(!std::isfinite(y) || !std::isfinite(ey) || ey <= 0.) continue;
      const double f = fn->Eval(x);
      if(!std::isfinite(f)) continue;
      const double pull = (y - f) / ey;
      chi2 += pull * pull;
      ++nPoints;
    }
    return chi2;
  }

  void DrawBkgSidebandFitPlot(TH1D *h, TF1 *bkgFn, TGraphErrors *gr, const BkgOutput &out,
                              const TString &label, const TString &year,
                              const TString &sample, const TString &status,
                              const TString &outDir, BackgroundModel bkg) {
    if(!h || !bkgFn) return;

    gBackgroundModel = bkg;
    gBkgShapeMin = gBkgFitMin;
    gBkgShapeMax = gBkgFitMax;

    h->SetStats(0);
    h->SetMarkerStyle(20);
    h->SetMarkerSize(0.55);
    h->SetLineColor(kBlack);
    h->SetMarkerColor(kBlack);
    bkgFn->SetLineColor(kGreen + 2);
    bkgFn->SetLineWidth(2);
    bkgFn->SetNpx(1000);

    TCanvas *c = new TCanvas((TString("c_bkg_") + Sanitise(label + sample + status)).Data(), "", 900, 800);
    c->SetLeftMargin(0.12);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetLogy(1);

    double yminPos = 1e99;
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(x < gBkgFitMin || x >= gBkgFitMax) continue;
      const double y = h->GetBinContent(ibin);
      if(y > 0.) yminPos = std::min(yminPos, y);
    }
    const double ymax = std::max(HistMaxInRange(h, gBkgFitMin, gBkgFitMax), 1.0) * 20.0;
    const double ymin = (yminPos < 1e98) ? std::max(0.05 * yminPos, 1e-3) : 1e-3;

    TH1D *frame = CloneAsTH1D(h, TString("frame_bkg_") + Sanitise(label + sample + status));
    frame->GetXaxis()->SetRangeUser(gBkgFitMin, gBkgFitMax);
    frame->GetYaxis()->SetRangeUser(ymin, ymax);
    frame->GetXaxis()->SetTitle("m(#mu#mu) [GeV]");
    frame->GetYaxis()->SetTitle("Events / GeV");
    frame->GetYaxis()->SetTitleOffset(0.95);
    frame->Draw("AXIS");

    auto drawBox = [&](const double x1, const double x2, Color_t color, double alpha) {
      TBox *box = new TBox(x1, ymin, x2, ymax);
      box->SetFillColorAlpha(color, alpha);
      box->SetLineColor(color);
      box->SetLineStyle(0);
      box->Draw("SAME");
    };

    if(gResonanceMode == kResJpsi) {
      drawBox(kJpsiVetoLow, kJpsiVetoHigh, kRed - 9, 0.18);
      drawBox(kPsiPVetoLow, kPsiPVetoHigh, kOrange - 9, 0.20);
    }
    else {
      drawBox(gYieldIntLow, gYieldIntHigh, kRed - 9, 0.18);
    }

    h->Draw("PE SAME");
    if(gr && gr->GetN() > 0) {
      gr->SetMarkerStyle(20);
      gr->SetMarkerSize(0.65);
      gr->SetLineColor(kBlack);
      gr->SetMarkerColor(kBlack);
      gr->Draw("PE SAME");
    }
    bkgFn->Draw("SAME");
    frame->Draw("AXIS SAME");

    TLegend *leg = new TLegend(0.7, 0.75, 0.90, 0.89);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.032);
    leg->AddEntry(h, sample + " " + status, "lep");
    leg->AddEntry(bkgFn, BackgroundModelName(bkg) + " fit", "l");
    leg->Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(0.038);
    latex.DrawLatex(0.12, 0.94, "#bf{CMS} #it{Preliminary}");
    latex.SetTextAlign(31);
    latex.DrawLatex(0.96, 0.94, LumiText(year));
    latex.SetTextAlign(13);
    latex.SetTextSize(0.030);
    latex.DrawLatex(0.15, 0.87, label);
    latex.DrawLatex(0.15, 0.83, TString("bkg sidebands: ") + SidebandRangesText() + " GeV");
    //latex.DrawLatex(0.15, 0.79, "excluded: J/#psi and #psi(2S) peak regions");
    if(out.ndf > 0) latex.DrawLatex(0.15, 0.79, Form("#chi^{2}/ndf = %.2f", out.chi2/out.ndf));
    //latex.DrawLatex(0.15, 0.64, Form("fit status = %d, cov = %d, N_{side} = %d", out.fitStatus, out.covStatus, out.nPoints));
    latex.DrawLatex(0.15, 0.75, gUseLogBkgFit ? "sideband fit in log(Events/GeV)" : "sideband fit in Events/GeV");
    //if(gSide2FitWeight > 1.0) latex.DrawLatex(0.15, 0.54, Form("[%.1f,%.1f] GeV weight = %.1f", kSide2Low, kSide2High, gSide2FitWeight));

    const TString safe = Sanitise(label);
    c->SaveAs(outDir + "/bkgfit_" + sample + "_" + status + "_" + safe + ".png");
    c->SaveAs(outDir + "/bkgfit_" + sample + "_" + status + "_" + safe + ".pdf");

    delete c;
  }

  BkgOutput FitBackgroundSidebands(TH1D *h, const TString &label, const TString &year,
                                   const TString &sample, const TString &status,
                                   const TString &outDir, BackgroundModel bkg,
                                   bool savePlot = true) {
    BkgOutput out;
    if(!h) return out;

    gBackgroundModel = bkg;
    gBkgShapeMin = gBkgFitMin;
    gBkgShapeMax = gBkgFitMax;

    TGraphErrors *grFit = BuildSidebandGraph(h, TString("gr_bkg_fit_") + Sanitise(label + sample + status),
                                             gUseLogBkgFit, true);
    TGraphErrors *grPlot = BuildSidebandGraph(h, TString("gr_bkg_plot_") + Sanitise(label + sample + status),
                                              false, false);
    if(!grFit || grFit->GetN() <= NBkgPars(bkg)) {
      cout << "[WARNING] Too few sideband points for bkg fit: " << sample << " " << status
           << " " << label << ", Npoints=" << (grFit ? grFit->GetN() : 0) << endl;
      delete grFit;
      delete grPlot;
      return out;
    }

    const int nBkg = NBkgPars(bkg);
    Double_t (*fitFunction)(Double_t *, Double_t *) = gUseLogBkgFit ? GenericBackgroundLog : GenericBackground;
    TF1 *fitFn = new TF1((TString("bkgOnlyFit_") + Sanitise(label + sample + status)).Data(),
                         fitFunction, gBkgFitMin, gBkgFitMax, nBkg);
    fitFn->SetNpx(1000);
    ConfigureBackgroundOnly(fitFn, h, bkg);

    TFitResultPtr fitRes = grFit->Fit(fitFn, "SRQ0");
    fitRes = grFit->Fit(fitFn, "SRQ0");

    out.fitStatus = int(fitRes);
    if(fitRes.Get()) out.covStatus = fitRes->CovMatrixStatus();
    out.pars.resize(nBkg);
    out.errs.resize(nBkg);
    for(int i = 0; i < nBkg; ++i) {
      out.pars[i] = fitFn->GetParameter(i);
      out.errs[i] = fitFn->GetParError(i);
    }
    out.chi2 = ComputeGraphChi2(grFit, fitFn, gBkgFitMin, gBkgFitMax, out.nPoints);
    out.ndf = out.nPoints - fitFn->GetNumberFreeParameters();

    out.ok = (out.nPoints > fitFn->GetNumberFreeParameters());
    for(double v : out.pars) out.ok = out.ok && std::isfinite(v);
    if(!out.ok || out.fitStatus != 0) {
      cout << "[WARNING] Background sideband fit diagnostic: " << sample << " " << status << " " << label
           << ", ok=" << out.ok << ", status=" << out.fitStatus
           << ", cov=" << out.covStatus << ", nPoints=" << out.nPoints << endl;
    }

    TF1 *drawFn = new TF1((TString("bkgOnlyDraw_") + Sanitise(label + sample + status)).Data(),
                          GenericBackground, gBkgFitMin, gBkgFitMax, nBkg);
    drawFn->SetNpx(1000);
    for(int i = 0; i < nBkg; ++i) drawFn->SetParameter(i, fitFn->GetParameter(i));

    if(savePlot) DrawBkgSidebandFitPlot(h, drawFn, grPlot, out, label, year, sample, status, outDir, bkg);

    delete drawFn;
    delete fitFn;
    delete grFit;
    delete grPlot;
    return out;
  }

  void ApplyBkgPrefitToModel(TF1 *model, const BkgOutput *bkgPrefit,
                             SignalModel sig, BackgroundModel bkg,
                             bool fixShapeFromSidebands) {
    if(!model || !bkgPrefit || !bkgPrefit->ok) return;

    const int nSig = NSignalPars(sig);
    const int nBkg = NBkgPars(bkg);
    if((int)bkgPrefit->pars.size() < nBkg) return;

    for(int i = 0; i < nBkg; ++i) model->SetParameter(nSig + i, bkgPrefit->pars[i]);

    // Keep the overall background normalisation near the sideband-prefit value.
    // It is not fixed, but it is allowed to move only by +/-5% in the final
    // signal+background fit.  For log-normalisation models this is applied as
    // p0 + log(0.95) ... p0 + log(1.05).
    if(UsesLogNorm(bkg)) {
      const double p0 = bkgPrefit->pars[0];
      model->SetParLimits(nSig + 0,
                          p0 + std::log(std::max(1e-9, 1.0 - kBkgNormRelConstraint)),
                          p0 + std::log(1.0 + kBkgNormRelConstraint));
    }
    else {
      const double p0 = bkgPrefit->pars[0];
      if(p0 > 0.) {
        model->SetParLimits(nSig + 0,
                            (1.0 - kBkgNormRelConstraint) * p0,
                            (1.0 + kBkgNormRelConstraint) * p0);
      }
      else {
        const double delta = std::max(1e-9, kBkgNormRelConstraint * std::fabs(p0));
        model->SetParLimits(nSig + 0, p0 - delta, p0 + delta);
      }
    }

    if(fixShapeFromSidebands) {
      for(int i = 1; i < nBkg; ++i) model->FixParameter(nSig + i, bkgPrefit->pars[i]);
    }
  }

  void DrawFitPlot(TH1D *h, TF1 *model, const FitOutput &out,
                   const TString &label, const TString &year,
                   const TString &sample, const TString &status,
                   const TString &outDir, double fitMin, double fitMax,
                   SignalModel sig, BackgroundModel bkg) {
    if(!h || !model) return;

    const int nSig = NSignalPars(sig);
    const int nBkg = NBkgPars(bkg);
    gSignalModel = sig;
    gBackgroundModel = bkg;
    gFitMin = fitMin;
    gFitMax = fitMax;

    TF1 *sigFn = new TF1((TString("sig_") + Sanitise(label + sample + status)).Data(), GenericSignal, fitMin, fitMax, nSig);
    TF1 *bkgFn = new TF1((TString("bkg_") + Sanitise(label + sample + status)).Data(), GenericBackground, fitMin, fitMax, nBkg);
    for(int i = 0; i < nSig; ++i) sigFn->SetParameter(i, model->GetParameter(i));
    for(int i = 0; i < nBkg; ++i) bkgFn->SetParameter(i, model->GetParameter(nSig + i));
    sigFn->SetLineColor(kBlue + 1);
    sigFn->SetLineStyle(2);
    sigFn->SetLineWidth(2);
    bkgFn->SetLineColor(kGreen + 2);
    bkgFn->SetLineStyle(3);
    bkgFn->SetLineWidth(2);

    h->SetStats(0);
    h->SetMarkerStyle(20);
    h->SetMarkerSize(0.65);
    h->SetLineColor(kBlack);
    h->SetMarkerColor(kBlack);
    model->SetLineColor(kRed + 1);
    model->SetLineWidth(2);

    TCanvas *c = new TCanvas((TString("c_") + Sanitise(label + sample + status)).Data(), "", 900, 850);
    TPad *upper = new TPad((TString("upper_") + Sanitise(label + sample + status)).Data(), "", 0., 0.30, 1., 1.);
    TPad *lower = new TPad((TString("lower_") + Sanitise(label + sample + status)).Data(), "", 0., 0.00, 1., 0.32);
    upper->SetLeftMargin(0.12);
    upper->SetRightMargin(0.04);
    upper->SetBottomMargin(0.03);
    upper->SetLogy(1);
    lower->SetLeftMargin(0.12);
    lower->SetRightMargin(0.04);
    lower->SetTopMargin(0.04);
    lower->SetBottomMargin(0.32);
    upper->Draw();
    lower->Draw();

    upper->cd();
    double yminPos = 1e99;
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(x < fitMin || x >= fitMax) continue;
      const double y = h->GetBinContent(ibin);
      if(y > 0.) yminPos = std::min(yminPos, y);
    }
    const double ymax = std::max(HistMaxInRange(h, fitMin, fitMax), 1.0) * 25.0;
    const double ymin = (yminPos < 1e98) ? std::max(0.05 * yminPos, 1e-3) : 1e-3;

    TH1D *frame = CloneAsTH1D(h, TString("frame_") + Sanitise(label + sample + status));
    frame->GetXaxis()->SetRangeUser(fitMin, fitMax);
    frame->GetYaxis()->SetRangeUser(ymin, ymax);
    frame->GetXaxis()->SetLabelSize(0.0);
    frame->GetYaxis()->SetTitle("Events / GeV");
    frame->GetYaxis()->SetTitleOffset(1.);
    frame->Draw("AXIS");
    h->Draw("PE SAME");
    model->Draw("SAME");
    sigFn->Draw("SAME");
    bkgFn->Draw("SAME");

    TLegend *leg = new TLegend(0.58, 0.67, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.032);
    leg->AddEntry(h, sample + " " + status, "lep");
    leg->AddEntry(model, SignalModelName(sig) + " + " + BackgroundModelName(bkg), "l");
    leg->AddEntry(sigFn, gResonanceLabel + " signal", "l");
    leg->AddEntry(bkgFn, "continuum bkg", "l");
    leg->Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(0.038);
    latex.DrawLatex(0.12, 0.94, "#bf{CMS} #it{Preliminary}");
    latex.SetTextAlign(31);
    latex.DrawLatex(0.96, 0.94, LumiText(year));
    latex.SetTextAlign(13);
    latex.SetTextSize(0.030);
    latex.DrawLatex(0.15, 0.84, label);
    if(UseFitNormYield()) {
      latex.DrawLatex(0.15, 0.79, Form("N_{%s}^{fit} = %.3g #pm %.2g", gResonanceLabel.Data(), out.yield, out.yieldErr));
    }
    else {
      latex.DrawLatex(0.15, 0.79, Form("N_{%s}^{%.1f-%.1f} = %.3g #pm %.2g", gResonanceLabel.Data(), gYieldIntLow, gYieldIntHigh, out.yield, out.yieldErr));
    }
    latex.DrawLatex(0.15, 0.74, Form("m = %.4f GeV, #sigma = %.4f GeV", out.mean, out.sigma));
    if(out.ndf > 0) latex.DrawLatex(0.15, 0.69, Form("weighted #chi^{2}/ndf = %.1f/%d = %.2f", out.chi2, out.ndf, out.chi2/out.ndf));
    latex.DrawLatex(0.15, 0.64, Form("fit status = %d, cov = %d%s", out.fitStatus, out.covStatus, out.usedCommonShape ? ", shared P/F shape" : ""));

    lower->cd();
    TH1D *ratioFrame = new TH1D((TString("ratioFrame_") + Sanitise(label + sample + status)).Data(), "", 1, fitMin, fitMax);
    ratioFrame->SetStats(0);
    ratioFrame->GetXaxis()->SetTitle("m(#mu#mu) [GeV]");
    ratioFrame->GetYaxis()->SetTitle("Data / fit");
    ratioFrame->GetYaxis()->SetRangeUser(0.8, 1.2);
    ratioFrame->GetXaxis()->SetTitleSize(0.11);
    ratioFrame->GetXaxis()->SetLabelSize(0.09);
    ratioFrame->GetYaxis()->SetTitleSize(0.09);
    ratioFrame->GetYaxis()->SetLabelSize(0.075);
    ratioFrame->GetYaxis()->SetTitleOffset(0.50);
    ratioFrame->GetYaxis()->SetNdivisions(505);
    ratioFrame->Draw("AXIS");

    TGraphErrors *gr = new TGraphErrors();
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(x < fitMin || x >= fitMax) continue;
      const double y = h->GetBinContent(ibin);
      const double ey = h->GetBinError(ibin);
      const double f = model->Eval(x);
      if(f <= 0. || !std::isfinite(f) || y <= 0.) continue;
      const int n = gr->GetN();
      gr->SetPoint(n, x, y / f);
      gr->SetPointError(n, 0.5 * h->GetXaxis()->GetBinWidth(ibin), ey / f);
    }
    gr->SetMarkerStyle(20);
    gr->SetMarkerSize(0.55);
    gr->SetLineColor(kBlack);
    gr->SetMarkerColor(kBlack);
    gr->Draw("PE SAME");

    TLine *one = new TLine(fitMin, 1.0, fitMax, 1.0);
    one->SetLineStyle(2);
    one->SetLineColor(kGray + 2);
    one->Draw("SAME");
    ratioFrame->Draw("AXIS SAME");

    const TString safe = Sanitise(label);
    const TString outPng = outDir + "/fit_" + sample + "_" + status + "_" + safe + ".png";
    const TString outPdf = outDir + "/fit_" + sample + "_" + status + "_" + safe + ".pdf";
    c->SaveAs(outPng);
    c->SaveAs(outPdf);

    delete c;
    delete sigFn;
    delete bkgFn;
  }

  FitOutput FitOne(TH1D *h, const TString &label, const TString &year,
                   const TString &sample, const TString &status, const TString &outDir,
                   SignalModel sig, BackgroundModel bkg,
                   double fitMin, double fitMax,
                   const FitOutput *commonShape = nullptr,
                   bool fixCommonShape = false,
                   const BkgOutput *bkgPrefit = nullptr,
                   bool fixBkgShapeFromSidebands = true,
                   bool savePlot = true) {
    FitOutput out;
    if(!h) return out;

    gSignalModel = sig;
    gBackgroundModel = bkg;
    gFitMin = fitMin;
    gFitMax = fitMax;

    out.eventsInFitRange = HistIntegralDensity(h, fitMin, fitMax, true);
    if(out.eventsInFitRange <= 0.) {
      cout << "[WARNING] Empty fit range: " << sample << " " << status << " " << label << endl;
      return out;
    }

    const int nSig = NSignalPars(sig);
    const int nBkg = NBkgPars(bkg);
    const int nPar = nSig + nBkg;

    TF1 *model = new TF1((TString("model_") + Sanitise(label + sample + status)).Data(),
                         GenericModel, fitMin, fitMax, nPar);
    model->SetNpx(1000);
    ConfigureModel(model, h, commonShape, fixCommonShape, fitMin, fitMax, sig, bkg);
    ApplyBkgPrefitToModel(model, bkgPrefit, sig, bkg, fixBkgShapeFromSidebands);

    gRejectPsiPInFinalFit = true;
    TFitResultPtr fitRes = h->Fit(model, "SRQ0", "", fitMin, fitMax);
    fitRes = h->Fit(model, "SRQ0", "", fitMin, fitMax);
    gRejectPsiPInFinalFit = false;

    out.fitStatus = int(fitRes);
    if(fitRes.Get()) out.covStatus = fitRes->CovMatrixStatus();
    out.usedCommonShape = (commonShape && commonShape->ok && fixCommonShape);
    out.rawNorm = model->GetParameter(0);
    out.rawNormErr = model->GetParError(0);
    if(UseFitNormYield()) {
      // CMS-POG-like fitted signal-yield definition:
      // use the signal normalisation parameter of the normalised signal PDF.
      out.yield = out.rawNorm;
      out.yieldErr = out.rawNormErr;
    }
    else {
      // Core-window yield definition:
      // use the fitted signal component integrated in [gYieldIntLow,gYieldIntHigh].
      out.yield = SignalIntegralFromModel(model, sig, gYieldIntLow, gYieldIntHigh);
      out.yieldErr = SignalIntegralErrorFromFit(model, fitRes, sig, gYieldIntLow, gYieldIntHigh);
    }
    if(!std::isfinite(out.yieldErr) || out.yieldErr <= 0.) {
      out.yieldErr = std::sqrt(std::max(0.0, out.yield));
    }
    out.mean = model->GetParameter(1);
    out.sigma = std::fabs(model->GetParameter(2));
    out.chi2 = model->GetChisquare();
    out.ndf = model->GetNDF();
    out.pars.resize(nPar);
    out.errs.resize(nPar);
    for(int i = 0; i < nPar; ++i) {
      out.pars[i] = model->GetParameter(i);
      out.errs[i] = model->GetParError(i);
    }

    out.ok = std::isfinite(out.yield) && out.yield >= 0. &&
             std::isfinite(out.yieldErr) && out.yieldErr >= 0. &&
             std::isfinite(out.mean) && std::isfinite(out.sigma) &&
             out.sigma > 0. && out.eventsInFitRange > 0.;

    if(!out.ok || out.fitStatus != 0) {
      cout << "[WARNING] Fit diagnostic: " << sample << " " << status << " " << label
           << ", ok=" << out.ok << ", status=" << out.fitStatus
           << ", cov=" << out.covStatus << ", N(core)=" << out.yield
           << ", A(raw)=" << out.rawNorm << endl;
    }

    if(savePlot) DrawFitPlot(h, model, out, label, year, sample, status, outDir, fitMin, fitMax, sig, bkg);

    delete model;
    return out;
  }

  void FillWilsonFloor(EffOutput &out, const double nCount);

  TH1D* BuildSimultaneousLikelihoodHist(TH1D *hPass, TH1D *hFail,
                                             const TString &name,
                                             double fitMin, double fitMax) {
    if(!hPass || !hFail) return nullptr;
    if(hPass->GetNbinsX() != hFail->GetNbinsX()) {
      cout << "[WARNING] Pass/Fail mass histograms have different bin counts in "
           << name << endl;
      return nullptr;
    }

    int firstBin = -1;
    int lastBin = -1;
    for(int ibin = 1; ibin <= hPass->GetNbinsX(); ++ibin) {
      const double mass = hPass->GetXaxis()->GetBinCenter(ibin);
      if(mass < fitMin || mass >= fitMax) continue;
      if(gUseFinalVeto && IsInFinalFitVeto(mass)) continue;
      if(firstBin < 0) firstBin = ibin;
      lastBin = ibin;
    }
    if(firstBin < 0 || lastBin < firstBin) return nullptr;

    // The current resonance configurations have no internal veto inside the
    // final fit range.  Require contiguous, equal-width mass bins so the two
    // categories can be concatenated without artificial likelihood bins.
    const double binWidth = hPass->GetXaxis()->GetBinWidth(firstBin);
    if(!(binWidth > 0.) || !std::isfinite(binWidth)) return nullptr;

    int nMassBins = 0;
    for(int ibin = firstBin; ibin <= lastBin; ++ibin) {
      const double mass = hPass->GetXaxis()->GetBinCenter(ibin);
      if(gUseFinalVeto && IsInFinalFitVeto(mass)) {
        cout << "[WARNING] Internal final-fit veto is not supported by the combined likelihood histogram." << endl;
        return nullptr;
      }
      const double bwP = hPass->GetXaxis()->GetBinWidth(ibin);
      const double bwF = hFail->GetXaxis()->GetBinWidth(ibin);
      if(std::fabs(bwP - binWidth) > 1e-9 || std::fabs(bwF - binWidth) > 1e-9) {
        cout << "[WARNING] Non-uniform Pass/Fail mass binning in " << name << endl;
        return nullptr;
      }
      ++nMassBins;
    }

    const double xMin = hPass->GetXaxis()->GetBinLowEdge(firstBin);
    const double span = nMassBins * binWidth;
    gSimultaneousOffset = span;
    gSimultaneousSplit = xMin + span;
    gSimultaneousBinWidth = binWidth;

    TH1D *joint = new TH1D(name.Data(), "", 2 * nMassBins, xMin, xMin + 2.0 * span);
    joint->SetDirectory(0);
    joint->Sumw2();

    for(int j = 0; j < nMassBins; ++j) {
      const int src = firstBin + j;
      const double bw = hPass->GetXaxis()->GetBinWidth(src);

      // RebinAndMakeDensity(...) divided both content and error by bw.
      // Restore sum(w) and sqrt(sum(w^2)) here.  Data uses these as event
      // counts for the Poisson likelihood; MC uses the content and SumW2 error
      // in the SumW2 chi2, which also permits signed weighted bins.
      const double passY = hPass->GetBinContent(src) * bw;
      const double passE = hPass->GetBinError(src) * bw;
      const double failY = hFail->GetBinContent(src) * bw;
      const double failE = hFail->GetBinError(src) * bw;

      joint->SetBinContent(j + 1, passY);
      joint->SetBinError(j + 1, std::max(0.0, passE));
      joint->SetBinContent(nMassBins + j + 1, failY);
      joint->SetBinError(nMassBins + j + 1, std::max(0.0, failE));
    }

    return joint;
  }

  void CopyBackgroundSetupToSimultaneous(TF1 *simModel, const int dstStart,
                                         TH1D *h, const FitOutput *shapeSeed,
                                         const BkgOutput *bkgPrefit,
                                         SignalModel sig, BackgroundModel bkg,
                                         double fitMin, double fitMax,
                                         bool fixBkgShapeFromSidebands,
                                         const TString &tag) {
    if(!simModel || !h) return;
    const int nSig = NSignalPars(sig);
    const int nBkg = NBkgPars(bkg);

    static int tmpCounter = 0;
    TF1 tmp((TString("tmpSimBkg_") + tag + "_" + TString::Itoa(tmpCounter++, 10)).Data(),
            GenericModel, fitMin, fitMax, nSig + nBkg);
    ConfigureModel(&tmp, h, shapeSeed, false, fitMin, fitMax, sig, bkg);
    ApplyBkgPrefitToModel(&tmp, bkgPrefit, sig, bkg, fixBkgShapeFromSidebands);

    for(int i = 0; i < nBkg; ++i) {
      const int src = nSig + i;
      const int dst = dstStart + i;
      const double value = tmp.GetParameter(src);
      double lo = 0.;
      double hi = 0.;
      tmp.GetParLimits(src, lo, hi);
      simModel->SetParName(dst, Form("%s_b%d", tag.Data(), i));
      simModel->SetParameter(dst, value);
      if(std::fabs(hi - lo) < 1e-14) simModel->FixParameter(dst, value);
      else simModel->SetParLimits(dst, lo, hi);
    }
  }

  double CategoryChi2(TH1D *h, TF1 *model, double fitMin, double fitMax, int &nPoints) {
    nPoints = 0;
    if(!h || !model) return 0.;
    double chi2 = 0.;
    for(int ibin = 1; ibin <= h->GetNbinsX(); ++ibin) {
      const double x = h->GetXaxis()->GetBinCenter(ibin);
      if(x < fitMin || x >= fitMax) continue;
      if(gUseFinalVeto && IsInFinalFitVeto(x)) continue;
      const double y = h->GetBinContent(ibin);
      double ey = h->GetBinError(ibin);
      if(!std::isfinite(y)) continue;
      if(!std::isfinite(ey) || ey <= 0.) {
        const double bw = std::max(h->GetXaxis()->GetBinWidth(ibin), 1e-9);
        ey = std::sqrt(std::max(std::fabs(y) * bw, 1.0)) / bw;
      }
      const double f = model->Eval(x);
      if(!std::isfinite(f)) continue;
      const double pull = (y - f) / ey;
      chi2 += pull * pull;
      ++nPoints;
    }
    return chi2;
  }

  SimultaneousOutput FitPassFailSimultaneous(
      TH1D *hPass, TH1D *hFail,
      const TString &label, const TString &year,
      const TString &sample, const TString &outDir,
      SignalModel sig, BackgroundModel bkg,
      double fitMin, double fitMax,
      const FitOutput *allShape,
      const FitOutput *passSeed,
      const FitOutput *failSeed,
      const BkgOutput *passBkg,
      const BkgOutput *failBkg,
      bool fixSharedSignalShape,
      bool fixBkgShapeFromSidebands,
      bool savePlots,
      bool computeStatError = true) {

    SimultaneousOutput out;
    // A true simultaneous fit needs both categories.  If a histogram is
    // genuinely absent (rather than present with zero entries), fall back to
    // the separate-yield + Wilson treatment below instead of fitting a
    // degenerate joint model.
    if(!hPass || !hFail) return out;

    gSignalModel = sig;
    gBackgroundModel = bkg;
    gFitMin = fitMin;
    gFitMax = fitMax;

    TH1D *jointHist = BuildSimultaneousLikelihoodHist(
      hPass, hFail, TString("hSimLike_") + Sanitise(label + sample), fitMin, fitMax);
    if(!jointHist || jointHist->GetNbinsX() < 4) {
      delete jointHist;
      return out;
    }

    const int nSig = NSignalPars(sig);
    const int nBkg = NBkgPars(bkg);
    const int passBkgStart = nSig + 1;
    const int failBkgStart = passBkgStart + nBkg;
    const int nPar = failBkgStart + nBkg;

    TF1 *model = new TF1(
      (TString("simModel_") + Sanitise(label + sample)).Data(),
      GenericSimultaneousCountModel,
      jointHist->GetXaxis()->GetXmin(), jointHist->GetXaxis()->GetXmax(), nPar);
    model->SetNpx(1600);
    model->SetParName(0, "efficiency");
    model->SetParName(1, "N_{sig,total}");

    const double pSeed =
      (passSeed && std::isfinite(passSeed->rawNorm) && passSeed->rawNorm > 0.)
      ? passSeed->rawNorm : 0.;
    const double fSeed =
      (failSeed && std::isfinite(failSeed->rawNorm) && failSeed->rawNorm > 0.)
      ? failSeed->rawNorm : 0.;
    const double seedSum = pSeed + fSeed;
    double effGuess = (seedSum > 0.) ? pSeed / seedSum : 0.95;
    effGuess = std::max(1e-3, std::min(1.0 - 1e-3, effGuess));

    const double eventsFit =
      std::max(HistIntegralDensity(hPass, fitMin, fitMax, true), 0.0) +
      (hFail ? std::max(HistIntegralDensity(hFail, fitMin, fitMax, true), 0.0) : 0.0);
    double totalGuess =
      (allShape && std::isfinite(allShape->rawNorm) && allShape->rawNorm > 0.)
      ? allShape->rawNorm : std::max(seedSum, 1.0);
    totalGuess = std::max(1e-6, std::min(totalGuess, 10.0 * std::max(eventsFit, 1.0) + 100.0));

    model->SetParameter(0, effGuess);
    model->SetParLimits(0, 0.0, 1.0);
    model->SetParameter(1, totalGuess);
    model->SetParLimits(1, 0.0, 10.0 * std::max(eventsFit, 1.0) + 100.0);

    // Shared signal-shape parameters.  The All fit is used as the seed.
    for(int i = 1; i < nSig; ++i) {
      const int dst = i + 1;
      double value = 0.;
      if(allShape && allShape->ok && (int)allShape->pars.size() > i) {
        value = allShape->pars[i];
      }
      else {
        if(i == 1) value = gPeakMass;
        else if(i == 2) value = gSignalSigmaInit;
        else if(i == 3) value = 1.6;
        else if(i == 4) value = 5.0;
        else if(i == 5) value = 2.0;
        else if(i == 6) value = 5.0;
      }

      if(i == 1) model->SetParName(dst, "m_{shared}");
      else if(i == 2) model->SetParName(dst, "#sigma_{shared}");
      else model->SetParName(dst, Form("sigShape%d", i));

      model->SetParameter(dst, value);
      if(fixSharedSignalShape && allShape && allShape->ok) {
        model->FixParameter(dst, value);
      }
      else {
        if(i == 1) model->SetParLimits(dst, gMeanFitLow, gMeanFitHigh);
        else if(i == 2) model->SetParLimits(dst, gSignalSigmaMin, gSignalSigmaMax);
        else if(i == 3) model->SetParLimits(dst, 0.4, 6.0);
        else if(i == 4) model->SetParLimits(dst, 1.05, 60.0);
        else if(i == 5) model->SetParLimits(dst, 0.4, 6.0);
        else if(i == 6) model->SetParLimits(dst, 1.05, 60.0);
      }
    }

    CopyBackgroundSetupToSimultaneous(
      model, passBkgStart, hPass, allShape, passBkg,
      sig, bkg, fitMin, fitMax, fixBkgShapeFromSidebands, "Pass");
    CopyBackgroundSetupToSimultaneous(
      model, failBkgStart, hFail ? hFail : hPass, allShape, failBkg,
      sig, bkg, fitMin, fitMax, fixBkgShapeFromSidebands, "Fail");

    // Nominal statistical fit:
    //   Data        -> Poisson binned likelihood;
    //   MC/reference -> chi2 with the stored SumW2 bin uncertainties.
    // "I" integrates the model over each mass bin.  Both objectives support
    // Minos, so the same statistical-error prescription is used throughout.
    const bool useDataLikelihood = sample.EqualTo("Data", TString::kIgnoreCase);
    const TString firstFitOpt = useDataLikelihood ? "SLIRQ0" : "SIRQ0";
    const TString errorFitOpt = useDataLikelihood ? "SLIERQ0" : "SIERQ0";

    TFitResultPtr fitRes = jointHist->Fit(model, firstFitOpt.Data());
    fitRes = jointHist->Fit(model, firstFitOpt.Data());
    if(computeStatError) fitRes = jointHist->Fit(model, errorFitOpt.Data());

    out.eff.fitStatus = int(fitRes);
    if(fitRes.Get()) out.eff.covStatus = fitRes->CovMatrixStatus();
    if(fitRes.Get()) {
      const double f = fitRes->MinFcnValue();
      out.eff.chi2 = std::isfinite(f) ? (useDataLikelihood ? 2.0 * f : f) : 0.;
    }
    out.eff.ndf = model->GetNDF();
    out.eff.eff = std::max(0.0, std::min(1.0, model->GetParameter(0)));
    out.eff.fitErr = model->GetParError(0);
    if(!std::isfinite(out.eff.fitErr) || out.eff.fitErr < 0.) out.eff.fitErr = 0.;

    if(computeStatError) {
      if(fitRes.Get() && fitRes->HasMinosError(0)) {
        out.eff.hasMinos = true;
        out.eff.minosLow = fitRes->LowerError(0);
        out.eff.minosHigh = fitRes->UpperError(0);
        const double lowAbs =
          std::isfinite(out.eff.minosLow) ? std::fabs(out.eff.minosLow) : 0.;
        const double highAbs =
          std::isfinite(out.eff.minosHigh) ? std::fabs(out.eff.minosHigh) : 0.;
        out.eff.statErr = std::max(lowAbs, highAbs);
      }
      else {
        out.eff.statErr = out.eff.fitErr;
      }
      out.eff.err = out.eff.statErr;
    }

    const double nTotal = std::max(0.0, model->GetParameter(1));
    out.eff.nCount = nTotal;
    out.eff.ok = std::isfinite(out.eff.eff) && std::isfinite(out.eff.err) &&
                 out.eff.err >= 0. && nTotal > 0.;

    // Build category models from the joint-fit parameters for diagnostics and
    // for the stored Pass/Fail signal yields.
    auto buildCategory = [&](const bool isFail, const TString &status, TH1D *h,
                             FitOutput &catOut) -> TF1* {
      if(!h) return nullptr;
      TF1 *cat = new TF1(
        (TString("jointCat_") + Sanitise(label + sample + status)).Data(),
        GenericModel, fitMin, fitMax, nSig + nBkg);
      cat->SetNpx(1000);

      const double eff = out.eff.eff;
      const double rawNorm = nTotal * (isFail ? (1.0 - eff) : eff);
      cat->SetParameter(0, rawNorm);
      for(int i = 1; i < nSig; ++i) cat->SetParameter(i, model->GetParameter(i + 1));
      const int srcBkg = isFail ? failBkgStart : passBkgStart;
      for(int i = 0; i < nBkg; ++i) cat->SetParameter(nSig + i, model->GetParameter(srcBkg + i));

      catOut.fitStatus = out.eff.fitStatus;
      catOut.covStatus = out.eff.covStatus;
      catOut.usedCommonShape = true;
      catOut.rawNorm = rawNorm;

      double varEff = 0.;
      double varN = 0.;
      double covEffN = 0.;
      if(fitRes.Get()) {
        varEff = fitRes->CovMatrix(0, 0);
        varN = fitRes->CovMatrix(1, 1);
        covEffN = fitRes->CovMatrix(0, 1);
      }
      if(isFail) {
        const double oneMinus = 1.0 - eff;
        const double varRaw =
          nTotal * nTotal * varEff +
          oneMinus * oneMinus * varN -
          2.0 * nTotal * oneMinus * covEffN;
        catOut.rawNormErr = (std::isfinite(varRaw) && varRaw > 0.) ? std::sqrt(varRaw) : 0.;
      }
      else {
        const double varRaw =
          nTotal * nTotal * varEff +
          eff * eff * varN +
          2.0 * nTotal * eff * covEffN;
        catOut.rawNormErr = (std::isfinite(varRaw) && varRaw > 0.) ? std::sqrt(varRaw) : 0.;
      }

      vector<double> sigPars(nSig, 0.);
      sigPars[0] = 1.0;
      for(int i = 1; i < nSig; ++i) sigPars[i] = model->GetParameter(i + 1);
      const double coreFraction =
        SignalIntegralFromPars(sigPars, sig, gYieldIntLow, gYieldIntHigh);

      if(UseFitNormYield()) {
        catOut.yield = rawNorm;
        catOut.yieldErr = catOut.rawNormErr;
      }
      else {
        catOut.yield = rawNorm * coreFraction;
        catOut.yieldErr = catOut.rawNormErr * coreFraction;
      }

      catOut.mean = model->GetParameter(2);
      catOut.sigma = std::fabs(model->GetParameter(3));
      catOut.eventsInFitRange = HistIntegralDensity(h, fitMin, fitMax, true);
      int nPoints = 0;
      catOut.chi2 = CategoryChi2(h, cat, fitMin, fitMax, nPoints);
      catOut.ndf = std::max(0, nPoints - 1);
      catOut.ok = std::isfinite(catOut.yield) && catOut.yield >= 0. &&
                  std::isfinite(catOut.mean) && std::isfinite(catOut.sigma);

      catOut.pars.resize(nSig + nBkg);
      catOut.errs.assign(nSig + nBkg, 0.);
      for(int i = 0; i < nSig + nBkg; ++i) catOut.pars[i] = cat->GetParameter(i);
      return cat;
    };

    TF1 *passModel = buildCategory(false, "Pass", hPass, out.pass);
    TF1 *failModel = buildCategory(true, "Fail", hFail, out.fail);

    const bool minimumValid = fitRes.Get() && fitRes->IsValid();
    const bool covarianceUsable = (out.eff.covStatus >= 2);
    const bool statErrorUsable =
      !computeStatError ||
      (std::isfinite(out.eff.statErr) && out.eff.statErr > 0.);

    // Nominal fits need a usable covariance/error.  Rebin-variation fits are
    // used only for their central efficiency, so do not reject a converged
    // minimum merely because the covariance matrix is poor.
    if(computeStatError) {
      out.ok = out.eff.ok && minimumValid && covarianceUsable && statErrorUsable;
    }
    else {
      const bool centralValueUsable =
        fitRes.Get() && int(fitRes) == 0 &&
        std::isfinite(out.eff.eff) && nTotal > 0.;
      out.ok = centralValueUsable;
    }

    if(computeStatError) {
      if(!out.ok) {
        cout << "[WARNING] Simultaneous Pass/Fail fit diagnostic: " << sample << " " << label
             << ", status=" << out.eff.fitStatus
             << ", validMinimum=" << minimumValid
             << ", cov=" << out.eff.covStatus
             << ", hasEffMinos=" << out.eff.hasMinos
             << ", eff=" << out.eff.eff << " +/- " << out.eff.statErr
             << endl;
      }
      else {
        cout << "[SIMULTANEOUS] " << sample << " " << label
             << ": eff=" << out.eff.eff << " +/- " << out.eff.statErr;
        if(out.eff.hasMinos) {
          cout << " (Minos=" << out.eff.minosLow << "/+" << out.eff.minosHigh
               << ", parabolic=" << out.eff.fitErr << ")";
        }
        else {
          cout << " (Minos unavailable, Hessian=" << out.eff.fitErr << ")";
        }
        cout << ", Nsig=" << nTotal
             << ", " << (useDataLikelihood ? "2NLL" : "chi2_{SumW2}")
             << "/ndf=" << out.eff.chi2 << "/" << out.eff.ndf
             << ", cov=" << out.eff.covStatus << endl;
      }
    }

    if(savePlots) {
      if(passModel && hPass) DrawFitPlot(hPass, passModel, out.pass, label, year, sample, "Pass",
                                         outDir, fitMin, fitMax, sig, bkg);
      if(failModel && hFail) DrawFitPlot(hFail, failModel, out.fail, label, year, sample, "Fail",
                                         outDir, fitMin, fitMax, sig, bkg);
    }

    delete passModel;
    delete failModel;
    delete model;
    delete jointHist;
    return out;
  }

  void ApplyBinningEnvelope(EffOutput &nominal,
                            const EffOutput &rebin1,
                            const EffOutput &rebin2,
                            const EffOutput &rebin4,
                            const EffOutput &rebin5,
                            const TString &sample,
                            const TString &label) {
    nominal.rebin1Ok = rebin1.ok;
    nominal.rebin2Ok = rebin2.ok;
    nominal.rebin4Ok = rebin4.ok;
    nominal.rebin5Ok = rebin5.ok;

    if(nominal.rebin1Ok) nominal.rebin1Eff = rebin1.eff;
    if(nominal.rebin2Ok) nominal.rebin2Eff = rebin2.eff;
    if(nominal.rebin4Ok) nominal.rebin4Eff = rebin4.eff;
    if(nominal.rebin5Ok) nominal.rebin5Eff = rebin5.eff;

    nominal.binningErr = 0.;
    int nValid = 0;
    auto includeVariation = [&](bool ok, double eff) {
      if(!ok || !std::isfinite(eff)) return;
      ++nValid;
      nominal.binningErr = std::max(
        nominal.binningErr, std::fabs(eff - nominal.eff));
    };

    includeVariation(nominal.rebin1Ok, nominal.rebin1Eff);
    includeVariation(nominal.rebin2Ok, nominal.rebin2Eff);
    includeVariation(nominal.rebin4Ok, nominal.rebin4Eff);
    includeVariation(nominal.rebin5Ok, nominal.rebin5Eff);

    nominal.binningOk = (nValid > 0);
    nominal.err = std::sqrt(
      nominal.statErr * nominal.statErr +
      nominal.binningErr * nominal.binningErr);

    cout << "[BINNING] " << sample << " " << label
         << ": nominal(r3)=" << nominal.eff;
    if(nominal.rebin1Ok) cout << ", r1=" << nominal.rebin1Eff; else cout << ", r1=FAIL";
    if(nominal.rebin2Ok) cout << ", r2=" << nominal.rebin2Eff; else cout << ", r2=FAIL";
    if(nominal.rebin4Ok) cout << ", r4=" << nominal.rebin4Eff; else cout << ", r4=FAIL";
    if(nominal.rebin5Ok) cout << ", r5=" << nominal.rebin5Eff; else cout << ", r5=FAIL";
    cout << ", nValid=" << nValid
         << ", stat=" << nominal.statErr
         << ", binning=" << nominal.binningErr
         << ", total=" << nominal.err << endl;

    if(nValid == 0) {
      cout << "[WARNING] No valid mass-binning variation for "
           << sample << " " << label
           << "; keep the nominal point with statistical uncertainty only." << endl;
    }
  }

  void FillWilsonFloor(EffOutput &out, const double nCount) {
    out.nCount = std::max(1e-6, nCount);
    const double z = 1.0; // approximately 68.27%
    const double z2 = z * z;
    const double invN = 1.0 / out.nCount;
    const double denom = 1.0 + z2 * invN;
    const double centre = (out.eff + 0.5 * z2 * invN) / denom;
    const double half =
      z / denom *
      std::sqrt(std::max(0.0,
                         out.eff * (1.0 - out.eff) * invN +
                         0.25 * z2 * invN * invN));

    out.wilsonLow = std::max(0.0, centre - half);
    out.wilsonHigh = std::min(1.0, centre + half);
    const double errLow = std::max(0.0, out.eff - out.wilsonLow);
    const double errHigh = std::max(0.0, out.wilsonHigh - out.eff);
    out.wilsonErr = std::max(errLow, errHigh);
  }

  EffOutput MakeEfficiency(const FitOutput &pass, const FitOutput &fail) {
    // AN-era prescription: fit Pass and Fail independently and propagate the
    // fitted signal-yield uncertainties through efficiency = P/(P+F).
    EffOutput out;
    if(!pass.ok || !fail.ok) return out;
    if(!std::isfinite(pass.yield) || !std::isfinite(fail.yield)) return out;

    const double p = std::max(0.0, pass.yield);
    const double f = std::max(0.0, fail.yield);
    const double total = p + f;
    if(!(total > 0.)) return out;

    const double sigmaP =
      (std::isfinite(pass.yieldErr) && pass.yieldErr >= 0.) ? pass.yieldErr : std::sqrt(p);
    const double sigmaF =
      (std::isfinite(fail.yieldErr) && fail.yieldErr >= 0.) ? fail.yieldErr : std::sqrt(f);

    out.eff = p / total;
    const double dEdP = f / (total * total);
    const double dEdF = -p / (total * total);
    const double var =
      dEdP * dEdP * sigmaP * sigmaP +
      dEdF * dEdF * sigmaF * sigmaF;

    out.fitErr = (std::isfinite(var) && var >= 0.) ? std::sqrt(var) : 0.;
    out.statErr = out.fitErr;
    out.err = out.statErr;
    out.nCount = total;
    out.ok = std::isfinite(out.eff) && std::isfinite(out.err) && out.err >= 0.;
    return out;
  }


  EffOutput FitSeparateEfficiencyVariation(
      TH1D *hPass, TH1D *hFail,
      const TString &label, const TString &year,
      const TString &sample, const TString &outDir,
      SignalModel sig, BackgroundModel bkg,
      double fitMin, double fitMax,
      bool useCommonShape,
      bool fixBkgShapeFromSidebands) {
    EffOutput invalid;
    if(!hPass || !hFail) return invalid;

    TH1D *hAll = AddHists(hPass, hFail,
                          TString("hVarAll_") + Sanitise(label + sample) +
                          "_" + TString::Itoa(hPass->GetNbinsX(), 10));
    if(!hAll) return invalid;

    const BkgOutput allBkg =
      FitBackgroundSidebands(hAll, label, year, sample, "AllVar",
                             outDir, bkg, false);
    const BkgOutput passBkg =
      FitBackgroundSidebands(hPass, label, year, sample, "PassVar",
                             outDir, bkg, false);
    const BkgOutput failBkg =
      FitBackgroundSidebands(hFail, label, year, sample, "FailVar",
                             outDir, bkg, false);

    const BkgOutput *allBkgForFit = allBkg.ok ? &allBkg : nullptr;
    const BkgOutput *passBkgForFit = passBkg.ok ? &passBkg : allBkgForFit;
    const BkgOutput *failBkgForFit = failBkg.ok ? &failBkg : allBkgForFit;

    const FitOutput all =
      FitOne(hAll, label, year, sample, "AllVar", outDir,
             sig, bkg, fitMin, fitMax, nullptr, false,
             allBkgForFit, fixBkgShapeFromSidebands, false);

    const FitOutput *shape =
      (useCommonShape && all.ok) ? &all : nullptr;

    const FitOutput pass =
      FitOne(hPass, label, year, sample, "PassVar", outDir,
             sig, bkg, fitMin, fitMax, shape, shape != nullptr,
             passBkgForFit, fixBkgShapeFromSidebands, false);
    const FitOutput fail =
      FitOne(hFail, label, year, sample, "FailVar", outDir,
             sig, bkg, fitMin, fitMax, shape, shape != nullptr,
             failBkgForFit, fixBkgShapeFromSidebands, false);

    EffOutput eff = MakeEfficiency(pass, fail);
    delete hAll;
    return eff;
  }

  double GraphMaxY(TGraphErrors *gr, double fallback = 0.0) {
    double out = fallback;
    if(!gr) return out;
    for(int i = 0; i < gr->GetN(); ++i) {
      double x = 0.;
      double y = 0.;
      gr->GetPoint(i, x, y);
      const double ey = gr->GetErrorY(i);
      if(std::isfinite(y)) out = std::max(out, y + (std::isfinite(ey) ? ey : 0.));
    }
    return out;
  }

  void StyleEffGraphs(TGraphErrors *grData, TGraphErrors *grQCD, TGraphErrors *grSF) {
    if(grData) {
      grData->SetMarkerStyle(20);
      grData->SetMarkerSize(0.85);
      grData->SetMarkerColor(kBlack);
      grData->SetLineColor(kBlack);
      grData->SetLineWidth(2);
    }
    if(grQCD) {
      grQCD->SetMarkerStyle(24);
      grQCD->SetMarkerSize(0.95);
      grQCD->SetMarkerColor(kRed + 1);
      grQCD->SetLineColor(kRed + 1);
      grQCD->SetLineWidth(2);
    }
    if(grSF) {
      grSF->SetMarkerStyle(20);
      grSF->SetMarkerSize(0.85);
      grSF->SetMarkerColor(kBlue + 1);
      grSF->SetLineColor(kBlue + 1);
      grSF->SetLineWidth(2);
    }
  }

  void AddEffPoint(TGraphErrors *gr, const double x, const double y,
                   const double ex, const double ey) {
    if(!gr) return;
    if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(ex) || !std::isfinite(ey)) return;
    const int n = gr->GetN();
    gr->SetPoint(n, x, y);
    gr->SetPointError(n, ex, ey);
  }

  void DrawEffSFPanel(TGraphErrors *grData,
                      TGraphErrors *grQCD,
                      TGraphErrors *grSF,
                      const TString &year,
                      const TString &outDir,
                      const TString &name,
                      const TString &referenceLabel,
                      const TString &selectionText,
                      const TString &xTitle,
                      const double xMin,
                      const double xMax,
                      const bool logX) {
    const bool hasEff = (grData && grData->GetN() > 0) || (grQCD && grQCD->GetN() > 0);
    const bool hasSF  = (grSF && grSF->GetN() > 0);
    if(!hasEff && !hasSF) return;

    StyleEffGraphs(grData, grQCD, grSF);

    TCanvas *c = new TCanvas((TString("c_effsf_") + name).Data(), "", 900, 900);
    c->cd();

    TPad *upper = new TPad((TString("upper_effsf_") + name).Data(), "", 0.0, 0.32, 1.0, 1.0);
    TPad *lower = new TPad((TString("lower_effsf_") + name).Data(), "", 0.0, 0.00, 1.0, 0.34);

    upper->SetLeftMargin(0.12);
    upper->SetRightMargin(0.04);
    upper->SetTopMargin(0.09);
    upper->SetBottomMargin(0.03);
    lower->SetLeftMargin(0.12);
    lower->SetRightMargin(0.04);
    lower->SetTopMargin(0.04);
    lower->SetBottomMargin(0.30);

    if(logX) {
      upper->SetLogx();
      lower->SetLogx();
    }

    upper->Draw();
    lower->Draw();

    upper->cd();

    TH1D *effFrame = new TH1D((TString("effFrame_") + name).Data(), "", 1, xMin, xMax);
    effFrame->SetStats(0);
    effFrame->GetXaxis()->SetLabelSize(0.0);
    effFrame->GetXaxis()->SetTitleSize(0.0);
    effFrame->GetYaxis()->SetTitle("Medium ID efficiency");
    effFrame->GetYaxis()->SetTitleSize(0.060);
    effFrame->GetYaxis()->SetLabelSize(0.052);
    effFrame->GetYaxis()->SetTitleOffset(0.82);
    // Efficiency validation plots use a fixed physical-looking range.  Large
    // symmetric validation uncertainties are intentionally clipped by the pad
    // at 1.1 rather than expanding the frame to several units.
    effFrame->GetYaxis()->SetRangeUser(0.0, 1.10);
    effFrame->Draw("AXIS");

    if(grData && grData->GetN() > 0) grData->Draw("PE SAME");
    if(grQCD && grQCD->GetN() > 0) grQCD->Draw("PE SAME");

    TLegend *leg = new TLegend(0.62, 0.72, 0.90, 0.87);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.045);
    if(grData && grData->GetN() > 0) leg->AddEntry(grData, "data", "pe");
    if(grQCD && grQCD->GetN() > 0) leg->AddEntry(grQCD, referenceLabel, "pe");
    leg->Draw();

    TLatex lat;
    lat.SetNDC();
    lat.SetTextFont(42);
    lat.SetTextSize(0.045);
    lat.DrawLatex(0.13, 0.94, "#bf{CMS} #it{Preliminary}");
    lat.SetTextAlign(31);
    lat.DrawLatex(0.96, 0.94, LumiText(year));
    lat.SetTextAlign(13);
    lat.SetTextSize(0.038);
    lat.DrawLatex(0.15, 0.84, selectionText);

    lower->cd();
    TH1D *sfFrame = new TH1D((TString("sfFrame_") + name).Data(), "", 1, xMin, xMax);
    sfFrame->SetStats(0);
    sfFrame->GetXaxis()->SetTitle(xTitle);
    sfFrame->GetXaxis()->SetTitleSize(0.105);
    sfFrame->GetXaxis()->SetLabelSize(0.090);
    sfFrame->GetXaxis()->SetTitleOffset(1.05);
    if(logX) {
      sfFrame->GetXaxis()->SetMoreLogLabels();
      sfFrame->GetXaxis()->SetNoExponent();
    }
    sfFrame->GetYaxis()->SetTitle("SF");
    sfFrame->GetYaxis()->SetTitleSize(0.100);
    sfFrame->GetYaxis()->SetLabelSize(0.080);
    sfFrame->GetYaxis()->SetTitleOffset(0.45);
    sfFrame->GetYaxis()->SetNdivisions(505);
    sfFrame->GetYaxis()->SetRangeUser(0.8, 1.2);
    sfFrame->Draw("AXIS");

    TLine *one = new TLine(xMin, 1.0, xMax, 1.0);
    one->SetLineStyle(2);
    one->SetLineColor(kGray + 2);
    one->SetLineWidth(2);
    one->Draw("SAME");
    if(grSF && grSF->GetN() > 0) grSF->Draw("PE SAME");
    sfFrame->Draw("AXIS SAME");

    c->SaveAs(outDir + "/" + name + ".png");
    c->SaveAs(outDir + "/" + name + ".pdf");

    delete c;
    delete effFrame;
    delete sfFrame;
    delete one;
  }

  vector<std::pair<double, double> > UniquePtIntervals(const vector<SummaryRow> &rows) {
    vector<std::pair<double, double> > out;
    for(const auto &r : rows) {
      if(r.bin.inclusive) continue;
      bool found = false;
      for(const auto &p : out) {
        if(NearlyEqual(p.first, r.bin.ptLow) && NearlyEqual(p.second, r.bin.ptHigh)) {
          found = true;
          break;
        }
      }
      if(!found) out.push_back(std::make_pair(r.bin.ptLow, r.bin.ptHigh));
    }
    std::sort(out.begin(), out.end(), [](const std::pair<double, double> &a,
                                         const std::pair<double, double> &b) {
      if(!NearlyEqual(a.first, b.first)) return a.first < b.first;
      return a.second < b.second;
    });
    return out;
  }

  void DrawSummaryGraphs(const vector<SummaryRow> &rows, const TString &year, const TString &outDir, const TString &referenceLabel) {
    if(gBinningMode == kBinningPtOnly) {
      const vector<double> ptEdges = PtOnlyEdges();
      TGraphErrors *grData = new TGraphErrors();
      TGraphErrors *grQCD  = new TGraphErrors();
      TGraphErrors *grSF   = new TGraphErrors();

      for(const auto &r : rows) {
        if(r.bin.inclusive) continue;
        if(!IsFullEtaRange(r.bin.etaLow, r.bin.etaHigh)) continue;
        const double x  = 0.5 * (r.bin.ptLow + r.bin.ptHigh);
        const double ex = 0.5 * (r.bin.ptHigh - r.bin.ptLow);
        if(r.dataEff.ok) AddEffPoint(grData, x, r.dataEff.eff, ex, r.dataEff.err);
        if(r.qcdEff.ok)  AddEffPoint(grQCD,  x, r.qcdEff.eff,  ex, r.qcdEff.err);
        if(r.sfOk)       AddEffPoint(grSF,   x, r.sf,          ex, r.sfErr);
      }

      DrawEffSFPanel(grData, grQCD, grSF, year, outDir,
                     TString("effsf_vs_pt_absEta_0p0_to_2p4"),
                     referenceLabel,
                     "0.0 < |#eta| < 2.4",
                     "probe p_{T} [GeV]",
                     ptEdges.front(), ptEdges.back(), true);

      delete grData;
      delete grQCD;
      delete grSF;
      return;
    }

    const vector<double> etaEdges = DefaultEtaEdges();

    // One canvas per |eta| bin: upper = efficiency(data, reference) vs probe pT,
    // lower = SF vs probe pT.  The pT binning follows the eta-dependent binning
    // used in MakeBins().
    for(unsigned int ieta = 0; ieta + 1 < etaEdges.size(); ++ieta) {
      const vector<double> ptEdges = PtEdgesForEta(etaEdges[ieta], etaEdges[ieta+1]);
      TGraphErrors *grData = new TGraphErrors();
      TGraphErrors *grQCD  = new TGraphErrors();
      TGraphErrors *grSF   = new TGraphErrors();

      for(const auto &r : rows) {
        if(r.bin.inclusive) continue;
        if(!NearlyEqual(r.bin.etaLow, etaEdges[ieta])) continue;
        if(!NearlyEqual(r.bin.etaHigh, etaEdges[ieta+1])) continue;
        const double x  = 0.5 * (r.bin.ptLow + r.bin.ptHigh);
        const double ex = 0.5 * (r.bin.ptHigh - r.bin.ptLow);
        if(r.dataEff.ok) AddEffPoint(grData, x, r.dataEff.eff, ex, r.dataEff.err);
        if(r.qcdEff.ok)  AddEffPoint(grQCD,  x, r.qcdEff.eff,  ex, r.qcdEff.err);
        if(r.sfOk)       AddEffPoint(grSF,   x, r.sf,          ex, r.sfErr);
      }

      const TString etaTag = EdgeLabel(etaEdges[ieta]) + "_to_" + EdgeLabel(etaEdges[ieta+1]);
      const TString etaText = Form("%.1f < |#eta| < %.1f", etaEdges[ieta], etaEdges[ieta+1]);
      DrawEffSFPanel(grData, grQCD, grSF, year, outDir,
                     TString("effsf_vs_pt_absEta_") + etaTag,
                     referenceLabel,
                     etaText,
                     "probe p_{T} [GeV]",
                     ptEdges.front(), ptEdges.back(), true);

      delete grData;
      delete grQCD;
      delete grSF;
    }

    // One canvas per pT interval that exists in the eta-dependent binning:
    // upper = efficiency(data, reference) vs probe |eta|, lower = SF vs probe |eta|.
    const vector<std::pair<double, double> > ptIntervals = UniquePtIntervals(rows);
    for(const auto &pt : ptIntervals) {
      TGraphErrors *grData = new TGraphErrors();
      TGraphErrors *grQCD  = new TGraphErrors();
      TGraphErrors *grSF   = new TGraphErrors();

      for(const auto &r : rows) {
        if(r.bin.inclusive) continue;
        if(!NearlyEqual(r.bin.ptLow, pt.first)) continue;
        if(!NearlyEqual(r.bin.ptHigh, pt.second)) continue;
        const double x  = 0.5 * (r.bin.etaLow + r.bin.etaHigh);
        const double ex = 0.5 * (r.bin.etaHigh - r.bin.etaLow);
        if(r.dataEff.ok) AddEffPoint(grData, x, r.dataEff.eff, ex, r.dataEff.err);
        if(r.qcdEff.ok)  AddEffPoint(grQCD,  x, r.qcdEff.eff,  ex, r.qcdEff.err);
        if(r.sfOk)       AddEffPoint(grSF,   x, r.sf,          ex, r.sfErr);
      }

      const TString ptTag = EdgeLabel(pt.first) + "_to_" + EdgeLabel(pt.second);
      const TString ptText = Form("%.0f < p_{T}^{probe} < %.0f GeV", pt.first, pt.second);
      DrawEffSFPanel(grData, grQCD, grSF, year, outDir,
                     TString("effsf_vs_absEta_Pt") + ptTag,
                     referenceLabel,
                     ptText,
                     "probe |#eta|",
                     etaEdges.front(), etaEdges.back(), false);

      delete grData;
      delete grQCD;
      delete grSF;
    }
  }
}

void id_eff(TString Year = "2018",
                      TString Trigger = "HighPtMuon",
                      TString BaseDir = "/data6/Users/joonblee/SKFlatOutput/Run2UltraLegacy_v3",
                      TString Analyzer = "NIsoMuon",
                      TString BaseRegion = "OS_POGMedium_tight_BJet_MuonIDEfficiency",
                      int RebinFactor = 3,
                      TString SignalModelInput = "CB",
                      TString BackgroundModelInput = "Bern5",
                      double FitMin = 2.00,
                      double FitMax = 5.00,
                      bool UseCommonShape = false,
                      bool FixBkgShapeFromSidebands = true,
                      double Side2FitWeight = 4.0,
                      bool UseLogBkgFit = true,
                      double MinBkgRelErr = 0.015,
                      TString ReferenceInput = "QCD",
                      TString OutputDirInput = "plots_IDEff",
                      bool SavePerBinPlots = true,
                      bool SaveSummaryPlots = true,
                      TString BinFilter = "all",
                      int MaxBins = -1,
                      TString HistName = "auto",
                      bool IncludeInclusive = false,
                      bool InspectOnly = false,
                      TString ResonanceInput = "Jpsi",
                      TString BinningInput = "eta-pt",
                      double YieldIntLowInput = -1.0,
                      double YieldIntHighInput = -1.0,
                      double BkgFitMinInput = -1.0,
                      double BkgFitMaxInput = -1.0,
                      TString YieldModeInput = "fitnorm") {
  using namespace JpsiMuonIDFit;

  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  const SignalModel sigModel = ParseSignalModel(SignalModelInput);
  const BackgroundModel bkgModel = ParseBackgroundModel(BackgroundModelInput);
  gBinningMode = ParseBinningMode(BinningInput);
  ConfigureResonance(ResonanceInput, FitMin, FitMax, HistName,
                     YieldIntLowInput, YieldIntHighInput, BkgFitMinInput, BkgFitMaxInput);
  gYieldMode = YieldModeInput;
  gYieldMode.ToLower();
  if(!(gYieldMode == "integral" || gYieldMode == "fitnorm" || gYieldMode == "norm" ||
       gYieldMode == "normalisation" || gYieldMode == "normalization")) {
    cout << "[WARNING] Unknown YieldModeInput='" << YieldModeInput << "'. Use 'fitnorm'." << endl;
    gYieldMode = "fitnorm";
  }
  if(!std::isfinite(FitMin) || !std::isfinite(FitMax) || FitMax <= FitMin) {
    cout << "[ERROR] FitMax must be larger than FitMin. Got FitMin=" << FitMin << ", FitMax=" << FitMax << endl;
    return;
  }

  gSignalModel = sigModel;
  gBackgroundModel = bkgModel;
  gFitMin = FitMin;
  gFitMax = FitMax;
  gBkgShapeMin = gBkgFitMin;
  gBkgShapeMax = gBkgFitMax;
  gSide2FitWeight = std::max(1.0, Side2FitWeight);
  gUseLogBkgFit = UseLogBkgFit;
  gMinBkgRelErr = std::max(0.0, MinBkgRelErr);
  gIncludeInclusive = IncludeInclusive;

  const TString inputDir = (Trigger == "")
                         ? BaseDir + "/" + Analyzer + "/" + Year + "/"
                         : BaseDir + "/" + Analyzer + "/" + Year + "/" + Trigger + "/";
  const TString dataFile = ResolveFile(inputDir, {"data.root"}, "Data");
  const TString refLabel = ReferenceLabel(ReferenceInput);
  const TString refFile  = ResolveReferenceFile(inputDir, ReferenceInput);

  TString outDir = OutputDirInput;
  if(outDir == "") outDir = "plots_IDEff";
  gSystem->mkdir(outDir, kTRUE);

  cout << "\n[INFO] Input directory : " << inputDir << endl;
  cout << "[INFO] Base region     : " << BaseRegion << endl;
  cout << "[INFO] Mass binning    : nominal rebin=3 (30 MeV); stability variations rebin=1,2,4,5 (10,20,40,50 MeV)" << endl;
  cout << "[INFO] Signal model    : " << SignalModelName(sigModel) << endl;
  cout << "[INFO] Background model: " << BackgroundModelName(bkgModel) << endl;
  cout << "[INFO] Available bkg models: " << AvailableBackgroundModelsText() << endl;
  if(IsMonoExpBackground(bkgModel) || IsMonoChebBackground(bkgModel) || IsMonoBernBackground(bkgModel)) {
    cout << "[INFO] Monotonic bkg  : enabled; continuum background is constrained to be non-increasing over [2,5] GeV" << endl;
  }
  cout << "[INFO] Resonance       : " << gResonanceLabel << endl;
  cout << "[INFO] Final " << gResonanceLabel << " fit : [" << FitMin << ", " << FitMax << "] GeV" << endl;
  cout << "[INFO] Final veto     : " << (gUseFinalVeto ? Form("[%.1f,%.1f] GeV if it overlaps", gFinalVetoLow, gFinalVetoHigh) : TString("none")) << endl;
  cout << "[INFO] Bkg prefit     : [" << gBkgFitMin << "," << gBkgFitMax << "] GeV; sidebands " << SidebandRangesText() << " GeV" << endl;
  cout << "[INFO] Bkg fit metric  : " << (gUseLogBkgFit ? "log-density chi2" : "density chi2")
       << "; each rebin variation repeats the full sideband prefit" << endl;
  cout << "[INFO] Common shape    : " << (UseCommonShape ? "true" : "false") << endl;
  cout << "[INFO] Fix bkg shape   : " << (FixBkgShapeFromSidebands ? "true" : "false") << endl;
  cout << "[INFO] Bkg norm range  : sideband prefit +/- " << 100.0 * kBkgNormRelConstraint << "% in final S+B fit" << endl;
  cout << "[INFO] Side2 weight    : " << Side2FitWeight << endl;
  cout << "[INFO] Min bkg rel err : " << MinBkgRelErr << endl;
  cout << "[INFO] Reference input : " << ReferenceInput << " (label: " << refLabel << ")" << endl;
  if(UseFitNormYield()) {
    cout << "[INFO] Yield definition: N_" << gResonanceLabel << " = fitted signal normalisation parameter" << endl;
    cout << "[INFO] Efficiency fit : AN-compatible separate Pass/Fail chi2 fits; eff=P/(P+F)" << endl;
    cout << "[INFO] Uncertainties  : fitted P/F errors propagated to eff; rebin=1,2,4,5 envelope around nominal rebin=3 added in quadrature" << endl;
  }
  else {
    cout << "[INFO] Yield definition: N_" << gResonanceLabel << " = Integral(signal function, " << gYieldIntLow << ", " << gYieldIntHigh << ") GeV" << endl;
    cout << "[INFO] Efficiency fit : separate Pass/Fail chi2 fits; eff=P/(P+F)" << endl;
    cout << "[INFO] Uncertainties  : fitted P/F errors propagated to eff; rebin=1,2,4,5 envelope around nominal rebin=3 added in quadrature" << endl;
  }
  if(UseCommonShape) {
    cout << "[INFO] Signal shape note: --common-shape fixes the separate Pass/Fail signal shapes to the All-fit shape; default AN-compatible mode fits them independently." << endl;
  }
  cout << "[INFO] Save bin plots  : " << (SavePerBinPlots ? "true" : "false") << endl;
  cout << "[INFO] Save summaries  : " << (SaveSummaryPlots ? "true" : "false") << endl;
  cout << "[INFO] Bin filter      : " << BinFilter << endl;
  cout << "[INFO] Max bins        : " << MaxBins << endl;
  cout << "[INFO] Hist name       : " << HistName << endl;
  cout << "[INFO] Binning mode    : " << BinningModeName(gBinningMode) << endl;
  cout << "[INFO] Output pT bins  : eta-dependent; 0-0.9: 10,20,30,50,120; 0.9-1.2: 10,30,50,120; 1.2-2.1: 10,30,120; 2.1-2.4: 10,120 GeV" << endl;
  cout << "[INFO] Input merging   : missing analyzer sub-bin histograms contribute zero when forming a merged pT bin; only a completely missing Pass/Fail input leaves the efficiency bin empty" << endl;
  cout << "[INFO] Include incl.   : " << (gIncludeInclusive ? "true" : "false") << endl;
  cout << "[INFO] Inspect only    : " << (InspectOnly ? "true" : "false") << endl;
  cout << "[INFO] Output dir      : " << outDir << "\n" << endl;

  if(InspectOnly) {
    InspectInputHistograms(dataFile, "Data", BaseRegion, HistName);
    InspectInputHistograms(refFile, refLabel, BaseRegion, HistName);
    return;
  }

  std::ofstream csv((outDir + "/summary_" + Year + ".csv").Data());
  csv << "bin,ptLow,ptHigh,absEtaLow,absEtaHigh,"
      << "dataAll,dataAllErr,dataPass,dataPassErr,dataFail,dataFailErr,dataEff,dataEffErr,dataEffStatErr,dataEffBinningErr,dataEffRebin1,dataEffRebin2,dataEffRebin4,dataEffRebin5,"
      << "qcdAll,qcdAllErr,qcdPass,qcdPassErr,qcdFail,qcdFailErr,qcdEff,qcdEffErr,qcdEffStatErr,qcdEffBinningErr,qcdEffRebin1,qcdEffRebin2,qcdEffRebin4,qcdEffRebin5,"
      << "SF,SFErr,SFStatErr,SFBinningErr,SFRebin1,SFRebin2,SFRebin4,SFRebin5,"
      << "dataAllStatus,dataPassStatus,dataFailStatus,qcdAllStatus,qcdPassStatus,qcdFailStatus,"
      << "dataAllBkgStatus,dataPassBkgStatus,dataFailBkgStatus,qcdAllBkgStatus,qcdPassBkgStatus,qcdFailBkgStatus\n";

  vector<SummaryRow> rows;
  const vector<BinDef> bins = MakeBins();
  const int nSelectedBins = CountSelectedBins(bins, BinFilter, MaxBins);
  int nProcessedBins = 0;
  int nRawHistsLoaded = 0;

  cout << "[INFO] Selected bins   : " << nSelectedBins << " / " << bins.size() << endl;

  for(const auto &bin : bins) {
    if(!MatchBinFilter(bin, BinFilter)) continue;
    if(MaxBins >= 0 && nProcessedBins >= MaxBins) break;
    ++nProcessedBins;
    cout << "[BIN " << nProcessedBins << "/" << nSelectedBins << "] " << bin.tag << endl;
    SummaryRow row;
    row.bin = bin;

    TH1D *hDataPassRaw = LoadForBin(dataFile, BaseRegion, bin, "Pass", bins, TString("hDataPassRaw_") + bin.tag, HistName);
    TH1D *hDataFailRaw = LoadForBin(dataFile, BaseRegion, bin, "Fail", bins, TString("hDataFailRaw_") + bin.tag, HistName);
    TH1D *hQCDPassRaw  = LoadForBin(refFile,  BaseRegion, bin, "Pass", bins, TString("hQCDPassRaw_")  + bin.tag, HistName);
    TH1D *hQCDFailRaw  = LoadForBin(refFile,  BaseRegion, bin, "Fail", bins, TString("hQCDFailRaw_")  + bin.tag, HistName);
    nRawHistsLoaded += (hDataPassRaw ? 1 : 0) + (hDataFailRaw ? 1 : 0) + (hQCDPassRaw ? 1 : 0) + (hQCDFailRaw ? 1 : 0);

    const bool dataInputsComplete = (hDataPassRaw != nullptr && hDataFailRaw != nullptr);
    const bool refInputsComplete  = (hQCDPassRaw  != nullptr && hQCDFailRaw  != nullptr);

    if(!dataInputsComplete) {
      cout << "[WARNING] Missing Data Pass/Fail histogram for " << bin.tag
           << "; leave Data efficiency empty for this bin." << endl;
    }
    if(!refInputsComplete) {
      cout << "[WARNING] Missing " << refLabel << " Pass/Fail histogram for " << bin.tag
           << "; leave reference efficiency/SF empty for this bin." << endl;
    }

    TH1D *hDataAllRaw = dataInputsComplete
                      ? AddHists(hDataPassRaw, hDataFailRaw, TString("hDataAllRaw_") + bin.tag)
                      : nullptr;
    TH1D *hQCDAllRaw  = refInputsComplete
                      ? AddHists(hQCDPassRaw, hQCDFailRaw, TString("hQCDAllRaw_") + bin.tag)
                      : nullptr;

    const int nominalRebin = 3;
    if(RebinFactor != nominalRebin && nProcessedBins == 1) {
      cout << "[WARNING] --rebin=" << RebinFactor
           << " is ignored for the standard result; nominal rebin=3 is fixed by the uncertainty prescription." << endl;
    }

    TH1D *hDataAll  = RebinAndMakeDensity(hDataAllRaw,  nominalRebin, TString("hDataAll_")  + bin.tag);
    TH1D *hDataPass = RebinAndMakeDensity(hDataPassRaw, nominalRebin, TString("hDataPass_") + bin.tag);
    TH1D *hDataFail = RebinAndMakeDensity(hDataFailRaw, nominalRebin, TString("hDataFail_") + bin.tag);
    TH1D *hQCDAll   = RebinAndMakeDensity(hQCDAllRaw,   nominalRebin, TString("hQCDAll_")   + bin.tag);
    TH1D *hQCDPass  = RebinAndMakeDensity(hQCDPassRaw,  nominalRebin, TString("hQCDPass_")  + bin.tag);
    TH1D *hQCDFail  = RebinAndMakeDensity(hQCDFailRaw,  nominalRebin, TString("hQCDFail_")  + bin.tag);

    TH1D *hDataPassR1 = RebinAndMakeDensity(hDataPassRaw, 1, TString("hDataPassR1_") + bin.tag, false);
    TH1D *hDataFailR1 = RebinAndMakeDensity(hDataFailRaw, 1, TString("hDataFailR1_") + bin.tag, false);
    TH1D *hDataPassR2 = RebinAndMakeDensity(hDataPassRaw, 2, TString("hDataPassR2_") + bin.tag, false);
    TH1D *hDataFailR2 = RebinAndMakeDensity(hDataFailRaw, 2, TString("hDataFailR2_") + bin.tag, false);
    TH1D *hDataPassR4 = RebinAndMakeDensity(hDataPassRaw, 4, TString("hDataPassR4_") + bin.tag, false);
    TH1D *hDataFailR4 = RebinAndMakeDensity(hDataFailRaw, 4, TString("hDataFailR4_") + bin.tag, false);
    TH1D *hDataPassR5 = RebinAndMakeDensity(hDataPassRaw, 5, TString("hDataPassR5_") + bin.tag, false);
    TH1D *hDataFailR5 = RebinAndMakeDensity(hDataFailRaw, 5, TString("hDataFailR5_") + bin.tag, false);
    TH1D *hQCDPassR1  = RebinAndMakeDensity(hQCDPassRaw,  1, TString("hQCDPassR1_")  + bin.tag, false);
    TH1D *hQCDFailR1  = RebinAndMakeDensity(hQCDFailRaw,  1, TString("hQCDFailR1_")  + bin.tag, false);
    TH1D *hQCDPassR2  = RebinAndMakeDensity(hQCDPassRaw,  2, TString("hQCDPassR2_")  + bin.tag, false);
    TH1D *hQCDFailR2  = RebinAndMakeDensity(hQCDFailRaw,  2, TString("hQCDFailR2_")  + bin.tag, false);
    TH1D *hQCDPassR4  = RebinAndMakeDensity(hQCDPassRaw,  4, TString("hQCDPassR4_")  + bin.tag, false);
    TH1D *hQCDFailR4  = RebinAndMakeDensity(hQCDFailRaw,  4, TString("hQCDFailR4_")  + bin.tag, false);
    TH1D *hQCDPassR5  = RebinAndMakeDensity(hQCDPassRaw,  5, TString("hQCDPassR5_")  + bin.tag, false);
    TH1D *hQCDFailR5  = RebinAndMakeDensity(hQCDFailRaw,  5, TString("hQCDFailR5_")  + bin.tag, false);

    // AN-era nominal method: background prefit and final Pass/Fail fits use
    // the same mass binning.  The full procedure is repeated for each rebin
    // variation so the envelope measures the actual fit-binning dependence.
    if(dataInputsComplete) {
      row.dataAllBkg  = FitBackgroundSidebands(hDataAll,  bin.tag, Year, "Data", "All",  outDir, bkgModel, false);
      row.dataPassBkg = FitBackgroundSidebands(hDataPass, bin.tag, Year, "Data", "Pass", outDir, bkgModel, SavePerBinPlots);
      row.dataFailBkg = FitBackgroundSidebands(hDataFail, bin.tag, Year, "Data", "Fail", outDir, bkgModel, SavePerBinPlots);

      const BkgOutput *allBkg = row.dataAllBkg.ok ? &row.dataAllBkg : nullptr;
      const BkgOutput *passBkg = row.dataPassBkg.ok ? &row.dataPassBkg : allBkg;
      const BkgOutput *failBkg = row.dataFailBkg.ok ? &row.dataFailBkg : allBkg;

      row.dataAll = FitOne(hDataAll, bin.tag, Year, "Data", "All", outDir,
                           sigModel, bkgModel, FitMin, FitMax, nullptr, false,
                           allBkg, FixBkgShapeFromSidebands, false);
      const FitOutput *shape =
        (UseCommonShape && row.dataAll.ok) ? &row.dataAll : nullptr;
      row.dataPass = FitOne(hDataPass, bin.tag, Year, "Data", "Pass", outDir,
                            sigModel, bkgModel, FitMin, FitMax, shape, shape != nullptr,
                            passBkg, FixBkgShapeFromSidebands, SavePerBinPlots);
      row.dataFail = FitOne(hDataFail, bin.tag, Year, "Data", "Fail", outDir,
                            sigModel, bkgModel, FitMin, FitMax, shape, shape != nullptr,
                            failBkg, FixBkgShapeFromSidebands, SavePerBinPlots);

      row.dataEff = MakeEfficiency(row.dataPass, row.dataFail);
      if(row.dataEff.ok) {
        const EffOutput r1 = FitSeparateEfficiencyVariation(
          hDataPassR1, hDataFailR1, bin.tag, Year, "Data", outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        const EffOutput r2 = FitSeparateEfficiencyVariation(
          hDataPassR2, hDataFailR2, bin.tag, Year, "Data", outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        const EffOutput r4 = FitSeparateEfficiencyVariation(
          hDataPassR4, hDataFailR4, bin.tag, Year, "Data", outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        const EffOutput r5 = FitSeparateEfficiencyVariation(
          hDataPassR5, hDataFailR5, bin.tag, Year, "Data", outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        ApplyBinningEnvelope(row.dataEff, r1, r2, r4, r5, "Data", bin.tag);
      }
    }

    if(refInputsComplete) {
      row.qcdAllBkg  = FitBackgroundSidebands(hQCDAll,  bin.tag, Year, refLabel, "All",  outDir, bkgModel, false);
      row.qcdPassBkg = FitBackgroundSidebands(hQCDPass, bin.tag, Year, refLabel, "Pass", outDir, bkgModel, SavePerBinPlots);
      row.qcdFailBkg = FitBackgroundSidebands(hQCDFail, bin.tag, Year, refLabel, "Fail", outDir, bkgModel, SavePerBinPlots);

      const BkgOutput *allBkg = row.qcdAllBkg.ok ? &row.qcdAllBkg : nullptr;
      const BkgOutput *passBkg = row.qcdPassBkg.ok ? &row.qcdPassBkg : allBkg;
      const BkgOutput *failBkg = row.qcdFailBkg.ok ? &row.qcdFailBkg : allBkg;

      row.qcdAll = FitOne(hQCDAll, bin.tag, Year, refLabel, "All", outDir,
                          sigModel, bkgModel, FitMin, FitMax, nullptr, false,
                          allBkg, FixBkgShapeFromSidebands, false);
      const FitOutput *shape =
        (UseCommonShape && row.qcdAll.ok) ? &row.qcdAll : nullptr;
      row.qcdPass = FitOne(hQCDPass, bin.tag, Year, refLabel, "Pass", outDir,
                           sigModel, bkgModel, FitMin, FitMax, shape, shape != nullptr,
                           passBkg, FixBkgShapeFromSidebands, SavePerBinPlots);
      row.qcdFail = FitOne(hQCDFail, bin.tag, Year, refLabel, "Fail", outDir,
                           sigModel, bkgModel, FitMin, FitMax, shape, shape != nullptr,
                           failBkg, FixBkgShapeFromSidebands, SavePerBinPlots);

      row.qcdEff = MakeEfficiency(row.qcdPass, row.qcdFail);
      if(row.qcdEff.ok) {
        const EffOutput r1 = FitSeparateEfficiencyVariation(
          hQCDPassR1, hQCDFailR1, bin.tag, Year, refLabel, outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        const EffOutput r2 = FitSeparateEfficiencyVariation(
          hQCDPassR2, hQCDFailR2, bin.tag, Year, refLabel, outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        const EffOutput r4 = FitSeparateEfficiencyVariation(
          hQCDPassR4, hQCDFailR4, bin.tag, Year, refLabel, outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        const EffOutput r5 = FitSeparateEfficiencyVariation(
          hQCDPassR5, hQCDFailR5, bin.tag, Year, refLabel, outDir,
          sigModel, bkgModel, FitMin, FitMax,
          UseCommonShape, FixBkgShapeFromSidebands);
        ApplyBinningEnvelope(row.qcdEff, r1, r2, r4, r5, refLabel, bin.tag);
      }
    }

    if(row.dataEff.ok && row.qcdEff.ok &&
       row.dataEff.eff > 0. && row.qcdEff.eff > 0.) {
      row.sf = row.dataEff.eff / row.qcdEff.eff;

      const double relDataStat = row.dataEff.statErr / row.dataEff.eff;
      const double relRefStat  = row.qcdEff.statErr  / row.qcdEff.eff;
      const double sfStatVarRel =
        relDataStat * relDataStat + relRefStat * relRefStat;
      row.sfStatErr =
        (std::isfinite(row.sf) && std::isfinite(sfStatVarRel) && sfStatVarRel >= 0.)
        ? row.sf * std::sqrt(sfStatVarRel) : 0.;

      row.sfBinningErr = 0.;
      int nValidSFVariations = 0;

      auto includeSFVariation = [&](bool dataOk, double dataEff,
                                    bool refOk, double refEff,
                                    double &storedSF) {
        if(!dataOk || !refOk || !(refEff > 0.)) return;
        storedSF = dataEff / refEff;
        if(!std::isfinite(storedSF)) return;
        ++nValidSFVariations;
        row.sfBinningErr = std::max(
          row.sfBinningErr, std::fabs(storedSF - row.sf));
      };

      includeSFVariation(row.dataEff.rebin1Ok, row.dataEff.rebin1Eff,
                         row.qcdEff.rebin1Ok, row.qcdEff.rebin1Eff,
                         row.sfRebin1);
      includeSFVariation(row.dataEff.rebin2Ok, row.dataEff.rebin2Eff,
                         row.qcdEff.rebin2Ok, row.qcdEff.rebin2Eff,
                         row.sfRebin2);
      includeSFVariation(row.dataEff.rebin4Ok, row.dataEff.rebin4Eff,
                         row.qcdEff.rebin4Ok, row.qcdEff.rebin4Eff,
                         row.sfRebin4);
      includeSFVariation(row.dataEff.rebin5Ok, row.dataEff.rebin5Eff,
                         row.qcdEff.rebin5Ok, row.qcdEff.rebin5Eff,
                         row.sfRebin5);

      row.sfBinningOk = (nValidSFVariations > 0);
      row.sfErr = std::sqrt(
        row.sfStatErr * row.sfStatErr +
        row.sfBinningErr * row.sfBinningErr);
      row.sfOk = std::isfinite(row.sf) && std::isfinite(row.sfErr);

      if(nValidSFVariations == 0) {
        cout << "[WARNING] No matched valid rebin variation for SF in "
             << bin.tag << "; keep SF with statistical uncertainty only." << endl;
      }
    }

    cout << "  Data: pass = " << row.dataPass.yield << " +/- " << row.dataPass.yieldErr
         << ", fail = " << row.dataFail.yield << " +/- " << row.dataFail.yieldErr
         << ", eff = " << row.dataEff.eff << " +/- " << row.dataEff.err;
    if(row.dataEff.ok) {
      cout << "  [stat=" << row.dataEff.statErr
           << ", binning=" << row.dataEff.binningErr
           << ", r1=" << (row.dataEff.rebin1Ok ? Form("%.6g", row.dataEff.rebin1Eff) : "FAIL")
           << ", r2=" << (row.dataEff.rebin2Ok ? Form("%.6g", row.dataEff.rebin2Eff) : "FAIL")
           << ", r4=" << (row.dataEff.rebin4Ok ? Form("%.6g", row.dataEff.rebin4Eff) : "FAIL")
           << ", r5=" << (row.dataEff.rebin5Ok ? Form("%.6g", row.dataEff.rebin5Eff) : "FAIL")
           << ", Nsig=" << row.dataEff.nCount << "]";
    }
    cout << endl;
    cout << "  " << refLabel << " : pass = " << row.qcdPass.yield << " +/- " << row.qcdPass.yieldErr
         << ", fail = " << row.qcdFail.yield << " +/- " << row.qcdFail.yieldErr
         << ", eff = " << row.qcdEff.eff << " +/- " << row.qcdEff.err;
    if(row.qcdEff.ok) {
      cout << "  [stat=" << row.qcdEff.statErr
           << ", binning=" << row.qcdEff.binningErr
           << ", r1=" << (row.qcdEff.rebin1Ok ? Form("%.6g", row.qcdEff.rebin1Eff) : "FAIL")
           << ", r2=" << (row.qcdEff.rebin2Ok ? Form("%.6g", row.qcdEff.rebin2Eff) : "FAIL")
           << ", r4=" << (row.qcdEff.rebin4Ok ? Form("%.6g", row.qcdEff.rebin4Eff) : "FAIL")
           << ", r5=" << (row.qcdEff.rebin5Ok ? Form("%.6g", row.qcdEff.rebin5Eff) : "FAIL")
           << ", Nsig=" << row.qcdEff.nCount << "]";
    }
    cout << endl;
    cout << "  SF  = " << row.sf << " +/- " << row.sfErr
         << "  [stat=" << row.sfStatErr
         << ", binning=" << row.sfBinningErr
         << ", r1=" << row.sfRebin1
         << ", r2=" << row.sfRebin2
         << ", r4=" << row.sfRebin4
         << ", r5=" << row.sfRebin5
         << "]  (sfOk=" << row.sfOk << ")\n" << endl;

    csv << bin.tag << "," << bin.ptLow << "," << bin.ptHigh << "," << bin.etaLow << "," << bin.etaHigh << ","
        << row.dataAll.yield << "," << row.dataAll.yieldErr << ","
        << row.dataPass.yield << "," << row.dataPass.yieldErr << ","
        << row.dataFail.yield << "," << row.dataFail.yieldErr << ","
        << row.dataEff.eff << "," << row.dataEff.err << ","
        << row.dataEff.statErr << "," << row.dataEff.binningErr << ","
        << row.dataEff.rebin1Eff << "," << row.dataEff.rebin2Eff << ","
        << row.dataEff.rebin4Eff << "," << row.dataEff.rebin5Eff << ","
        << row.qcdAll.yield << "," << row.qcdAll.yieldErr << ","
        << row.qcdPass.yield << "," << row.qcdPass.yieldErr << ","
        << row.qcdFail.yield << "," << row.qcdFail.yieldErr << ","
        << row.qcdEff.eff << "," << row.qcdEff.err << ","
        << row.qcdEff.statErr << "," << row.qcdEff.binningErr << ","
        << row.qcdEff.rebin1Eff << "," << row.qcdEff.rebin2Eff << ","
        << row.qcdEff.rebin4Eff << "," << row.qcdEff.rebin5Eff << ","
        << row.sf << "," << row.sfErr << ","
        << row.sfStatErr << "," << row.sfBinningErr << ","
        << row.sfRebin1 << "," << row.sfRebin2 << ","
        << row.sfRebin4 << "," << row.sfRebin5 << ","
        << row.dataAll.fitStatus << "," << row.dataPass.fitStatus << "," << row.dataFail.fitStatus << ","
        << row.qcdAll.fitStatus << "," << row.qcdPass.fitStatus << "," << row.qcdFail.fitStatus << ","
        << row.dataAllBkg.fitStatus << "," << row.dataPassBkg.fitStatus << "," << row.dataFailBkg.fitStatus << ","
        << row.qcdAllBkg.fitStatus << "," << row.qcdPassBkg.fitStatus << "," << row.qcdFailBkg.fitStatus << "\n";

    rows.push_back(row);

    delete hDataPassRaw;
    delete hDataFailRaw;
    delete hQCDPassRaw;
    delete hQCDFailRaw;
    delete hDataAllRaw;
    delete hQCDAllRaw;
    delete hDataAll;
    delete hDataPass;
    delete hDataFail;
    delete hQCDAll;
    delete hQCDPass;
    delete hQCDFail;
    delete hDataPassR1;
    delete hDataFailR1;
    delete hDataPassR2;
    delete hDataFailR2;
    delete hDataPassR4;
    delete hDataFailR4;
    delete hDataPassR5;
    delete hDataFailR5;
    delete hQCDPassR1;
    delete hQCDFailR1;
    delete hQCDPassR2;
    delete hQCDFailR2;
    delete hQCDPassR4;
    delete hQCDFailR4;
    delete hQCDPassR5;
    delete hQCDFailR5;
  }

  csv.close();
  if(nRawHistsLoaded == 0) {
    cout << "[ERROR] No input histograms were loaded. Fit plots and summary plots were not produced." << endl;
    cout << "[ERROR] Tried BaseRegion='" << BaseRegion << "', HistName='" << HistName << "'." << endl;
    cout << "[ERROR] Run with --inspect-hists to print matching keys from the ROOT files." << endl;
    cout << "[DONE] Summary CSV : " << outDir + "/summary_" + Year + ".csv" << " (empty/invalid because inputs were missing)" << endl;
    return;
  }
  if(SaveSummaryPlots) DrawSummaryGraphs(rows, Year, outDir, refLabel);

  cout << "[DONE] Summary CSV : " << outDir + "/summary_" + Year + ".csv" << endl;
  if(SavePerBinPlots) {
    cout << "[DONE] Bkg fits    : " << outDir << "/bkgfit_<sample>_<Pass|Fail>_<bin>.png" << endl;
    cout << "[DONE] Fit plots   : " << outDir << "/fit_<sample>_<Pass|Fail>_<bin>.png" << endl;
  }
  if(SaveSummaryPlots) {
    cout << "[DONE] Eff/SF plots: " << outDir << "/effsf_vs_*.png" << endl;
  }
}

// Convenience wrapper kept for old commands that still call id_eff_v8(...).
void id_eff_v8(TString Year = "2018",
               TString Trigger = "HighPtMuon",
               TString BaseDir = "/data6/Users/joonblee/SKFlatOutput/Run2UltraLegacy_v3",
               TString Analyzer = "NIsoMuon",
               TString BaseRegion = "OS_POGMedium_tight_BJet_MuonIDEfficiency",
               int RebinFactor = 3,
               TString SignalModelInput = "CB",
               TString BackgroundModelInput = "Bern5",
               double FitMin = 2.00,
               double FitMax = 5.00,
               bool UseCommonShape = false,
               bool FixBkgShapeFromSidebands = true,
               double Side2FitWeight = 4.0,
               bool UseLogBkgFit = true,
               double MinBkgRelErr = 0.015,
               TString ReferenceInput = "QCD",
               TString OutputDirInput = "plots_IDEff",
               bool SavePerBinPlots = true,
               bool SaveSummaryPlots = true,
               TString BinFilter = "all",
               int MaxBins = -1,
               TString HistName = "auto",
               bool IncludeInclusive = false,
               bool InspectOnly = false,
               TString ResonanceInput = "Jpsi",
               TString BinningInput = "eta-pt",
               double YieldIntLowInput = -1.0,
               double YieldIntHighInput = -1.0,
               double BkgFitMinInput = -1.0,
               double BkgFitMaxInput = -1.0,
               TString YieldModeInput = "fitnorm") {
  id_eff(Year, Trigger, BaseDir, Analyzer, BaseRegion, RebinFactor,
         SignalModelInput, BackgroundModelInput, FitMin, FitMax,
         UseCommonShape, FixBkgShapeFromSidebands, Side2FitWeight,
         UseLogBkgFit, MinBkgRelErr, ReferenceInput, OutputDirInput,
         SavePerBinPlots, SaveSummaryPlots, BinFilter, MaxBins,
         HistName, IncludeInclusive, InspectOnly,
         ResonanceInput, BinningInput, YieldIntLowInput, YieldIntHighInput,
         BkgFitMinInput, BkgFitMaxInput, YieldModeInput);
}
"""



def script_path() -> Path:
    try:
        return Path(__file__).resolve()
    except NameError:
        return Path.cwd() / "id_eff.py"


def shell_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, got {value!r}."
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, got {parsed}."
        )
    return parsed


def resonance_defaults(
    resonance: str,
) -> Tuple[float, float, str, float, float, float, float]:
    key = resonance.strip().lower().replace("_", "").replace("-", "")
    if key in {"z", "zpeak", "zmumu", "zboson"}:
        # This follows the uploaded C++ implementation exactly.
        return (70.0, 110.0, "Dilepton_Mass", 80.0, 100.0, 60.0, 120.0)
    return (2.00, 5.00, "DileptonJPsi_Mass", 3.00, 3.20, 2.00, 5.00)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="id_eff.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Measure NIsoMuon muon-ID efficiencies and data/reference scale factors.\n\n"
            "The full up-to-date C++ implementation is embedded in this Python file.\n"
            "No external id_eff.C is read.  Input and output bases are fixed."
        ),
        epilog=(
            "Fixed inputs:\n"
            "  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon/\n"
            "    MuonIDEfficiency/<era>/data.root\n"
            "    MuonIDEfficiency/<era>/NIsoMuon_QCD_Inclusive.root\n"
            "    MuonIDEfficiency/<era>/NIsoMuon_Top.root\n\n"
            "Fixed outputs:\n"
            "  /data6/Users/joonblee/PlotMaker/plots/MuonIDEfficiency/<era>/\n\n"
            "Main examples:\n"
            "  python3 id_eff.py --year 2023 --inspect-hists\n"
            "  python3 id_eff.py --year 2023 --quick-check\n"
            "  python3 id_eff.py --year 2023\n"
            "  python3 id_eff.py --year 2023 --reference Top\n"
            "  python3 id_eff.py --year 2023 --binning pt-only\n"
            "  python3 id_eff.py --year 2023 --yield-mode fitnorm\n"
            "  python3 id_eff.py --year all\n\n"
            "J/psi defaults from the uploaded C++:\n"
            "  HistName=DileptonJPsi_Mass; final fit [2.00,5.00] GeV with the psi(2S) region vetoed;\n"
            "  core-yield integral [3.00,3.20] GeV; bkg prefit [2,5] GeV.\n\n"
            "Z defaults from the uploaded C++:\n"
            "  HistName=Dilepton_Mass; final fit [70,110] GeV;\n"
            "  yield integral [80,100] GeV; bkg prefit [60,120] GeV.\n\n"
            "Muon-ID efficiency pT output bins:\n"
            "  |eta| 0.0-0.9 : 10,20,30,50,120 GeV\n"
            "  |eta| 0.9-1.2 : 10,30,50,120 GeV\n"
            "  |eta| 1.2-2.1 : 10,30,120 GeV\n"
            "  |eta| 2.1-2.4 : 10,120 GeV\n"
            "  retained edges coincide with analyzer input-bin boundaries.\n\n"
            "Run with python3; never source this file."
        ),
    )

    parser.add_argument(
        "--year",
        "--era",
        dest="year",
        choices=(*VALID_ERAS, "all"),
        default="2023",
        help="data-taking era, or all; default: %(default)s",
    )
    parser.add_argument(
        "--base-region",
        default="MuonIDEfficiency",
        help="base histogram region; default: %(default)s",
    )
    parser.add_argument(
        "--reference",
        default="Top",
        help=(
            "QCD, Top, AllMC, QCDTop, or an explicit ROOT-file path. "
            "The standard merged efficiency inputs provide QCD and ttbar. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--resonance",
        choices=["Jpsi", "Z"],
        default="Jpsi",
        help="resonance used for pass/fail fits; default: %(default)s",
    )
    parser.add_argument(
        "--binning",
        choices=["eta-pt", "pt-only"],
        default="eta-pt",
        help="eta-dependent pT bins or eta-integrated pT bins; default: %(default)s",
    )
    parser.add_argument(
        "--rebin",
        "--rebin-factor",
        dest="rebin_factor",
        type=positive_integer,
        default=3,
        help="nominal mass-histogram rebin factor; standard result fixes rebin=3 and evaluates automatic rebin=1,2,4,5 stability variations; default: %(default)s",
    )
    parser.add_argument(
        "--signal-model",
        choices=["CB", "DSCB"],
        default="CB",
        help="signal model; AN-compatible default: %(default)s",
    )
    parser.add_argument(
        "--background-model",
        default="Bern5",
        help=(
            "Exp1..4, MonoExp1..4, Cheb1..4, MonoCheb1..4, "
            "Bern1..8, or MonoBern1..8; default: %(default)s"
        ),
    )
    parser.add_argument("--fit-min", type=float, default=None)
    parser.add_argument("--fit-max", type=float, default=None)
    parser.add_argument("--yield-min", type=float, default=None)
    parser.add_argument("--yield-max", type=float, default=None)
    parser.add_argument(
        "--yield-mode",
        choices=["integral", "fitnorm"],
        default="fitnorm",
        help=(
            "integral: signal-function integral in the configured window; "
            "fitnorm: fitted signal normalization; default: %(default)s"
        ),
    )
    parser.add_argument("--bkg-fit-min", type=float, default=None)
    parser.add_argument("--bkg-fit-max", type=float, default=None)
    parser.add_argument(
        "--common-shape",
        action="store_true",
        help="fix the separate Pass/Fail signal shapes to the pass+fail (All) fit; default fits them independently",
    )
    parser.add_argument(
        "--no-fix-bkg-shape",
        dest="fix_bkg_shape",
        action="store_false",
        help="do not fix continuum shape parameters to the sideband prefit",
    )
    parser.set_defaults(fix_bkg_shape=True)
    parser.add_argument(
        "--side2-fit-weight",
        type=float,
        default=4.0,
        help="weight of the sideband nearest the signal peak; default: %(default)s",
    )
    parser.add_argument(
        "--linear-bkg-fit",
        dest="use_log_bkg_fit",
        action="store_false",
        help="fit sideband density rather than log-density",
    )
    parser.set_defaults(use_log_bkg_fit=True)
    parser.add_argument(
        "--min-bkg-rel-err",
        type=float,
        default=0.015,
        help="minimum relative error in log-background prefit; default: %(default)s",
    )

    parser.add_argument(
        "--no-per-bin-plots",
        dest="save_per_bin_plots",
        action="store_false",
        help="skip background-prefit and final-fit plots for individual bins",
    )
    parser.set_defaults(save_per_bin_plots=True)
    parser.add_argument(
        "--no-summary-plots",
        dest="save_summary_plots",
        action="store_false",
        help="skip final efficiency and scale-factor panels",
    )
    parser.set_defaults(save_summary_plots=True)
    parser.add_argument(
        "--bin-filter",
        default="all",
        help="all, Inclusive, noInclusive, or a substring of a bin tag",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=-1,
        help="maximum selected bins; -1 means all",
    )
    parser.add_argument(
        "--hist-name",
        default="auto",
        help="mass histogram name; auto follows --resonance",
    )
    parser.add_argument(
        "--include-inclusive",
        action="store_true",
        help="also process the optional Inclusive bin",
    )
    parser.add_argument(
        "--inspect-hists",
        action="store_true",
        help="print matching ROOT keys and return without fitting",
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="process only the first selected bin and save no plots",
    )

    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the resolved equivalent Python command and configuration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved paths/configuration without importing ROOT or fitting",
    )
    return parser


def apply_defaults(args: argparse.Namespace) -> None:
    fit_min, fit_max, hist_name, y_min, y_max, bkg_min, bkg_max = resonance_defaults(
        args.resonance
    )
    if args.fit_min is None:
        args.fit_min = fit_min
    if args.fit_max is None:
        args.fit_max = fit_max
    if args.hist_name == "auto":
        args.hist_name = hist_name
    if args.yield_min is None:
        args.yield_min = y_min
    if args.yield_max is None:
        args.yield_max = y_max
    if args.bkg_fit_min is None:
        args.bkg_fit_min = bkg_min
    if args.bkg_fit_max is None:
        args.bkg_fit_max = bkg_max


def selected_eras(value: str) -> Sequence[str]:
    return VALID_ERAS if value == "all" else (value,)


def input_dir(era: str) -> Path:
    return INPUT_BASE / INPUT_COLLECTION / era


def resolve_reference(era: str, raw_reference: str) -> Tuple[str, Path]:
    directory = input_dir(era)
    key = "".join(ch for ch in raw_reference.lower() if ch.isalnum())

    if key in {"qcd", "qcdmc"}:
        return "QCD", directory / "NIsoMuon_QCD_Inclusive.root"
    if key in {"top", "tt", "ttbar", "tops", "tttw"}:
        return "Top", directory / "NIsoMuon_Top.root"
    if key in {"allmc", "mc"}:
        return "AllMC", directory / "NIsoMuon_AllMC.root"
    if key in {"qcdtop", "qcdtops", "qcdtttw"}:
        return "QCDTop", directory / "NIsoMuon_QCDTop.root"

    supplied = Path(os.path.expandvars(os.path.expanduser(raw_reference)))
    if not supplied.is_absolute():
        supplied = directory / supplied
    return str(supplied.resolve()), supplied.resolve()


def validate_inputs(
    era: str,
    reference_path: Path,
    *,
    allow_missing: bool,
) -> Tuple[Path, Path]:
    directory = input_dir(era)
    data_path = directory / "data.root"
    missing = [path for path in (data_path, reference_path) if not path.is_file()]

    print(f"[INPUT] era       : {era}")
    print(f"[INPUT] directory : {directory}")
    print(f"[INPUT] data      : {data_path}")
    print(f"[INPUT] reference : {reference_path}")

    if missing and not allow_missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing required ROOT file(s):\n  {rendered}\n"
            "Run hadd.sh for the corresponding MuonIDEfficiency era first."
        )
    for path in missing:
        print(f"[DRY-RUN WARNING] Missing file: {path}", file=sys.stderr)
    return data_path, reference_path


def output_dir(era: str) -> Path:
    return PLOT_BASE / era


def equivalent_command(args: argparse.Namespace, era: str) -> str:
    cmd: List[str] = [sys.executable, str(script_path()), "--year", era]
    cmd += ["--base-region", args.base_region]
    cmd += ["--reference", args.reference]
    cmd += ["--resonance", args.resonance]
    cmd += ["--binning", args.binning]
    cmd += ["--rebin", str(args.rebin_factor)]
    cmd += ["--signal-model", args.signal_model]
    cmd += ["--background-model", args.background_model]
    cmd += ["--fit-min", str(args.fit_min), "--fit-max", str(args.fit_max)]
    cmd += ["--yield-min", str(args.yield_min), "--yield-max", str(args.yield_max)]
    cmd += ["--yield-mode", args.yield_mode]
    cmd += ["--bkg-fit-min", str(args.bkg_fit_min), "--bkg-fit-max", str(args.bkg_fit_max)]
    cmd += ["--side2-fit-weight", str(args.side2_fit_weight)]
    cmd += ["--min-bkg-rel-err", str(args.min_bkg_rel_err)]
    cmd += ["--bin-filter", args.bin_filter, "--max-bins", str(args.max_bins)]
    cmd += ["--hist-name", args.hist_name]
    if args.common_shape:
        cmd.append("--common-shape")
    if not args.fix_bkg_shape:
        cmd.append("--no-fix-bkg-shape")
    if not args.use_log_bkg_fit:
        cmd.append("--linear-bkg-fit")
    if not args.save_per_bin_plots:
        cmd.append("--no-per-bin-plots")
    if not args.save_summary_plots:
        cmd.append("--no-summary-plots")
    if args.include_inclusive:
        cmd.append("--include-inclusive")
    if args.inspect_hists:
        cmd.append("--inspect-hists")
    return shell_join(cmd)


def import_root():
    original_argv = sys.argv[:]
    try:
        # Prevent ROOT from interpreting this script's argparse options.
        sys.argv = [sys.argv[0]]
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import PyROOT. Activate the Nano/ROOT environment first. "
            f"Original error: {exc}"
        ) from exc
    finally:
        sys.argv = original_argv

    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    try:
        ROOT.TH1.AddDirectory(False)
    except Exception:
        pass
    return ROOT


def load_embedded_cpp(ROOT) -> Path:
    """Compile/load the embedded C++ through ACLiC.

    Direct gInterpreter.Declare() is intentionally not used.  In the tamsa
    Nano ROOT environment it can fail to materialize libGpad symbols such as
    TCanvas/TPad for this large STL-heavy macro.
    """

    required_libraries = (
        "libCore",
        "libRIO",
        "libHist",
        "libGraf",
        "libGpad",
        "libTree",
        "libMathCore",
        "libMatrix",
    )
    optional_libraries = ("libMinuit", "libMinuit2")

    for library in required_libraries:
        status = int(ROOT.gSystem.Load(library))
        if status < 0:
            raise RuntimeError(f"Could not load required ROOT library: {library}")

    for library in optional_libraries:
        status = int(ROOT.gSystem.Load(library))
        if status < 0:
            print(f"[WARNING] Optional ROOT library not loaded: {library}")

    digest = hashlib.sha256(CPP_SOURCE.encode("utf-8")).hexdigest()[:16]
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    source_path = CACHE_BASE / f"id_eff_embedded_{digest}.C"

    if not source_path.exists() or source_path.read_text() != CPP_SOURCE:
        source_path.write_text(CPP_SOURCE)
        print(f"[INFO] Wrote ACLiC cache source: {source_path}")

    status = int(ROOT.gROOT.LoadMacro(str(source_path) + "+"))
    if status < 0:
        raise RuntimeError(
            "ROOT ACLiC failed while compiling/loading the embedded id_eff source: "
            f"{source_path}"
        )

    return source_path


def run_one_era(ROOT, args: argparse.Namespace, era: str) -> int:
    reference_for_cpp, reference_path = resolve_reference(era, args.reference)
    validate_inputs(era, reference_path, allow_missing=False)

    out_dir = output_dir(era)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OUTPUT] {out_dir}")
    print(
        "[CONFIG] "
        f"resonance={args.resonance}, hist={args.hist_name}, "
        f"binning={args.binning}, AN-fit, nominal-rebin=3 (variations=1,2,4,5), "
        f"fit=[{args.fit_min},{args.fit_max}], "
        f"yield-mode={args.yield_mode}, "
        f"yield=[{args.yield_min},{args.yield_max}], "
        f"reference={args.reference}"
    )

    # The embedded C++ now handles an empty Trigger without adding an obsolete
    # trigger-directory level.  This resolves exactly to:
    #   INPUT_BASE/MuonIDEfficiency/<era>/
    def cpp_string(value: object) -> str:
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'

    def cpp_bool(value: bool) -> str:
        return "true" if value else "false"

    call_args = [
        cpp_string(era),
        cpp_string(""),
        cpp_string(INPUT_BASE),
        cpp_string(INPUT_COLLECTION),
        cpp_string(args.base_region),
        str(int(args.rebin_factor)),
        cpp_string(args.signal_model),
        cpp_string(args.background_model),
        repr(float(args.fit_min)),
        repr(float(args.fit_max)),
        cpp_bool(bool(args.common_shape)),
        cpp_bool(bool(args.fix_bkg_shape)),
        repr(float(args.side2_fit_weight)),
        cpp_bool(bool(args.use_log_bkg_fit)),
        repr(float(args.min_bkg_rel_err)),
        cpp_string(reference_for_cpp),
        cpp_string(out_dir),
        cpp_bool(bool(args.save_per_bin_plots)),
        cpp_bool(bool(args.save_summary_plots)),
        cpp_string(args.bin_filter),
        str(int(args.max_bins)),
        cpp_string(args.hist_name),
        cpp_bool(bool(args.include_inclusive)),
        cpp_bool(bool(args.inspect_hists)),
        cpp_string(args.resonance),
        cpp_string(args.binning),
        repr(float(args.yield_min)),
        repr(float(args.yield_max)),
        repr(float(args.bkg_fit_min)),
        repr(float(args.bkg_fit_max)),
        cpp_string(args.yield_mode),
    ]

    call = "id_eff(" + ",".join(call_args) + ");"
    ROOT.gROOT.ProcessLine(call)

    return 0


def run_dry(args: argparse.Namespace) -> int:
    for era in selected_eras(args.year):
        print("\n" + "=" * 78)
        print(f"[MUON ID EFFICIENCY DRY RUN] {era}")
        print("=" * 78)
        _, reference_path = resolve_reference(era, args.reference)
        validate_inputs(era, reference_path, allow_missing=True)
        print(f"[OUTPUT] {output_dir(era)}")
        print(f"[COMMAND] {equivalent_command(args, era)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv:
        parser.print_help(sys.stdout)
        return 0

    args = parser.parse_args(argv)
    apply_defaults(args)

    if args.quick_check:
        args.max_bins = 1
        args.save_per_bin_plots = False
        args.save_summary_plots = False

    if args.dry_run:
        return run_dry(args)

    if args.print_command:
        for era in selected_eras(args.year):
            print(f"[COMMAND] {equivalent_command(args, era)}")

    try:
        ROOT = import_root()
        cache_path = load_embedded_cpp(ROOT)
        print(f"[INFO] Loaded the embedded C++ with ACLiC: {cache_path}")

        for era in selected_eras(args.year):
            print("\n" + "=" * 78)
            print(f"[MUON ID EFFICIENCY] {era}")
            print("=" * 78)
            status = run_one_era(ROOT, args, era)
            if status != 0:
                return status
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

