#!/usr/bin/env python3
"""Plot the DY jet-veto map on/off ratio from the 2018 and 2022EE CSVs."""

import argparse
import csv
from array import array
from itertools import groupby
from math import fsum
from pathlib import Path

MASS_EDGES = (11, 20, 30, 50, 80)
REGIONS = (("BJet", "b-jet region"), ("LightJet", "Light-jet region"))


def read_rows(path, era):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["era"] == era]
    if not rows:
        raise ValueError(f"No rows for {era} in {path}")
    return rows


def rebin_ratios(rows, region):
    ratios = []
    region_rows = [row for row in rows if row["region"] == region]
    for low, high in zip(MASS_EDGES, MASS_EDGES[1:]):
        bins = [
            row for row in region_rows
            if low <= float(row["mass_low_GeV"]) < high
        ]
        on = fsum(float(row["on"]) for row in bins)
        off = fsum(float(row["off"]) for row in bins)
        ratios.append(on / off if off > 0 else None)
    return ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-2018", type=Path,
                        default=Path("jetveto_2018/dy_veto_mass_bins.csv"))
    parser.add_argument("--csv-2022ee", type=Path,
                        default=Path("jetveto_2022EE/dy_veto_mass_bins.csv"))
    parser.add_argument("--output", type=Path, default=Path("dy_jetveto_map_ratio_rebinned.pdf"))
    args = parser.parse_args()

    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    sources = (
        ("2018", read_rows(args.csv_2018, "2018"), ROOT.kBlue + 1, 1),
        ("2022EE", read_rows(args.csv_2022ee, "2022EE"), ROOT.kRed + 1, 2),
    )
    canvas = ROOT.TCanvas("jetveto", "DY jet veto map", 1050, 440)
    canvas.Divide(2, 1)
    objects = []

    for panel, (region, title) in enumerate(REGIONS, start=1):
        pad = canvas.cd(panel)
        pad.SetLeftMargin(0.15)
        histograms = []
        values = [1.0]

        for era, rows, colour, line_style in sources:
            ratios = rebin_ratios(rows, region)
            segments = []
            # Leave a gap wherever the map-off yield is nonpositive.
            for valid, group in groupby(enumerate(ratios), key=lambda item: item[1] is not None):
                if not valid:
                    continue
                bins = list(group)
                first, last = bins[0][0], bins[-1][0]
                edges = array("d", MASS_EDGES[first:last + 2])
                hist = ROOT.TH1D(f"ratio_{era}_{region}_{first}", "", len(bins), edges)
                hist.SetDirectory(0)
                hist.SetLineColor(colour)
                hist.SetLineStyle(line_style)
                hist.SetLineWidth(2)
                for bin_number, (_, ratio) in enumerate(bins, start=1):
                    hist.SetBinContent(bin_number, ratio)
                    values.append(ratio)
                segments.append(hist)
            if not segments:
                raise ValueError(f"No positive map-off yields for {era} in {region}")
            histograms.append((era, segments))

        low, high = min(values), max(values)
        margin = max(0.01, 0.1 * (high - low))
        frame = pad.DrawFrame(MASS_EDGES[0], low - margin, MASS_EDGES[-1], high + margin)
        frame.SetTitle(f"{title};m_{{#mu#mu}} [GeV];DY yield ratio (map on / off)")

        line = ROOT.TLine(MASS_EDGES[0], 1, MASS_EDGES[-1], 1)
        line.SetLineStyle(2)
        line.Draw()

        legend = ROOT.TLegend(0.66, 0.17, 0.88, 0.32)
        legend.SetBorderSize(0)
        for era, segments in histograms:
            for hist in segments:
                hist.Draw("HIST SAME")
            legend.AddEntry(segments[0], era, "l")
            objects.extend(segments)
        legend.Draw()
        objects.extend([frame, line, legend])

    canvas.SaveAs(str(args.output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(str(error)) from error
