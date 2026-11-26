#!/usr/bin/env python3
"""Plot the DY jet-veto map on/off ratio from the 2018 and 2022EE CSVs."""

import argparse
import csv
from math import fsum
from pathlib import Path

MASS_EDGES = (11, 15, 20, 25, 30, 35, 40, 50, 60, 80)
REGIONS = (("BJet", "b-jet region"), ("LightJet", "Light-jet region"))


def read_rows(path, era):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["era"] == era]
    if not rows:
        raise ValueError(f"No rows for {era} in {path}")
    return rows


def rebin_ratios(rows, region):
    points = []
    region_rows = [row for row in rows if row["region"] == region]
    for low, high in zip(MASS_EDGES, MASS_EDGES[1:]):
        bins = [
            row for row in region_rows
            if low <= float(row["mass_low_GeV"]) < high
        ]
        on = fsum(float(row["on"]) for row in bins)
        off = fsum(float(row["off"]) for row in bins)
        if off > 0:
            points.append(((low + high) / 2, on / off))
    return points


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
        ("2018", read_rows(args.csv_2018, "2018"), ROOT.kBlue + 1, 20),
        ("2022EE", read_rows(args.csv_2022ee, "2022EE"), ROOT.kRed + 1, 24),
    )
    canvas = ROOT.TCanvas("jetveto", "DY jet veto map", 1050, 440)
    canvas.Divide(2, 1)
    objects = []

    for panel, (region, title) in enumerate(REGIONS, start=1):
        pad = canvas.cd(panel)
        pad.SetLeftMargin(0.15)
        graphs = []
        values = [1.0]

        for era, rows, colour, marker in sources:
            points = rebin_ratios(rows, region)
            if not points:
                raise ValueError(f"No positive map-off yields for {era} in {region}")
            graph = ROOT.TGraph()
            for centre, ratio in points:
                graph.SetPoint(graph.GetN(), centre, ratio)
                values.append(ratio)
            graph.SetMarkerStyle(marker)
            graph.SetMarkerSize(0.8)
            graph.SetMarkerColor(colour)
            graphs.append((era, graph))

        low, high = min(values), max(values)
        margin = max(0.01, 0.1 * (high - low))
        frame = pad.DrawFrame(MASS_EDGES[0], low - margin, MASS_EDGES[-1], high + margin)
        frame.SetTitle(f"{title};m_{{#mu#mu}} [GeV];DY yield ratio (map on / off)")

        line = ROOT.TLine(MASS_EDGES[0], 1, MASS_EDGES[-1], 1)
        line.SetLineStyle(2)
        line.Draw()

        legend = ROOT.TLegend(0.66, 0.17, 0.88, 0.32)
        legend.SetBorderSize(0)
        for era, graph in graphs:
            graph.Draw("P SAME")
            legend.AddEntry(graph, era, "p")
        legend.Draw()
        objects.extend([frame, line, legend, *(graph for _, graph in graphs)])

    canvas.SaveAs(str(args.output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(str(error)) from error
