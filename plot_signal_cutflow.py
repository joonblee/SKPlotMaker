#!/usr/bin/env python3
"""Plot cumulative signal selection fractions from the SKPlotMaker cutflow CSVs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from textwrap import fill


def read_cutflows(directory):
    files = (
        ("signal_reco_cutflow.csv", "reco", "cumulative_reco_fraction"),
        ("signal_fiducial_cutflow.csv", None, "cumulative_fraction"),
    )
    cutflows = defaultdict(list)
    for filename, level, column in files:
        path = directory / filename
        if level is None and not path.exists():
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["era"], level or row["level"], int(row["mass_GeV"]))
                value = 100 * float(row[column]) if row[column] else float("nan")
                cutflows[key].append((int(row["step"]), row["selection"], value))
    return cutflows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Directory produced by collect_jetveto_cutflow.py")
    parser.add_argument("--era", help="Plot one era (default: all eras in the CSV)")
    parser.add_argument("--mass", type=int, nargs="+", help="Signal masses in GeV (default: all available)")
    parser.add_argument("--out", type=Path, help="PDF directory (default: SUMMARY/plots)")
    args = parser.parse_args()
    try:
        cutflows = read_cutflows(args.summary)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    eras = sorted({era for era, level, _ in cutflows if level == "reco"})
    if args.era and args.era not in eras:
        parser.error(f"No reco cutflow for {args.era} in {args.summary}")
    if not eras:
        parser.error(f"No reco cutflows in {args.summary}")
    eras = [args.era] if args.era else eras
    for era in eras:
        missing = set(args.mass or ()) - {m for e, level, m in cutflows if e == era and level == "reco"}
        if missing:
            parser.error(f"No reco cutflow for {era} at masses {sorted(missing)} GeV")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.out or args.summary / "plots"
    output.mkdir(parents=True, exist_ok=True)
    levels = (
        ("reco", "Reco selection"),
        ("truth", "Particle-level selection after GEN filter"),
        ("reco_fiducial", "Reco selection in the particle-level fiducial region"),
    )
    for era in eras:
        for level, title in levels:
            curves = {
                mass: sorted(rows) for (period, kind, mass), rows in cutflows.items()
                if period == era and kind == level and (args.mass is None or mass in args.mass)
            }
            if not curves:
                continue
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            steps, labels, _ = zip(*next(iter(curves.values())))
            for mass, rows in sorted(curves.items()):
                ax.plot(
                    [row[0] for row in rows], [row[2] for row in rows],
                    marker="o", markersize=4, label=f"{mass} GeV",
                )
            ax.set_xticks(steps, [fill(label, width=17) for label in labels])
            ax.set_ylabel("Cumulative fraction [%]")
            ax.set_title(f"{era}  |  {title}", loc="left")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(title="Signal mass", ncol=min(4, len(curves)))
            path = output / f"signal_cutflow_{era}_{level}.pdf"
            fig.savefig(path)
            plt.close(fig)
            print(path)


if __name__ == "__main__":
    main()
