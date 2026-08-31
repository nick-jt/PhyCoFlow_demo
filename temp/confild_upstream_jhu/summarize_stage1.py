from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare isolated CoNFiLD stage-1 runs")
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        config = json.loads((run / "config.json").read_text())
        records = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        diagnostics = [record for record in records if "train_rel_l2" in record]
        best = min(diagnostics, key=lambda record: record["train_rel_l2"])
        rows.append(
            {
                "run": str(run),
                "latent": config["latent_dim"],
                "decoder_lr": config["decoder_lr"],
                "latent_lr": config["latent_lr"],
                "points": config["points_per_item"],
                "best_epoch": best["epoch"],
                "best_train_rel_l2": best["train_rel_l2"],
                "latent_rms": best["latent_rms"],
            }
        )
    rows.sort(key=lambda row: row["best_train_rel_l2"])
    headings = list(rows[0]) if rows else []
    if headings:
        print("\t".join(headings))
        for row in rows:
            print("\t".join(str(row[key]) for key in headings))


if __name__ == "__main__":
    main()

