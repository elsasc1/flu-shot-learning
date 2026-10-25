"""Command line: python -m flushot <command>.

  train MODEL [MODEL ...]   CV + full-data refit, saves outputs/preds_MODEL.npz
  baseline                  logistic regression baseline, saves outputs/preds_logreg.npz
  compare                   OOF AUC of every combination of saved models
  submit MODEL [MODEL ...]  average the models, check the format, write the submission
  fake-data                 write synthetic data with the real schema (for a smoke run)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import baseline, cv, data, submission, synthetic
from .models import MODELS


def _load_preds(out_dir: Path, names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    preds = {}
    for name in names:
        path = out_dir / f"preds_{name}.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found: run `python -m flushot train {name}` first.")
        with np.load(path) as f:
            preds[name] = dict(f)
    return preds


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="flushot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("models", nargs="+", choices=MODELS)
    sub.add_parser("baseline")
    sub.add_parser("compare")
    sm = sub.add_parser("submit")
    sm.add_argument("models", nargs="+")
    sm.add_argument("--output", type=Path, default=None)
    fk = sub.add_parser("fake-data")
    fk.add_argument("--n-train", type=int, default=600)
    fk.add_argument("--n-test", type=int, default=200)
    args = p.parse_args(argv)

    if args.cmd == "fake-data":
        d = synthetic.write(args.data_dir, n_train=args.n_train, n_test=args.n_test)
        print(f"Synthetic data written to {d}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd in ("train", "baseline"):
        X, Y, T = data.load(args.data_dir)
        runs = [("logreg", lambda: baseline.run(X, Y, T))] if args.cmd == "baseline" else \
               [(m, lambda m=m: cv.run(m, X, Y, T)) for m in args.models]
        for name, fn in runs:
            preds = fn()
            np.savez(args.out_dir / f"preds_{name}.npz", **preds)
            print(f"{name}: OOF mean AUC {cv.mean_auc(Y, preds):.5f} -> {args.out_dir / f'preds_{name}.npz'}")
    elif args.cmd == "compare":
        _, Y, _ = data.load(args.data_dir)
        names = [f.stem.removeprefix("preds_") for f in sorted(args.out_dir.glob("preds_*.npz"))]
        for combo, score in submission.rank_combos(Y, _load_preds(args.out_dir, names)):
            print(f"{'+'.join(combo):35s} {score:.5f}")
    elif args.cmd == "submit":
        _, Y, _ = data.load(args.data_dir)
        preds = list(_load_preds(args.out_dir, args.models).values())
        oof = submission.average(preds, "oof")
        score = np.mean([cv.roc_auc_score(Y[t], oof[t]) for t in data.TARGETS])
        sub_df = submission.build(data.load_format(args.data_dir), submission.average(preds, "test"))
        out = args.output or args.out_dir / f"submission_{'+'.join(args.models)}.csv"
        sub_df.to_csv(out, index=False)
        print(f"Blend {'+'.join(args.models)}: OOF mean AUC {score:.5f}. Checks OK -> {out}")


if __name__ == "__main__":
    main()
