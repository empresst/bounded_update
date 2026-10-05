"""
script28_multiseed_extras.py
Five-seed movement and model-level separability (supplementary material, Table S4):
  1. attribution-mass shift |mean|IG|_arm - mean|IG|_src| per seed
  2. vector distance mean_c |a_arm - a_src| / mean_c |a_src|
  3. model-level separability: mean signed rho per model, 5 B vs 5 C models
  4. treatment-stream signed rho per seed
Reuses the IG, data loading and checkpoints of script22_relocation_check.py.
Outputs  multiseed_extras.json
"""
import json, time
import numpy as np
import torch
from scipy.stats import spearmanr
from script22_relocation_check import (SAVE_PATH, BASE_PATH, SEQ_LEN, SENTINEL, LEAKAGE_TIMING_FEATS,
                                            SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, load_enriched_split,
                                            normalize, ICUDataset, TwoStreamModel, integrated_gradients, paired)
import polars as pl


def main():
    t0 = time.time()
    mk = torch.load(SAVE_PATH / "multiseed_adapted_weights.pt", map_location="cpu", weights_only=False)
    states, SEEDS, src_state, ts = mk["states"], mk["seeds"], mk["source"], mk["train_stats"]
    expl = json.load(open(SAVE_PATH / "pubmed_rag_explanations.json"))
    cases = [(c["stay_id"], c["label"]) for c in expl["source"]]
    import os
    n_pilot = int(os.environ.get("PILOT_N", "0"))    # pilot: every k-th case, keeps the outcome mix
    if n_pilot:
        cases = cases[::max(1, len(cases) // n_pilot)][:n_pilot]
    df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
    for f in LEAKAGE_TIMING_FEATS:
        if f in df.columns:
            df = df.with_columns(pl.when(pl.col(f) > SEQ_LEN).then(SENTINEL).otherwise(pl.col(f)).alias(f))
    df = normalize(df, ts).drop([c for c in ["vasopressor_flag", "ventilation_flag"] if c in df.columns])
    ds = ICUDataset(df.filter(pl.col("stay_id").is_in([c[0] for c in cases])),
                    SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    idx = {int(s): i for i, s in enumerate(ds.stay_ids)}

    def attrs(state):
        m = TwoStreamModel(mk["seq_dim"], mk["treat_dim"], mk["n_targets"]); m.load_state_dict(state); m.eval()
        P, T = [], []
        for sid, lab in cases:
            xs, xt, _ = ds[idx[sid]]
            sa, ta = integrated_gradients(m, xs.unsqueeze(0), xt.unsqueeze(0), LABEL_COLS.index(lab))
            P.append(sa[0].sum(axis=0)); T.append(ta[0])
        return np.array(P), np.array(T)

    Ps, Ts = attrs(src_state)
    out = {"seeds": SEEDS, "per_seed": {}}
    for s in SEEDS:
        r = {}
        for arm in ("B", "C"):
            P, T = attrs(states[arm][s])
            r[arm] = dict(
                mass_shift=np.abs(np.abs(P).mean(1) - np.abs(Ps).mean(1)),
                vec_dist=np.abs(P - Ps).mean(1) / np.abs(Ps).mean(1),
                rho_phys=np.array([spearmanr(a, b)[0] for a, b in zip(Ps, P)]),
                rho_treat=np.nan_to_num([spearmanr(a, b)[0] for a, b in zip(Ts, T)]))
        row = {}
        for k in r["B"]:
            o, lo, hi, p = paired(r["B"][k], r["C"][k])
            row[k] = dict(B=float(r["B"][k].mean()), C=float(r["C"][k].mean()), diff=float(o),
                          lo=float(lo), hi=float(hi), p=float(p))
        out["per_seed"][str(s)] = row
        print(f"seed {s} ({time.time()-t0:.0f}s): " + "  ".join(
            f"{k} B{v['B']:.4f} C{v['C']:.4f} d{v['diff']:+.4f}[{v['lo']:+.4f},{v['hi']:+.4f}]"
            for k, v in row.items()))
    for k in ("mass_shift", "vec_dist", "rho_phys", "rho_treat"):
        b = np.array([out["per_seed"][str(s)][k]["B"] for s in SEEDS])
        c = np.array([out["per_seed"][str(s)][k]["C"] for s in SEEDS])
        auc = float((b[:, None] > c[None, :]).mean() + 0.5 * (b[:, None] == c[None, :]).mean())
        out[k] = dict(B=b.tolist(), C=c.tolist(), model_level_auc_B_gt_C=auc,
                      n_seed_B_gt_C=int((b > c).sum()))
        print(f"{k:10s} B {np.round(b,4).tolist()}  C {np.round(c,4).tolist()}  model-AUC(B>C) {auc:.2f}")
    json.dump(out, open(SAVE_PATH / (f"multiseed_extras_pilot{n_pilot}.json" if n_pilot else "multiseed_extras.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
