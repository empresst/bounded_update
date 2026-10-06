"""
script30_noise_floor.py
Seed-to-seed noise floor for the five-seed replication: how closely do two models
of the SAME arm, trained under different seeds, agree with each other, compared
with how closely each agrees with the source model? Same IG, cases and
checkpoints as script28.  Outputs  noise_floor.json
"""
import json, itertools
import numpy as np, torch, polars as pl
from scipy.stats import spearmanr
from script22_relocation_check import (SAVE_PATH, BASE_PATH, SEQ_LEN, SENTINEL, LEAKAGE_TIMING_FEATS,
    SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, load_enriched_split, normalize, ICUDataset,
    TwoStreamModel, integrated_gradients)

mk = torch.load(SAVE_PATH / "multiseed_adapted_weights.pt", map_location="cpu", weights_only=False)
expl = json.load(open(SAVE_PATH / "pubmed_rag_explanations.json"))
cases = [(c["stay_id"], c["label"]) for c in expl["source"]]
df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
for f in LEAKAGE_TIMING_FEATS:
    df = df.with_columns(pl.when(pl.col(f) > SEQ_LEN).then(SENTINEL).otherwise(pl.col(f)).alias(f))
df = normalize(df, mk["train_stats"]).drop(["vasopressor_flag", "ventilation_flag"], strict=False)
ds = ICUDataset(df.filter(pl.col("stay_id").is_in([c[0] for c in cases])), SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
idx = {int(s): i for i, s in enumerate(ds.stay_ids)}

def attrs(state):
    m = TwoStreamModel(mk["seq_dim"], mk["treat_dim"], mk["n_targets"]); m.load_state_dict(state); m.eval()
    P, T = [], []
    for sid, lab in cases:
        xs, xt, _ = ds[idx[sid]]
        sa, ta = integrated_gradients(m, xs.unsqueeze(0), xt.unsqueeze(0), LABEL_COLS.index(lab))
        P.append(sa[0].sum(0)); T.append(ta[0])
    return np.array(P), np.array(T)

agree = lambda A, B: float(np.mean(np.nan_to_num([spearmanr(a, b)[0] for a, b in zip(A, B)])))
src = attrs(mk["source"])
A = {(arm, s): attrs(mk["states"][arm][s]) for arm in ("B", "C") for s in mk["seeds"]}
out = {}
for arm in ("B", "C"):
    for k, name in ((0, "physio"), (1, "treat")):
        vs_src = [agree(src[k], A[(arm, s)][k]) for s in mk["seeds"]]
        within = [agree(A[(arm, s1)][k], A[(arm, s2)][k]) for s1, s2 in itertools.combinations(mk["seeds"], 2)]
        out[f"{arm}_{name}"] = dict(vs_source_mean=float(np.mean(vs_src)), vs_source_range=[float(min(vs_src)), float(max(vs_src))],
                                   seed_to_seed_mean=float(np.mean(within)), seed_to_seed_range=[float(min(within)), float(max(within))])
        print(f"{arm} {name:6s} vs source {np.mean(vs_src):.3f} [{min(vs_src):.3f},{max(vs_src):.3f}]  "
              f"seed-to-seed {np.mean(within):.3f} [{min(within):.3f},{max(within):.3f}]")
json.dump(out, open(SAVE_PATH / "noise_floor.json", "w"), indent=1)
