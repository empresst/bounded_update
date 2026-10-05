"""
script29_grounding_fixed.py
Corrected version of script25 (patient-level grounding of physiology-query retrieval).

Problems in script25 that this fixes:
  1. The logged run used CORPUS_N <= 1177, so the "stratified subset" kept the
     1,177 comorbidity-bearing abstracts and 0 others: every candidate document
     was a positive for someone, so P@5 had no room to discriminate.
     -> full 3,668-abstract broad corpus.
  2. Substring matching ("hypertension" hit pulmonary / portal / intracranial
     hypertension, etc.). -> exact MeSH descriptors (prefix match for families).
  3. No chance level. Without it, 0.30 vs 0.34 is uninterpretable.
     -> shuffled-patient null: score patient i's ground truth against the
        retrievals of every other patient with the same outcome label (the
        query carries the label term, so this isolates what the attribution adds).
  4. Ground truth mismatched to the query. Comorbidities are not model inputs and
     the physiology query names vital signs and labs. -> second, pre-specified
     ground truth: the patient's own abnormal physiology in hours 0-5.

Grounding = observed P@5 - shuffled P@5 (per patient, paired tests).
All definitions below were fixed before the run.
Outputs  grounding_fixed.json ; caches corpus embeddings in corpus_broad_emb.npy
"""
import os
import json, time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModel

SAVE = Path(os.environ.get("OUT_DIR", "."))
TOP_K = 5
rng = np.random.default_rng(42)
torch.set_num_threads(12)

# descriptor families: exact match, or prefix match when the entry ends with '*'
COMORB = {
    "has_hypertension": ["hypertension", "hypertension, malignant", "hypertension, renal",
                         "hypertension, renovascular", "essential hypertension"],
    "has_diabetes": ["diabetes mellitus*", "diabetic*", "diabetes complications"],
    "has_chf": ["heart failure*"],
    "has_ckd": ["renal insufficiency, chronic", "kidney failure, chronic", "renal dialysis",
                "diabetic nephropathies"],
    "has_copd": ["pulmonary disease, chronic obstructive", "pulmonary emphysema", "lung diseases, obstructive"],
    "has_liver_disease": ["liver cirrhosis*", "liver diseases*", "liver failure*", "hepatic encephalopathy",
                          "hepatic insufficiency", "end stage liver disease", "hypertension, portal",
                          "esophageal and gastric varices"],
    "has_malignancy": ["neoplasms*", "*neoplasms", "carcinoma*", "lymphoma*", "leukemia*"],
}
# patient abnormality (raw values, hours 0-5) -> MeSH descriptors
PHYS = {
    "hypotension":   (lambda r: r["map_min"] < 65 or r["sbp_min"] < 90,
                      ["hypotension", "shock", "shock, septic", "shock, cardiogenic"]),
    "tachycardia":   (lambda r: r["hr_max"] > 100, ["tachycardia*", "atrial fibrillation"]),
    "hypoxaemia":    (lambda r: r["spo2_min"] < 90, ["hypoxia", "respiratory insufficiency"]),
    "tachypnoea":    (lambda r: r["rr_max"] > 24, ["tachypnea", "respiratory insufficiency"]),
    "lactate":       (lambda r: r["lac_max"] > 2, ["hyperlactatemia", "acidosis, lactic", "lactic acid"]),
    "aki":           (lambda r: r["cr_max"] > 1.5, ["acute kidney injury", "renal insufficiency"]),
    "low_gcs":       (lambda r: r["gcs_min"] < 13, ["coma", "consciousness disorders", "glasgow coma scale"]),
    "fever":         (lambda r: r["t_max"] > 38.3, ["fever"]),
    "hypothermia":   (lambda r: r["t_min"] < 36, ["hypothermia*"]),
    "hyperglycaemia": (lambda r: r["glu_max"] > 180, ["hyperglycemia", "blood glucose"]),
    "thrombocytopenia": (lambda r: r["plt_min"] < 100, ["thrombocytopenia"]),
    "acidosis":      (lambda r: r["ph_min"] < 7.30, ["acidosis*"]),
    "hyperkalaemia": (lambda r: r["k_max"] > 5.5, ["hyperkalemia"]),
    "hyponatraemia": (lambda r: r["na_min"] < 135, ["hyponatremia"]),
    "anaemia":       (lambda r: r["hct_min"] < 30, ["anemia"]),
    "oliguria":      (lambda r: r["uo_min"] < 0.5, ["oliguria"]),
}
AGG = dict(map_min=("map_invasive", "min"), sbp_min=("sbp_noninvasive", "min"), hr_max=("heart_rate", "max"),
           spo2_min=("spo2", "min"), rr_max=("resp_rate", "max"), lac_max=("lactate", "max"),
           cr_max=("creatinine", "max"), t_max=("temperature_c", "max"), t_min=("temperature_c", "min"),
           glu_max=("glucose", "max"), plt_min=("platelets", "min"), ph_min=("ph_venous", "min"),
           k_max=("potassium", "max"), na_min=("sodium", "min"), hct_min=("hematocrit", "min"),
           uo_min=("urine_output_ml_kg_hr", "min"), gcs_min=("gcs", "min"))


def match(mesh, pats):
    for m in mesh:
        for p in pats:
            if p.startswith("*") and m.endswith(p[1:]): return True
            if p.endswith("*") and m.startswith(p[:-1]): return True
            if m == p: return True
    return False


def paired(a, b, n=10000):
    d = np.asarray(a) - np.asarray(b); o = d.mean()
    p = (np.sum(np.abs((rng.choice([-1, 1], size=(n, len(d))) * d).mean(axis=1)) >= abs(o)) + 1) / (n + 1)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(o), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(p)


def main():
    t0 = time.time()
    corpus = json.load(open(SAVE / "pubmed_corpus_broad.json"))
    import os
    n_pilot = int(os.environ.get("PILOT_N", "0"))    # pilot: uniform random subset, not stratified
    if n_pilot:
        corpus = [corpus[i] for i in sorted(np.random.default_rng(0).choice(len(corpus), n_pilot, replace=False))]
    mesh = [[m.lower() for m in (a.get("mesh") or [])] for a in corpus]
    expl = json.load(open(SAVE / "pubmed_rag_explanations.json"))
    sids = [c["stay_id"] for c in expl["source"]]
    labels = np.array([c["label"] for c in expl["source"]])

    df = (pl.read_parquet(SAVE / "test_final_enriched4.parquet")
          .filter(pl.col("stay_id").is_in(sids) & (pl.col("hrs_from_admit") <= 5))
          .with_columns((pl.col("gcs_eye") + pl.col("gcs_verbal") + pl.col("gcs_motor")).alias("gcs")))
    agg = df.group_by("stay_id").agg([getattr(pl.col(c), f)().alias(k) for k, (c, f) in AGG.items()]
                                     + [pl.col(k).max().alias(k) for k in COMORB])
    rows = {int(r["stay_id"]): r for r in agg.iter_rows(named=True)}

    def flags(r):
        r = {k: (np.nan if v is None else v) for k, v in r.items()}
        com = [k for k in COMORB if r[k] > 0]
        ph = [k for k, (f, _) in PHYS.items() if f(r)]
        return com, ph
    F = [flags(rows[s]) for s in sids]
    # doc x concept hits, then doc x patient relevance
    DC = {"comorbidity": {k: np.array([match(m, v) for m in mesh]) for k, v in COMORB.items()},
          "physiology": {k: np.array([match(m, v[1]) for m in mesh]) for k, v in PHYS.items()}}
    R = {}
    for gi, g in enumerate(("comorbidity", "physiology")):
        R[g] = np.stack([np.any([DC[g][k] for k in F[i][gi]], axis=0) if F[i][gi]
                         else np.zeros(len(mesh), bool) for i in range(len(sids))], axis=1)
        print(g, {k: int(v.sum()) for k, v in DC[g].items()})
    for g in R:
        print(f"{g}: patients with >=1 target {int((R[g].any(0)).sum())}/{len(sids)}  "
              f"mean doc base rate {R[g].mean():.3f}")

    cache = SAVE / (f"corpus_broad_emb_pilot{n_pilot}.npy" if n_pilot else "corpus_broad_emb.npy")
    if cache.exists():
        E = np.load(cache)
    else:
        tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        am = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").eval()
        embs = []
        for i in range(0, len(corpus), 32):
            b = corpus[i:i + 32]
            with torch.no_grad():
                enc = tok([e["title"] for e in b], [e["abstract"] for e in b], truncation=True,
                          padding=True, return_tensors="pt", max_length=512)
                embs.append(am(**enc).last_hidden_state[:, 0, :].numpy())
            if (i // 32) % 10 == 0: print(f"  corpus {i+32}/{len(corpus)} ({time.time()-t0:.0f}s)", flush=True)
        E = np.vstack(embs); E /= np.linalg.norm(E, axis=1, keepdims=True); np.save(cache, E)

    qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    qm = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()
    out = {"corpus_size": len(corpus), "n_cases": len(sids)}
    per = {}
    for arm in ("source", "run_b", "run_c"):
        qs = [c["physio_query"] for c in expl[arm]]
        with torch.no_grad():
            e = qt(qs, truncation=True, padding=True, return_tensors="pt", max_length=64)
            Q = qm(**e).last_hidden_state[:, 0, :].numpy()
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        hits = np.argsort(-(Q @ E.T), axis=1)[:, :TOP_K]
        for g, M in R.items():
            P = M[hits].mean(1)            # P[j, i]: precision of patient j's top-5 for patient i's truth
            obs = np.diag(P).copy()
            null = np.array([P[(labels == labels[i]) & (np.arange(len(sids)) != i), i].mean()
                             for i in range(len(sids))])
            keep = M.any(0)                # patients with >=1 target that exists in the corpus
            per[(arm, g)] = (obs[keep], null[keep])
            o, lo, hi, p = paired(obs[keep], null[keep])
            out[f"{arm}/{g}"] = dict(n=int(keep.sum()), observed=float(obs[keep].mean()),
                                     shuffled=float(null[keep].mean()), grounding=o, lo=lo, hi=hi, p=p)
            print(f"{arm:7s} {g:11s} n={keep.sum()} obs {obs[keep].mean():.4f} null {null[keep].mean():.4f} "
                  f"grounding {o:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4g}")
    for g in R:
        for x, y in (("run_b", "run_c"), ("run_b", "source"), ("run_c", "source")):
            gx = per[(x, g)][0] - per[(x, g)][1]; gy = per[(y, g)][0] - per[(y, g)][1]
            o, lo, hi, p = paired(gx, gy)
            out[f"{x}-{y}/{g}"] = dict(diff=o, lo=lo, hi=hi, p=p)
            print(f"  {g:11s} grounding {x}-{y}: {o:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4g}")
    json.dump(out, open(SAVE / (f"grounding_fixed_pilot{n_pilot}.json" if n_pilot else "grounding_fixed.json"), "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
