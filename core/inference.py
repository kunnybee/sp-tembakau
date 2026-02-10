import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

UNKNOWN_LABEL = "Unknown_Penyakit"

def normalize_gejala_input(gejala_input):
    if isinstance(gejala_input, list) and len(gejala_input) == 1 and "," in gejala_input[0]:
        gejala_input = [g.strip() for g in gejala_input[0].split(",") if g.strip()]
    if isinstance(gejala_input, str):
        gejala_input = [g.strip() for g in gejala_input.split(",") if g.strip()]
    return gejala_input

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build_df_gejala(disease2gejala: dict) -> tuple[dict, int]:
    df_gejala = {}
    for did, gset in disease2gejala.items():
        for g in gset:
            df_gejala[g] = df_gejala.get(g, 0) + 1
    N_disease = max(1, len(disease2gejala))
    return df_gejala, N_disease

def idf(gid: str, df_gejala: dict, N_disease: int) -> float:
    df = df_gejala.get(gid, 1)
    return float(np.log((N_disease + 1e-9) / (df + 1e-9)))

def att_weight(
    gid: str,
    df_gejala: dict,
    N_disease: int,
    gamma=2.0,
    umum_floor=0.03,
    w_clip=5.0
) -> float:
    w = max(idf(gid, df_gejala, N_disease), 1e-6) ** gamma
    if df_gejala.get(gid, 1) > 1:
        w = max(w, umum_floor)
    w = min(w, w_clip)
    return float(w)

def agg_logsum(term_values, lam=1.0) -> float:
    t = np.maximum(np.array(term_values, dtype=float), 0.0)
    return float(np.sum(np.log1p(lam * t)))

def compute_confidence(
    topk,
    main_name,
    T_margin=0.25,
    w_margin=0.20,
    w_coverage=0.45,
    w_specific=0.15,
    w_inputfit=0.20,
    cov_boost_start=0.75,
    cov_boost_full=0.95,
    khas_boost_min=1,
    boost_max=0.25,
    cap_max=0.99
):
    if len(topk) == 0:
        return 0.0

    s1 = topk[0]["score"]
    s2 = topk[1]["score"] if len(topk) > 1 else s1 - 1e-6
    C_margin = sigmoid((s1 - s2) / T_margin)

    if main_name == UNKNOWN_LABEL:
        return float((1 - C_margin) * 60)

    matched_count = len(topk[0]["matched_gejala"])
    total_count = max(1, int(topk[0]["total_gejala_penyakit"]))
    khas_match = int(topk[0]["matched_khas_count"])

    C_coverage = min(1.0, matched_count / total_count)
    C_specific = (khas_match + 1) / (matched_count + 2)
    C_specific = float(np.clip(C_specific, 0.0, 1.0))

    C_inputfit = float(topk[0].get("input_fit", 1.0))

    base = (
        (w_margin * C_margin) +
        (w_coverage * C_coverage) +
        (w_specific * C_specific) +
        (w_inputfit * C_inputfit)
    )

    boost = 0.0
    if khas_match >= khas_boost_min and C_coverage >= cov_boost_start:
        t = (C_coverage - cov_boost_start) / max((cov_boost_full - cov_boost_start), 1e-9)
        t = float(np.clip(t, 0.0, 1.0))
        boost = boost_max * t

    conf = min(base + boost, cap_max)
    return float(conf * 100)

def diagnose(
    gejala_input,
    Z: np.ndarray,
    maps: dict,
    top_k=10,
    gamma_w=2.0,
    umum_floor=0.03,
    w_clip=5.0,
    lam_logsum=1.0,
    margin_delta=0.08,
    min_khas_match=1,
    min_match_total=1,
    min_weighted_evidence=0.15
):
    name2id = maps["name2id"]
    id2name = maps["id2name"]
    id2idx  = maps["id2idx"]
    disease_ids = maps["disease_ids"]
    disease_names = maps["disease_names"]
    disease2gejala = maps["disease2gejala"]

    gejala_input = normalize_gejala_input(gejala_input)

    # mapping input gejala (kg_name) -> node_id
    gejala_ids = [name2id[g] for g in gejala_input if g in name2id]
    if len(gejala_ids) == 0:
        raise ValueError("Tidak ada gejala input valid (cek mapping UI→KG).")

    gejala_ids = [g for g in gejala_ids if g in id2idx]
    if len(gejala_ids) == 0:
        raise ValueError("Gejala valid, tapi tidak ada embedding (cek nodes vs Z).")

    Gin = set(gejala_ids)

    df_gejala, N_disease = build_df_gejala(disease2gejala)

    emb_g = {g: Z[id2idx[g]] for g in gejala_ids}
    w_g   = {g: att_weight(g, df_gejala, N_disease, gamma=gamma_w, umum_floor=umum_floor, w_clip=w_clip) for g in gejala_ids}

    has_khas_input = any(df_gejala.get(g, 1) <= 1 for g in gejala_ids)

    results = []
    for did, dname in zip(disease_ids, disease_names):
        if did not in id2idx:
            continue

        Gd = disease2gejala.get(did, set())
        matched = list(Gin & Gd)
        mismatch = list(Gin - set(matched))
        inputfit = len(matched) / max(1, len(Gin))

        if len(matched) == 0:
            results.append((dname, -1e9, 0, [], len(Gd), 0.0, float(inputfit), [id2name[g] for g in mismatch]))
            continue

        dvec = Z[id2idx[did]].reshape(1, -1)

        term_vals = []
        khas_count = 0
        evidence_sum = 0.0

        for g in matched:
            cos = float(cosine_similarity(emb_g[g].reshape(1, -1), dvec)[0][0])
            term = w_g[g] * cos
            term_vals.append(term)
            evidence_sum += max(term, 0.0)
            if df_gejala.get(g, 1) <= 1:
                khas_count += 1

        score = agg_logsum(term_vals, lam=lam_logsum)

        results.append((
            dname,
            float(score),
            int(khas_count),
            [id2name[g] for g in matched],
            int(len(Gd)),
            float(evidence_sum),
            float(inputfit),
            [id2name[g] for g in mismatch]
        ))

    if len(results) == 0:
        raise RuntimeError("Tidak ada penyakit yang bisa dievaluasi. Cek node_class==Penyakit atau disease2gejala kosong.")

    results.sort(key=lambda x: x[1], reverse=True)

    topk = []
    for dname, score, khas_count, matched_names, total_gejala, evidence_sum, inputfit, mismatch_names in results[:top_k]:
        topk.append({
            "penyakit": dname,
            "score": float(score),
            "matched_khas_count": int(khas_count),
            "matched_gejala": matched_names,
            "total_gejala_penyakit": int(total_gejala),
            "evidence_sum": float(evidence_sum),
            "input_fit": float(inputfit),
            "mismatch_gejala": mismatch_names
        })

    main = topk[0]["penyakit"]
    s1 = topk[0]["score"]
    s2 = topk[1]["score"] if len(topk) > 1 else s1 - 1e-6
    margin = s1 - s2

    if len(topk[0]["matched_gejala"]) < min_match_total:
        main = UNKNOWN_LABEL
    if topk[0]["evidence_sum"] < min_weighted_evidence:
        main = UNKNOWN_LABEL
    if has_khas_input and topk[0]["matched_khas_count"] < min_khas_match:
        main = UNKNOWN_LABEL
    if margin < margin_delta:
        main = UNKNOWN_LABEL

    conf_pct = compute_confidence(topk=topk, main_name=main)
    return main, conf_pct, topk
