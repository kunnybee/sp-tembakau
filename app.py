import streamlit as st
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PATH DATA
# =========================
DATA_DIR = Path("data")
NODES_PATH = DATA_DIR / "nodes_filtered.csv"
EDGES_PATH = DATA_DIR / "edges_filtered.csv"
EMB_PATH   = DATA_DIR / "GCN_embeddings.npy"
MAP_PATH   = DATA_DIR / "gejala_ui_map.csv"
DISEASE_MAP_PATH = DATA_DIR / "penyakit_ui_map.csv"

REL_DIAGNOSA = "terdiagnosaPenyakit"
UNKNOWN_LABEL = "Unknown_Penyakit"

st.set_page_config(page_title="Sistem Pakar Tembakau", layout="wide")

# =========================
# CSS (styling mirip UI)
# =========================
st.markdown(
    """
<style>
html, body, [class*="css"]  { font-family: Arial, sans-serif; }

.block-container {
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}

/* header */
.topbar {
  background: #F7E37E;
  padding: 16px 20px;
  border-bottom: 1px solid rgba(0,0,0,0.08);
}
.topbar .small {
    font-family: "Poppins", sans-serif;
    color: #D07C00;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 1.2px;
    margin: 0;
    line-height: 1.1;
    text-transform: uppercase;
}

.topbar .title {
    font-family: "Montserrat", sans-serif;
    color: #111;
    font-weight: 800;
    font-size: 26px;
    margin: 0;
    line-height: 1.1;
}

/* hero */
.hero-wrap { padding: 28px 0 12px 0; }
.hero-left h2 { margin: 0; font-size: 22px; line-height: 1.35; }
.hero-left p { margin: 10px 0 18px 0; color: #333; line-height: 1.5; }

.btn-primary > button {
  background: #C96F0A !important;
  color: white !important;
  border-radius: 18px !important;
  padding: 10px 18px !important;
  border: none !important;
  font-weight: 800 !important;
}
.btn-primary > button:hover { filter: brightness(0.95); }

/* divider */
.hr-orange { border-top: 2px solid rgba(201,111,10,0.65); margin: 18px 0; }

/* patogen pills as buttons */
.pill-row { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; margin-top: 8px; }
.pill-note { text-align:center; color:#333; margin-top: 6px; font-size: 13px; }

.pillbtn > button{
  background: #000000 !important;
  color:#ffffff !important;
  border-radius: 16px !important;
  padding: 10px 26px !important;
  border: none !important;
  font-weight: 800 !important;
  letter-spacing: 0.8px !important;
  font-size: 13px !important;
}
.pillbtn > button:hover{
  background: #222222 !important;
}

/* titles */
.section-title { text-align:center; margin: 6px 0 2px 0; font-weight: 900; font-size: 25px; color: #C96F0A; }
.section-sub { text-align:center; margin: 0 0 15px 0; color: #333; }

/* category */
.cat-title { color:#C96F0A; font-weight: 900; margin-top: 10px; }
.cat-underline { width: 36px; height: 3px; background: #111; margin: 4px 0 10px 0; border-radius: 2px; }
.small-link { color: #777; font-size: 13px; text-decoration: underline; cursor: default; margin-top: 6px; }

/* result */
.result-wrap { margin-top: 16px; }
.result-label { text-align:center; font-weight: 900; font-size: 18px; color: #C96F0A; margin-bottom: 10px; }
.result-bar {
  background: #C96F0A;
  border-radius: 18px;
  padding: 12px 16px;
  color: #fff;
  font-weight: 900;
  text-align: center;
}
.result-bar.muted { background: #C9A27A; }

/* compact checkbox */
div[data-testid="stCheckbox"] label { font-size: 15px; }

/* simple back button */
.backbtn > button{
  background: transparent !important;
  border: 1px solid rgba(0,0,0,0.2) !important;
  border-radius: 12px !important;
  padding: 8px 12px !important;
  font-weight: 700 !important;
}

.topk-wrap{
  background: #ffffff;
  border-radius: 14px;
  padding: 14px 16px;
  margin-top: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.topk-title{
  font-weight: 900;
  margin-bottom: 8px;
  color: #111;
}

.topk-item{
  display: flex;
  justify-content: space-between;
  padding: 8px 10px;
  border-radius: 10px;
  background: #f7f7f7;
  margin-bottom: 6px;
  font-weight: 600;
}

.topk-rank{
  font-weight: 900;
  color: #C96F0A;
}

/* PATOGEN BUTTONS: force style */
/* semua tombol SECONDARY jadi hitam (patogen) */
div[data-testid="stButton"] > button[kind="secondary"]{
  background: #000000 !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 16px !important;
  padding: 10px 26px !important;
  font-weight: 800 !important;
  letter-spacing: 0.8px !important;
  font-size: 13px !important;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover{
  background: #222222 !important;
}

/* tombol PRIMARY tetap orange (Mulai Diagnosis & Diagnosa) */
div[data-testid="stButton"] > button[kind="primary"]{
  background: #C96F0A !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 18px !important;
  padding: 10px 18px !important;
  font-weight: 900 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover{
  background: #B96309 !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================
# STATIC UI LIST (sesuai kamu)
# =========================
GEJALA_UI = {
    "Daun": [
        "Daun terdapat bercak busuk berwarna coklat dan kuning (bercak lanas)",
        "Daun terdapat bercak cincin kecil yang simetris (frogeye)",
        "Daun terdapat bercak titik hitam di tengah bercak putih",
        "Bawah daun terdapat serbuk jamur keabu-abuan di area bercak (frogeye)",
        "Daun terdapat bercak cincin besar yang berpusat jelas (kosentris)",
        "Daun terdapat bercak cincin kecil yang tidak simetris",
        "Bercak cincin pada daun cepat melebar/menyatu",
        "Daun terdapat bercak coklat",
        "Daun terdapat banyak bercak coklat/hitam",
        "Bercak pada daun berminyak",
        "Bercak pada daun kering dan rapuh",
        "Banyak jaringan rusak daun yang rontok (daun berlubang)",
        "Daun terdapat mosaik kontras hijau tua-hijau muda",
        "Daun menyempit",
        "Daun berwarna belang tidak teratur (seperti kulit mentimun)",
        "Daun mengkerut",
        "Urat daun menebal",
        "Daun layu pada siang hari",
        "Daun layu menyeluruh",
        "Daun layu bertahap",
        "Daun mengalami nekrosis",
        "Daun mengalami klorosis",
        "Pusat bercak cincin pada daun tipis dan mudah robek",
        "Daun keriting/bergelombang",
        "Daun terdapat pola jaring laba-laba kuning kehitaman",
    ],
    "Batang": [
        "Empulur bersekat hitam saat dibelah",
        "Empulur jika dicampur dengan air menjadi keruh",
        "Berkas pembuluh pada batang berwarna coklat dan bila ditekan keluar lendir",
        "Batang berlubang saat dibelah",
        "Empulur tidak mau bersatu dengan air",
        "Batang basah/lunak/berlendir dan berbau busuk",
        "Empulur kosong sehingga batang dapat patah di tengah",
        "Batang mengering dan layu",
        "Pangkal batang menghitam",
    ],
    "Akar": [
        "Akar membusuk",
        "Akar membengkak",
        "Akar terdapat benjolan",
    ],
    "Lainnya": [
        "Penyakit bermula dari daun bawah dekat tanah",
        "Penyakit bermula dari luka topping/midrib",
        "Penyakit cepat menyebar",
        "Muncul banyak cabang secara berlebihan",
        "Pertumbuhan tanaman terhambat (kerdil)",
    ],
}

# =========================
# LOADERS (robust)
# =========================
@st.cache_data
def load_kg(nodes_path: Path, edges_path: Path):
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    for c in ["node_id", "node_name", "node_class"]:
        if c not in nodes.columns:
            raise ValueError(f"Kolom '{c}' tidak ada di nodes CSV.")
        nodes[c] = nodes[c].astype(str).str.strip()

    for c in ["source", "target", "relation"]:
        if c not in edges.columns:
            raise ValueError(f"Kolom '{c}' tidak ada di edges CSV.")
        edges[c] = edges[c].astype(str).str.strip()

    return nodes, edges

@st.cache_resource
def load_embeddings(emb_path: Path):
    return np.load(emb_path)

@st.cache_data
def load_gejala_ui_map(map_path: Path):
    """
    Reader tahan koma di ui_label.
    Wajib header: ui_label,kg_name
    """
    if not map_path.exists():
        raise FileNotFoundError(f"File mapping tidak ditemukan: {map_path}")

    rows = []
    with open(map_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ui_label" not in reader.fieldnames or "kg_name" not in reader.fieldnames:
            raise ValueError("gejala_ui_map.csv harus punya header: ui_label,kg_name")

        for r in reader:
            ui = (r.get("ui_label") or "").strip()
            kg = (r.get("kg_name") or "").strip()
            if ui and kg:
                rows.append((ui, kg))

    ui2kg = dict(rows)
    kg2ui = {kg: ui for ui, kg in rows}
    return ui2kg, kg2ui

@st.cache_data
def build_maps(nodes: pd.DataFrame, edges: pd.DataFrame):
    name2id = dict(zip(nodes["node_name"], nodes["node_id"]))
    id2name = dict(zip(nodes["node_id"], nodes["node_name"]))
    id2idx  = dict(zip(nodes["node_id"], range(len(nodes))))

    disease_nodes = nodes[nodes["node_class"] == "Penyakit"].copy()
    disease_ids = disease_nodes["node_id"].tolist()
    disease_names = disease_nodes["node_name"].tolist()

    tp = edges[edges["relation"] == REL_DIAGNOSA].copy()
    disease2gejala = tp.groupby("target")["source"].apply(set).to_dict()

    return {
        "name2id": name2id,
        "id2name": id2name,
        "id2idx": id2idx,
        "disease_ids": disease_ids,
        "disease_names": disease_names,
        "disease2gejala": disease2gejala,
    }

@st.cache_data
def load_penyakit_ui_map(map_path: Path):
    if not map_path.exists():
        return {}  # kalau belum ada file, fallback pakai nama KG
    rows = {}
    with open(map_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "kg_name" not in reader.fieldnames or "ui_label" not in reader.fieldnames:
            raise ValueError("penyakit_ui_map.csv harus punya header: kg_name,ui_label")
        for r in reader:
            kg = (r.get("kg_name") or "").strip()
            ui = (r.get("ui_label") or "").strip()
            if kg and ui:
                rows[kg] = ui
    return rows

# =========================
# INFERENCE (ringkas, sesuai pipeline kamu)
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def build_df_gejala(disease2gejala: dict):
    df_gejala = {}
    for _, gset in disease2gejala.items():
        for g in gset:
            df_gejala[g] = df_gejala.get(g, 0) + 1
    N_disease = max(1, len(disease2gejala))
    return df_gejala, N_disease

def idf(gid, df_gejala, N_disease):
    df = df_gejala.get(gid, 1)
    return float(np.log((N_disease + 1e-9) / (df + 1e-9)))

def att_weight(gid, df_gejala, N_disease, gamma=2.0, umum_floor=0.03, w_clip=5.0):
    w = max(idf(gid, df_gejala, N_disease), 1e-6) ** gamma
    if df_gejala.get(gid, 1) > 1:
        w = max(w, umum_floor)
    return float(min(w, w_clip))

def agg_logsum(term_values, lam=1.0):
    t = np.maximum(np.array(term_values, dtype=float), 0.0)
    return float(np.sum(np.log1p(lam * t)))

def compute_confidence(topk, main_name, T_margin=0.25):
    if not topk:
        return 0.0
    s1 = topk[0]["score"]
    s2 = topk[1]["score"] if len(topk) > 1 else s1 - 1e-6
    C_margin = sigmoid((s1 - s2) / T_margin)
    if main_name == UNKNOWN_LABEL:
        return float((1 - C_margin) * 60)

    matched_count = len(topk[0]["matched_gejala"])
    total_count = max(1, int(topk[0]["total_gejala_penyakit"]))
    khas_match = int(topk[0]["matched_khas_count"])
    inputfit = float(topk[0].get("input_fit", 1.0))

    C_coverage = min(1.0, matched_count / total_count)
    C_specific = (khas_match + 1) / (matched_count + 2)
    C_specific = float(np.clip(C_specific, 0.0, 1.0))

    base = (0.20 * C_margin) + (0.45 * C_coverage) + (0.15 * C_specific) + (0.20 * inputfit)
    return float(min(base, 0.99) * 100)

def diagnose(gejala_input_kg_names, Z, maps, top_k=5,
             margin_delta=0.08, min_khas_match=1, min_match_total=1, min_weighted_evidence=0.15):
    name2id = maps["name2id"]
    id2name = maps["id2name"]
    id2idx  = maps["id2idx"]
    disease_ids = maps["disease_ids"]
    disease_names = maps["disease_names"]
    disease2gejala = maps["disease2gejala"]

    # mapping kg_name -> node_id
    gejala_ids = [name2id[g] for g in gejala_input_kg_names if g in name2id]
    if not gejala_ids:
        raise ValueError("Tidak ada gejala input valid (cek mapping UI→KG).")

    gejala_ids = [g for g in gejala_ids if g in id2idx]
    if not gejala_ids:
        raise ValueError("Gejala valid, tapi tidak ada embedding (cek nodes vs GCN_embeddings).")

    Gin = set(gejala_ids)
    df_gejala, N_disease = build_df_gejala(disease2gejala)

    emb_g = {g: Z[id2idx[g]] for g in gejala_ids}
    w_g   = {g: att_weight(g, df_gejala, N_disease) for g in gejala_ids}
    has_khas_input = any(df_gejala.get(g, 1) <= 1 for g in gejala_ids)

    results = []
    for did, dname in zip(disease_ids, disease_names):
        if did not in id2idx:
            continue

        Gd = disease2gejala.get(did, set())
        matched = list(Gin & Gd)
        mismatch = list(Gin - set(matched))
        inputfit = len(matched) / max(1, len(Gin))

        if not matched:
            results.append((dname, -1e9, 0, [], len(Gd), 0.0, float(inputfit), [id2name[g] for g in mismatch]))
            continue

        dvec = Z[id2idx[did]].reshape(1, -1)

        term_vals, evidence_sum, khas_count = [], 0.0, 0
        for g in matched:
            cos = float(cosine_similarity(emb_g[g].reshape(1, -1), dvec)[0][0])
            term = w_g[g] * cos
            term_vals.append(term)
            evidence_sum += max(term, 0.0)
            if df_gejala.get(g, 1) <= 1:
                khas_count += 1

        score = agg_logsum(term_vals, lam=1.0)
        results.append((
            dname, float(score), int(khas_count),
            [id2name[g] for g in matched],
            int(len(Gd)), float(evidence_sum), float(inputfit),
            [id2name[g] for g in mismatch]
        ))

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
            "mismatch_gejala": mismatch_names,
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

    conf_pct = compute_confidence(topk, main)
    return main, conf_pct, topk

# =========================
# PAGES (tanpa sidebar)
# =========================
PATOGEN_CONTENT = {
    "Virus": {
        "judul": "VIRUS — Penyebab Penyakit Tembakau",
        "isi": [
            "Virus tembakau merupakan parasit obligat non-seluler (umumnya RNA) yang menyerang sel tanaman.",
            "Gejala umum dapat berupa pola mosaik pada daun, daun keriting/cacat bentuk, pertumbuhan kerdil, hingga nekrosis.",
            "Penanganan berfokus pada pencegahan: varietas tahan, sanitasi alat/tangan, pengendalian vektor (kutu daun/thrips), serta pencabutan tanaman sakit."
        ]
    },
    "Bakteri": {
        "judul": "BAKTERI — Penyebab Penyakit Tembakau",
        "isi": [
            "Bakteri patogen dapat menginfeksi jaringan tanaman melalui luka atau stomata dan berkembang pada kondisi lembap.",
            "Gejala dapat berupa layu, busuk batang/akar, lendir pada pembuluh, serta perubahan warna jaringan.",
            "Penanganan umumnya meliputi sanitasi lahan, pengaturan drainase, rotasi tanaman, dan penggunaan bibit sehat."
        ]
    },
    "Jamur": {
        "judul": "JAMUR — Penyebab Penyakit Tembakau",
        "isi": [
            "Jamur berkembang pada kelembapan tinggi dan sering menimbulkan bercak daun, busuk, atau serbuk/spora pada permukaan daun.",
            "Gejala bisa berupa bercak cincin (kosentris), bercak frogeye, hingga jaringan daun rusak/berlubang.",
            "Pengendalian meliputi pemangkasan daun sakit, pengaturan jarak tanam/kelembapan, dan aplikasi fungisida sesuai anjuran."
        ]
    },
    "Nematoda": {
        "judul": "NEMATODA — Penyebab Penyakit Tembakau",
        "isi": [
            "Nematoda adalah cacing mikroskopis yang menyerang akar dan dapat menyebabkan pembengkakan/benjolan pada akar.",
            "Dampak lanjutan bisa berupa pertumbuhan terhambat, tanaman kerdil, dan layu pada siang hari.",
            "Pengendalian meliputi rotasi tanaman, solarisasi tanah, bahan organik, dan varietas tahan bila tersedia."
        ]
    }
}

def topbar():
    st.markdown(
        """
<div class="topbar">
  <div class="small">SISTEM PAKAR</div>
  <div class="title">DIAGNOSA PENYAKIT TEMBAKAU</div>
</div>
""",
        unsafe_allow_html=True,
    )

def page_patogen_detail(patogen_key: str):
    topbar()
    data = PATOGEN_CONTENT.get(patogen_key)
    if not data:
        st.error("Konten patogen tidak ditemukan.")
        return

    st.markdown("<div class='hero-wrap'></div>", unsafe_allow_html=True)
    st.markdown(f"## {data['judul']}")
    for p in data["isi"]:
        st.write(p)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="backbtn">', unsafe_allow_html=True)
    if st.button("← Kembali", key="back_home"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def page_home(nodes, edges, Z, maps, ui2kg, kg2ui, disease_ui):
    topbar()

    # HERO
    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    c1, c2 = st.columns([1.0, 1.15])

    with c1:
        st.markdown(
            """
<div class="hero-right">
  <h1></h1>
  <h2><span style="color:#C96F0A;font-weight:900;">\nDiagnosa penyakit tanaman tembakau berdasarkan gejalanya</span>
  untuk mendukung kesehatan tanaman dan produksi yang optimal.</h2>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        # if st.button("Mulai Diagnosis", type="primary"):
        #     # “geser” ke diagnosis: kita set flag, lalu rerun & tampilkan diagnosis di atas (fokus)
        #     st.session_state.focus_diag = True
        #     st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        # ganti dengan gambar lokal kamu kalau ada
        st.image(
            "https://jatengprov.go.id/wp-content/uploads/2024/11/WhatsApp-Image-2024-11-06-at-14.12.15-1170x500-1.jpeg",
            use_column_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hr-orange"></div>', unsafe_allow_html=True)

    # PATOGEN SECTION (button)
    st.markdown('<div class="section-title">MACAM-MACAM PATOGEN</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Penyebab Penyakit Tembakau</div>', unsafe_allow_html=True)

    # Baris tombol patogen
    st.markdown('<div class="patogen-row">', unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6, p7, p8 = st.columns(8)
    with p3:
        st.markdown('<div class="pillbtn">', unsafe_allow_html=True)
        if st.button("VIRUS", key="btn_patogen_virus", type="secondary"):
            st.session_state.page = "patogen"
            st.session_state.patogen = "Virus"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with p4:
        st.markdown('<div class="pillbtn">', unsafe_allow_html=True)
        if st.button("BAKTERI", key="btn_patogen_bakteri", type="secondary"):
            st.session_state.page = "patogen"
            st.session_state.patogen = "Bakteri"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with p5:
        st.markdown('<div class="pillbtn">', unsafe_allow_html=True)
        if st.button("JAMUR", key="btn_patogen_jamur", type="secondary"):
            st.session_state.page = "patogen"
            st.session_state.patogen = "Jamur"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with p6:
        st.markdown('<div class="pillbtn">', unsafe_allow_html=True)
        if st.button("NEMATODA", key="btn_patogen_nematoda", type="secondary"):
            st.session_state.page = "patogen"
            st.session_state.patogen = "Nematoda"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="hr-orange"></div>', unsafe_allow_html=True)

    # Jika tombol Mulai Diagnosis ditekan → tampilkan DIAGNOSIS lebih dulu (biar langsung terlihat)
    if st.session_state.get("focus_diag", False):
        st.session_state.focus_diag = False
        page_diagnosis(nodes, edges, Z, maps, ui2kg, kg2ui, disease_ui, focus=True)
    else:
        page_diagnosis(nodes, edges, Z, maps, ui2kg, kg2ui, disease_ui, focus=False)

def page_diagnosis(nodes, edges, Z, maps, ui2kg, kg2ui, disease_ui, focus=False):
    # DIAGNOSIS SECTION
    if focus:
        st.success("Silakan pilih gejala, lalu klik Diagnosa.")

    st.markdown('<div class="section-title">DIAGNOSA PENYAKIT TEMBAKAU</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pilih semua gejala yang sesuai dengan kondisi tanaman tembakau yang sakit!</div>', unsafe_allow_html=True)

    selected_ui = []

    # Daun
    st.markdown('<div class="cat-title">Daun</div><div class="cat-underline"></div>', unsafe_allow_html=True)
    for lab in GEJALA_UI["Daun"]:
        if st.checkbox(lab, key=f"ck_Daun_{lab}"):
            selected_ui.append(lab)
    # st.markdown('<div class="small-link">selengkapnya</div>', unsafe_allow_html=True)

    # Batang
    st.markdown('<div class="cat-title">Batang</div><div class="cat-underline"></div>', unsafe_allow_html=True)
    for lab in GEJALA_UI["Batang"]:
        if st.checkbox(lab, key=f"ck_Batang_{lab}"):
            selected_ui.append(lab)
    # st.markdown('<div class="small-link">selengkapnya</div>', unsafe_allow_html=True)

    # Akar
    st.markdown('<div class="cat-title">Akar</div><div class="cat-underline"></div>', unsafe_allow_html=True)
    for lab in GEJALA_UI["Akar"]:
        if st.checkbox(lab, key=f"ck_Akar_{lab}"):
            selected_ui.append(lab)

    # Lainnya
    st.markdown('<div class="cat-title">Lainnya</div><div class="cat-underline"></div>', unsafe_allow_html=True)
    for lab in GEJALA_UI["Lainnya"]:
        if st.checkbox(lab, key=f"ck_Lainnya_{lab}"):
            selected_ui.append(lab)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        run = st.button("Diagnosa", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_info:
        st.caption(f"<div style='text-align:right;'>Total gejala dipilih: <b>{len(selected_ui)}</b></div>",
        unsafe_allow_html=True)

    # Run diagnosis
    if run:
        if len(selected_ui) == 0:
            st.warning("Pilih minimal 1 gejala terlebih dahulu.")
            st.session_state.result_main = None
            st.session_state.result_conf = None
            st.session_state.result_topk = None
        else:
            # convert UI label -> KG name
            selected_kg = []
            missing = []
            for x in selected_ui:
                kg = ui2kg.get(x)
                if kg:
                    selected_kg.append(kg)
                else:
                    missing.append(x)

            if missing:
                st.error(
                    "Ada gejala UI yang belum ada di mapping `gejala_ui_map.csv`:\n- "
                    + "\n- ".join(missing)
                )
            else:
                try:
                    main, conf, topk = diagnose(selected_kg, Z, maps)
                    st.session_state.result_main = main
                    st.session_state.result_conf = conf
                    st.session_state.result_topk = topk
                except Exception as e:
                    st.error(f"Gagal melakukan diagnosis: {e}")

    # Result UI
    st.markdown('<div class="hr-orange"></div>', unsafe_allow_html=True)
    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">HASIL :</div>', unsafe_allow_html=True)

    main = st.session_state.get("result_main")
    conf = st.session_state.get("result_conf")
    topk = st.session_state.get("result_topk")

    # pakai nama UI untuk penyakit utama
    display_main = disease_ui.get(main, main)

    # tampilkan placeholder kalau belum ada hasil yang valid
    if (main is None) or (conf is None):
        st.markdown(
            '<div class="result-bar muted">Belum ada hasil diagnosis</div>',
            unsafe_allow_html=True
        )
    else:
        conf_val = float(conf)

        if main == UNKNOWN_LABEL:
            st.markdown(
                f'<div class="result-bar">Unknown_Penyakit — {conf_val:.1f}%</div>',
                unsafe_allow_html=True
            )
        else:
            # gunakan display_main (nama ramah pengguna)
            st.markdown(
                f'<div class="result-bar">{display_main} — {conf_val:.1f}%</div>',
                unsafe_allow_html=True
            )

        # Top-3 (abaikan kalau topk kosong)
        if isinstance(topk, list) and len(topk) > 0:
            st.markdown('<div class="topk-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="topk-title">Top-3 penyakit lain yang mungkin</div>', unsafe_allow_html=True)

            for i, r in enumerate(topk[:3], start=1):
                display_name = disease_ui.get(r.get("penyakit"), r.get("penyakit"))
                st.markdown(
                    f'<div class="topk-item"><span class="topk-rank">{i}.</span>{display_name}</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MAIN ROUTER (tanpa sidebar)
# =========================
def main():
    nodes, edges = load_kg(NODES_PATH, EDGES_PATH)
    Z = load_embeddings(EMB_PATH)
    maps = build_maps(nodes, edges)
    ui2kg, kg2ui = load_gejala_ui_map(MAP_PATH)
    disease_ui = load_penyakit_ui_map(DISEASE_MAP_PATH)

    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "patogen" not in st.session_state:
        st.session_state.patogen = "Virus"

    if st.session_state.page == "home":
        page_home(nodes, edges, Z, maps, ui2kg, kg2ui, disease_ui)
    elif st.session_state.page == "patogen":
        page_patogen_detail(st.session_state.patogen)
    else:
        st.session_state.page = "home"
        page_home(nodes, edges, Z, maps, ui2kg, kg2ui, disease_ui)

main()
