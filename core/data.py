import pandas as pd

REL_DIAGNOSA = "terdiagnosaPenyakit"

def load_kg(nodes_path: str, edges_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    # Kolom wajib
    for c in ["node_id", "node_name", "node_class"]:
        if c not in nodes.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan di nodes CSV.")
        nodes[c] = nodes[c].astype(str).str.strip()

    for c in ["source", "target", "relation"]:
        if c not in edges.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan di edges CSV.")
        edges[c] = edges[c].astype(str).str.strip()

    return nodes, edges


def build_maps(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    name2id = dict(zip(nodes["node_name"], nodes["node_id"]))
    id2name = dict(zip(nodes["node_id"], nodes["node_name"]))
    id2idx  = dict(zip(nodes["node_id"], range(len(nodes))))

    # node penyakit
    disease_nodes = nodes[nodes["node_class"] == "Penyakit"].copy()
    disease_ids = disease_nodes["node_id"].tolist()
    disease_names = disease_nodes["node_name"].tolist()

    # relasi diagnosis: Gejala -> Penyakit
    tp = edges[edges["relation"] == REL_DIAGNOSA].copy()
    disease2gejala = tp.groupby("target")["source"].apply(set).to_dict()

    # patogen (opsional, untuk halaman informasi)
    patogen_nodes = nodes[nodes["node_class"].str.lower() == "patogen"].copy()
    patogen_names = sorted(patogen_nodes["node_name"].tolist())

    return {
        "name2id": name2id,
        "id2name": id2name,
        "id2idx": id2idx,
        "disease_ids": disease_ids,
        "disease_names": disease_names,
        "disease2gejala": disease2gejala,
        "patogen_names": patogen_names,
    }
