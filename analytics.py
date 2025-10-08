# streamlit_knowledge_graph_analysis_app.py
# Knowledge Graph Visualization + Centrality + Community Detection + Querying + File Saving
# Supports Directed / Undirected distinction and node highlighting for queries

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional semantic search
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# ---------------- Fix for NumPy 2.0 & NetworkX GraphML bug ----------------
if not hasattr(np, "float_"):
    np.float_ = np.float64
# -------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Knowledge Graph Analytics", initial_sidebar_state="expanded")

# ---------------- Helper Functions ----------------

def build_graph(df: pd.DataFrame, directed: bool = True) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in df.iterrows():
        s, r, o = str(row["Subject"]), str(row["Relation"]), str(row["Object"])
        G.add_edge(s, o, Relation=r)
    return G

def compute_centralities(G: nx.Graph):
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}
    return {"degree": deg, "betweenness": btw, "closeness": clo, "eigenvector": eig}

def detect_communities(G: nx.Graph, method: str = "greedy"):
    und = G.to_undirected()
    if method == "greedy":
        comms = nx.algorithms.community.greedy_modularity_communities(und)
    else:
        comms = list(nx.algorithms.community.label_propagation_communities(und))
    mapping = {}
    for i, comm in enumerate(comms):
        for n in comm:
            mapping[n] = i
    for n in G.nodes():
        mapping.setdefault(n, -1)
    return mapping

def create_pyvis_graph(G, node_attrs=None, highlight_nodes=None, height="1200px", directed=True):
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=directed,
    )
    net.force_atlas_2based()

    node_attrs = node_attrs or {}
    highlight_nodes = set(highlight_nodes or [])
    communities = sorted({a.get("community", -1) for a in node_attrs.values()})
    comm_to_hue = {c: (i * 360 // max(1, len(communities))) for i, c in enumerate(communities)}

    for n in G.nodes():
        a = node_attrs.get(n, {})
        color = None
        if a.get("community", -1) != -1:
            hue = comm_to_hue[a["community"]]
            color = f"hsl({hue},70%,50%)"
        size = 35 if n in highlight_nodes else 15
        title = "<br>".join([f"{k}: {v}" for k, v in a.items()])
        net.add_node(n, label=n, title=title, color=color, size=size)

    for u, v, d in G.edges(data=True):
        relation = d.get("Relation", "")
        net.add_edge(u, v, title=relation, label=relation, arrows="to" if directed else "")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp.name)
    return tmp.name

def make_centrality_df(cent, comm_map):
    nodes = list(cent["degree"].keys())
    df = pd.DataFrame({
        "Node": nodes,
        "Degree": [cent["degree"].get(n, 0) for n in nodes],
        "Betweenness": [cent["betweenness"].get(n, 0) for n in nodes],
        "Closeness": [cent["closeness"].get(n, 0) for n in nodes],
        "Eigenvector": [cent["eigenvector"].get(n, 0) for n in nodes],
        "Community": [comm_map.get(n, -1) for n in nodes],
    })
    return df.sort_values(by="Degree", ascending=False)

def ensure_output_dir():
    outdir = "kg_output"
    os.makedirs(outdir, exist_ok=True)
    return outdir

@st.cache_resource
def load_sbert_model():
    if SBERT_AVAILABLE:
        return SentenceTransformer("all-MiniLM-L6-v2")
    return None

# ---------------- Streamlit UI ----------------
st.title("Knowledge Graph — Analytics & Querying Dashboard")

with st.sidebar:
    st.header("Dataset Selection")
    use_uploaded = st.toggle("Upload different CSV (instead of triples_cleaned.csv)")
    if use_uploaded:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {uploaded.name}")
        else:
            st.stop()
    else:
        if not os.path.exists("triples_cleaned.csv"):
            st.error("triples_cleaned.csv not found!")
            st.stop()
        df = pd.read_csv("triples_cleaned.csv")
        st.info("Using default triples_cleaned.csv")

if not {"Subject", "Relation", "Object"}.issubset(df.columns):
    st.error("CSV must have Subject, Relation, Object columns")
    st.stop()

with st.sidebar:
    st.header("Graph Options")
    directed = st.radio("Graph Type", ["Directed", "Undirected"]) == "Directed"
    algo = st.selectbox("Community Detection", ["greedy", "label_prop"])
    highlight_n = st.slider("Highlight top N centrality nodes", 0, 50, 5)
    metric = st.selectbox("Metric", ["degree", "betweenness", "closeness", "eigenvector"])
    height = st.slider("Graph Height (px)", 600, 2000, 1200)
    search = st.text_input("Search Node (contains)")
    rels = sorted(df["Relation"].unique().tolist())
    selected_rels = st.multiselect("Filter by Relation", rels, default=rels)

# Build Graph
G = build_graph(df, directed=directed)

# Filter relations
if set(selected_rels) != set(rels):
    G = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d.get("Relation") in selected_rels]).copy()

# Compute metrics
centralities = compute_centralities(G)
communities = detect_communities(G, algo)
node_attrs = {
    n: {
        "degree": centralities["degree"].get(n, 0),
        "betweenness": centralities["betweenness"].get(n, 0),
        "closeness": centralities["closeness"].get(n, 0),
        "eigenvector": centralities["eigenvector"].get(n, 0),
        "community": communities.get(n, -1),
    }
    for n in G.nodes()
}

# Highlight top centrality nodes
highlights = sorted(G.nodes(), key=lambda n: node_attrs[n][metric], reverse=True)[:highlight_n]

# ---------------- Querying / Search ----------------
st.sidebar.header("Query / Search Options")
start_node = st.sidebar.text_input("Shortest path: Start node")
end_node = st.sidebar.text_input("Shortest path: End node")
triple_query = st.sidebar.text_input("Search triples (subject/object/relation)")

# Always show semantic search box
semantic_query = st.sidebar.text_input("Semantic search (NL query)")
sbert_model = load_sbert_model() if SBERT_AVAILABLE else None
semantic_results = pd.DataFrame()
semantic_nodes = []

if sbert_model and semantic_query:
    texts = [f"{s} {r} {o}" for s,r,o in zip(df["Subject"], df["Relation"], df["Object"])]
    embeddings = sbert_model.encode(texts, convert_to_tensor=True)
    q_emb = sbert_model.encode(semantic_query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0]
    topk = torch.topk(scores, k=min(5, len(scores)))
    top_indices = topk.indices.tolist()  
    semantic_results = pd.DataFrame([texts[i] for i in top_indices], columns=["Triple"])
    for i in top_indices:
        s, r, o = df.iloc[int(i)]["Subject"], df.iloc[int(i)]["Relation"], df.iloc[int(i)]["Object"]
        semantic_nodes.extend([s, o])
elif semantic_query and not SBERT_AVAILABLE:
    st.warning("Semantic search is not available. Install sentence-transformers in requirements.txt.")

# Compute shortest path
shortest_path = None
shortest_nodes = []
if start_node in G.nodes() and end_node in G.nodes():
    try:
        shortest_path = nx.shortest_path(G, source=start_node, target=end_node)
        shortest_nodes = shortest_path
    except nx.NetworkXNoPath:
        shortest_path = []

matched_triples = pd.DataFrame()
if triple_query:
    matched_triples = df[
        df.apply(lambda r: triple_query.lower() in str(r["Subject"]).lower()
                            or triple_query.lower() in str(r["Object"]).lower()
                            or triple_query.lower() in str(r["Relation"]).lower(), axis=1)
    ]

search_results = [n for n in G.nodes() if search.lower() in n.lower()] if search else []

# Combine highlights (centrality + search + semantic + shortest path)
all_highlights = set(highlights + search_results + shortest_nodes + semantic_nodes)
html_path = create_pyvis_graph(G, node_attrs=node_attrs, highlight_nodes=all_highlights, height=f"{height}px", directed=directed)

# ---------------- Visualization ----------------
st.subheader("Interactive Knowledge Graph")
with open(html_path, encoding="utf-8") as f:
    html = f.read()
components.html(html, height=height, scrolling=True)

# Centrality metrics visualization
df_metrics = make_centrality_df(centralities, communities)

st.markdown("## Top Centrality Nodes")
cols = st.columns(4)
metrics = ["Degree", "Betweenness", "Closeness", "Eigenvector"]
for i, m in enumerate(metrics):
    with cols[i]:
        st.metric(label=f"Top {m} Node", value=df_metrics.iloc[0]["Node"])
        top_nodes = df_metrics.nlargest(10, m)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=m, y="Node", data=top_nodes, palette="viridis", ax=ax)
        ax.set_xlabel(m)
        ax.set_ylabel("Node")
        st.pyplot(fig)

# Show shortest path below graph
if shortest_path is not None:
    st.subheader(f"Shortest Path: {start_node} → {end_node}")
    if shortest_path:
        st.write(" → ".join(shortest_path))
    else:
        st.warning("No path found")

if not matched_triples.empty:
    st.subheader(f"Matched Triples for '{triple_query}'")
    st.dataframe(matched_triples)

if not semantic_results.empty:
    st.subheader(f"Semantic Search Results for '{semantic_query}'")
    st.dataframe(semantic_results)

# ---------------- Show Detected Communities ----------------
st.markdown("## Detected Communities")

# Convert to DataFrame
comm_df = pd.DataFrame(list(communities.items()), columns=["Node", "Community"])

# Compute community sizes
comm_summary = comm_df["Community"].value_counts().reset_index()
comm_summary.columns = ["Community", "Size"]

# Add example nodes (top 5 nodes by degree in each community)
example_nodes = []
for comm_id in comm_summary["Community"]:
    nodes_in_comm = comm_df[comm_df["Community"] == comm_id]["Node"].tolist()
    example_nodes.append(", ".join(nodes_in_comm[:5]))  # top 5 nodes
comm_summary["Example Nodes"] = example_nodes

# Display table
st.dataframe(comm_summary)

# ---------------- Save outputs ----------------
output_dir = ensure_output_dir()
central_csv = os.path.join(output_dir, "node_metrics.csv")
comm_csv = os.path.join(output_dir, "communities.csv")
graphml_file = os.path.join(output_dir, "knowledge_graph.graphml")
png_path = os.path.join(output_dir, "knowledge_graph.png")

df_metrics.to_csv(central_csv, index=False)
comm_df.to_csv(comm_csv, index=False)

try:
    nx.write_graphml(G, graphml_file)
except Exception as e:
    st.warning(f"Could not save GraphML: {e}")

try:
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12,10))
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_color=[communities[n] for n in G.nodes()],
        cmap="tab10",
        node_size=300,
        font_size=8,
        arrows=directed
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
except Exception as e:
    st.warning(f"Could not save PNG: {e}")

try:
    os.remove(html_path)
except:
    pass
