import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# Load cleaned triples
df = pd.read_csv("triples_cleaned.csv")  # Columns: Subject, Relation, Object, Category, Filename

# Create graph and node → category mapping
G = nx.DiGraph()
node_category = {}

for idx, row in df.iterrows():
    head, rel, tail, cat = row['Subject'], row['Relation'], row['Object'], row['Category']
    
    # Add nodes
    G.add_node(head, label=head)
    G.add_node(tail, label=tail)
    
    # Add edge
    G.add_edge(head, tail, label=rel)
    
    # Map node → category
    node_category[head] = cat
    node_category[tail] = cat

nodes = list(G.nodes)

# Load sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
node_embeddings = model.encode(nodes, convert_to_tensor=True)

# Streamlit UI
st.set_page_config(page_title="Semantic Knowledge Graph", layout="wide")
st.title("Semantic Knowledge Graph")

# Show extracted triples
st.subheader("Extracted Triples")
st.dataframe(df[["Subject", "Relation", "Object", "Category"]])

# Search box
query = st.text_input("Enter your search query:")

if st.button("Search") and query:
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]
    
    # Get top matches
    top_k = 5
    results = sorted(zip(nodes, cosine_scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    st.write("### Top matches:")
    for node, score in results:
        category = node_category.get(node, "Unknown")
        st.write(f"{node} (Category: {category}) - Score: {score:.4f}")
    
    # PyVis
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    
    # Add nodes
    for node in nodes:
        if node in [r[0] for r in results]:
            net.add_node(node, label=f"{node}\n({node_category[node]})", color='red', size=30)
        else:
            net.add_node(node, label=f"{node}\n({node_category[node]})", size=15)
    
    # Add edges
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], label=edge[2]['label'])
    
    # Save & display
    net.save_graph('semantic_graph.html')
    HtmlFile = open("semantic_graph.html", 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=600)
