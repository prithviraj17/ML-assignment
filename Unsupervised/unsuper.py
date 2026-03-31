import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Title
st.title("🧠 Customer Segmentation (Unsupervised Learning)")

st.write("Group customers into segments using clustering algorithms.")

# Load dataset
data = pd.read_csv("customer_segmentation.csv")

st.subheader("📂 Dataset Preview")
st.dataframe(data)

# Features
X = data[['age', 'income', 'spending_score', 'purchase_frequency']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- K-MEANS ----------------
k = st.slider("Select number of clusters (K-Means)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------- HIERARCHICAL ----------------
hierarchical = AgglomerativeClustering(n_clusters=k)
data['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)

# ---------------- GRAPH 1 ----------------
st.subheader("📈 Graph 1: K-Means Clustering")

fig1, ax1 = plt.subplots()
ax1.scatter(data['income'], data['spending_score'], c=data['KMeans_Cluster'])
ax1.set_xlabel("Income")
ax1.set_ylabel("Spending Score")
ax1.set_title("K-Means Clustering Result")
st.pyplot(fig1)

# ---------------- GRAPH 2 ----------------
st.subheader("📈 Graph 2: Hierarchical Clustering")

fig2, ax2 = plt.subplots()
ax2.scatter(data['income'], data['spending_score'], c=data['Hierarchical_Cluster'])
ax2.set_xlabel("Income")
ax2.set_ylabel("Spending Score")
ax2.set_title("Hierarchical Clustering Result")
st.pyplot(fig2)

# ---------------- GRAPH 3 ----------------
st.subheader("🌳 Graph 3: Dendrogram (Hierarchical)")

linked = linkage(X_scaled, method='ward')

fig3, ax3 = plt.subplots(figsize=(8, 4))
dendrogram(linked, ax=ax3)
ax3.set_title("Dendrogram")
st.pyplot(fig3)

# Show clustered data
st.subheader("📊 Clustered Data")
st.dataframe(data)