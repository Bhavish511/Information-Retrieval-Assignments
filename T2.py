import os
import numpy as np
import math
import re
import time
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def load_tfidf_results(filename):
    doc_tfidf = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            doc_line = lines[i].strip()
            if doc_line.startswith('Document:'):
                doc_id = int(doc_line.split(':')[1].strip())
                tfidf_line = lines[i+1].strip()
                tfidf_vector = [float(x) for x in tfidf_line.split(':')[1].split(',')]
                doc_tfidf[doc_id] = tfidf_vector
                i += 2
            else:
                i += 1
    return doc_tfidf

def euclidean_distance(vec1, vec2):
    return np.around(np.sqrt(np.sum(np.square(vec1 - vec2))), 3)

def calculate_centroid(cluster):
    l = len(cluster)
    if l == 0:
        return np.zeros(len(next(iter(doc_tfidf.values()))))
    n_centroid = np.zeros(len(next(iter(doc_tfidf.values()))))
    for i in range(l):
        n_centroid += doc_tfidf[cluster[i]]
    n_centroid *= (1 / l)
    return n_centroid


def initialize_centroids(data, k):
    np.random.seed(int(time.time()))
    indices = np.random.choice(len(data), size=k, replace=False)
    centroids = np.array([doc_tfidf[data[index]] for index in indices])
    return centroids, [data[index] for index in indices]

def build_clusters(docs, centroids, k=5):
    clusters = {i: [] for i in range(k)}
    for doc in docs:
        distances = [euclidean_distance(doc_tfidf[doc], centroid) for centroid in centroids]
        min_dis_cen = np.argmin(distances)
        clusters[min_dis_cen].append(doc)
    return clusters

def update_centroids(clusters, centroids):
    for c_id, c_docs in clusters.items():
        centroids[c_id] = calculate_centroid(c_docs)
    return centroids

def calculate_rss(clusters, centroids):
    rss = 0
    for cluster_id in range(len(clusters)):
        for doc in clusters[cluster_id]:
            rss += np.around(np.square(euclidean_distance(doc_tfidf[doc], centroids[cluster_id])), 3)
    return rss

def kmeans(docs, k):
    clusters = []
    counter = 0
    centroids, seed_docs = initialize_centroids(docs, k)
    rss = float("inf")
    new_rss = 1000000
    while new_rss > 0 and new_rss < rss:
        counter += 1
        rss = new_rss
        clusters = build_clusters(docs, centroids, k)
        centroids = update_centroids(clusters, centroids)
        new_rss = calculate_rss(clusters, centroids)
    return clusters, seed_docs

def calculate_purity(golden_clusters, test_clusters):
    total_instances = sum(len(cluster) for cluster in test_clusters.values())
    total_correct = 0
    
    for test_cluster in test_clusters.values():
        max_common = 0
        for golden_cluster in golden_clusters.values():
            common = len(set(test_cluster).intersection(golden_cluster))
            if common > max_common:
                max_common = common
        total_correct += max_common
        
    purity = total_correct / total_instances
    return total_correct, purity

def calculate_rand_index(golden_clusters, test_clusters):
    golden_labels = []
    test_labels = []
    
    for golden_cluster_id, golden_cluster in golden_clusters.items():
        golden_labels.extend([golden_cluster_id] * len(golden_cluster))
        
    for test_cluster_id, test_cluster in test_clusters.items():
        test_labels.extend([test_cluster_id] * len(test_cluster))
    
    rand_index = adjusted_rand_score(golden_labels, test_labels)
    return rand_index

def calculate_silhouette_score(test_clusters, data):
    labels = []
    for cluster_id, cluster in test_clusters.items():
        labels.extend([cluster_id] * len(cluster))
    
    labels = np.array(labels)
    data = data.reshape(-1, 1)
    silhouette_avg = silhouette_score(data, labels)
    return silhouette_avg

def plot_silhouette_graph(k_values, silhouette_scores):
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.show()

def evaluate_kmeans(k):
    clusters, seed_docs = kmeans(docs, k)

    # Golden clusters
    golden_clusters = {
        0: [1, 2, 3, 7],
        1: [8, 9, 11],
        2: [12, 13, 14, 15, 16],
        3: [17, 18, 21],
        4: [22, 23, 24, 25, 26]
    }

    # Calculate purity
    total_correct, purity = calculate_purity(golden_clusters, clusters)
    print("Total Correct Assignments:", total_correct)
    print("Purity:", purity)

    # Calculate Rand index
    rand_index = calculate_rand_index(golden_clusters, clusters)
    print("Rand Index:", rand_index)

    # Calculate silhouette score
    silhouette_avg = calculate_silhouette_score(clusters, docs)
    print("Silhouette Score:", silhouette_avg)

    # Display clusters with documents
    cluster_output = ""
    for cluster_id, cluster_docs in clusters.items():
        cluster_output += f"Cluster {cluster_id + 1}: {cluster_docs}\n"

    output_text.delete('1.0', tk.END)
    output_text.insert(tk.END, f"Purity: {purity}\nRand Index: {rand_index}\nSilhouette Score: {silhouette_avg}\nTotal Correct Assignments: {total_correct}\nSeed Documents: {seed_docs}\n\n{cluster_output}")

    # Plot silhouette graph
    k_values = range(2, 11)
    silhouette_scores = []

    for k in k_values:
        clusters, _ = kmeans(docs, k=k)
        silhouette_avg = calculate_silhouette_score(clusters, docs)
        silhouette_scores.append(silhouette_avg)

    plot_silhouette_graph(k_values, silhouette_scores)

def main():
    global doc_tfidf, docs
    doc_tfidf = load_tfidf_results("tfidf_results.txt")
    ResearchPapers = [1, 2, 3, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
    docs = np.array(ResearchPapers)

    root = tk.Tk()
    root.title("K-Means Clustering Evaluation")

    ttk.Label(root, text="Enter the value of K:").pack()
    k_entry = ttk.Entry(root)
    k_entry.pack()

    ttk.Button(root, text="Evaluate", command=lambda: evaluate_kmeans(int(k_entry.get()))).pack()

    global output_text
    output_text = tk.Text(root)
    output_text.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
