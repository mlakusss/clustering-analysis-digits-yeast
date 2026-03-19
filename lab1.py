import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE

# =========================
# ФУНКЦИИ
# =========================

def evaluate_clustering(X, y_true, model):
    start = time.time()
    labels = model.fit_predict(X)
    elapsed = time.time() - start

    ari = adjusted_rand_score(y_true, labels)
    ami = adjusted_mutual_info_score(y_true, labels)

    return labels, elapsed, ari, ami


def elbow_method(X, k_range):
    inertias = []
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)
    return inertias


def silhouette_method(X, k_range):
    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    return scores


# =========================
# ЧАСТЬ 1: DIGITS
# =========================

print("="*60)
print("ЧАСТЬ 1. DIGITS")
print("="*60)

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

print(f"Размерность: {X_scaled.shape}")
print(f"Классов: {len(np.unique(y_digits))}")

# PCA (для анализа)
pca_full = PCA()
pca_full.fit(X_scaled)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cumsum >= 0.9) + 1
print(f"Компонент для 90% дисперсии: {n_components_90}")

# PCA 2D
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# --- КЛАСТЕРИЗАЦИЯ ---
models = {
    "k-means++": KMeans(n_clusters=10, init='k-means++', random_state=42),
    "random": KMeans(n_clusters=10, init='random', random_state=42),
    "PCA init": KMeans(n_clusters=10, init=pca_full.components_[:10], n_init=1),
    "Agglomerative": AgglomerativeClustering(n_clusters=10)
}

print("\n--- СРАВНЕНИЕ ---")
print(f"{'Метод':<20}{'Время':<10}{'ARI':<10}{'AMI'}")

results = {}

for name, model in models.items():
    labels, t, ari, ami = evaluate_clustering(X_scaled, y_digits, model)
    results[name] = labels
    print(f"{name:<20}{t:<10.4f}{ari:<10.4f}{ami:.4f}")

# --- ВИЗУАЛИЗАЦИЯ PCA vs t-SNE ---
labels_best = results["Agglomerative"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_best, cmap='tab10', s=15)
axes[0].set_title(f"PCA (2D, {pca_2d.explained_variance_ratio_.sum()*100:.1f}%)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_best, cmap='tab10', s=15)
axes[1].set_title("t-SNE (чёткие кластеры)")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.show()

# =========================
# ЧАСТЬ 2: YEAST
# =========================

print("\n" + "="*60)
print("ЧАСТЬ 2. YEAST")
print("="*60)

# Загрузка yeast (локальный файл!)
import pandas as pd

df = pd.read_csv("yeast.data", delim_whitespace=True, header=None)

X_yeast = df.iloc[:, 1:-1].values
y_yeast = df.iloc[:, -1].astype('category').cat.codes.values

scaler = StandardScaler()
X_yeast_scaled = scaler.fit_transform(X_yeast)

print(f"Размерность: {X_yeast.shape}")

# PCA
pca = PCA(n_components=2)
X_yeast_pca = pca.fit_transform(X_yeast_scaled)

print("\nОбъясненная дисперсия:")
print(pca.explained_variance_ratio_)
print("Сумма:", pca.explained_variance_ratio_.sum())

# --- ПОДБОР k ---
k_range = range(2, 15)

inertias = elbow_method(X_yeast_scaled, k_range)
sil_scores = silhouette_method(X_yeast_scaled, k_range)

best_k = k_range[np.argmax(sil_scores)]
print(f"\nЛучшее k по силуэту: {best_k}")

# --- ГРАФИКИ ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, marker='o')
axes[0].set_title("Elbow method (Yeast)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(k_range, sil_scores, marker='o')
axes[1].set_title("Silhouette method (Yeast)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.show()

# --- КЛАСТЕРИЗАЦИЯ ---
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
labels_yeast = kmeans.fit_predict(X_yeast_scaled)

ari = adjusted_rand_score(y_yeast, labels_yeast)
ami = adjusted_mutual_info_score(y_yeast, labels_yeast)
sil = silhouette_score(X_yeast_scaled, labels_yeast)

print("\nМетрики Yeast:")
print(f"ARI: {ari:.4f}")
print(f"AMI: {ami:.4f}")
print(f"Silhouette: {sil:.4f}")

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(7, 5))
plt.scatter(X_yeast_pca[:, 0], X_yeast_pca[:, 1], c=labels_yeast, cmap='tab10', s=20)

centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='red', s=200, marker='X', label='Центры')

plt.title("Yeast (PCA, слабая разделимость данных)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# ИТОГ
# =========================

print("\nИТОГ:")
print("- PCA плохо разделяет кластеры из-за потери информации")
print("- t-SNE показывает реальную структуру данных")
print("- KMeans хорошо работает для Digits, но плохо для Yeast")
print("- Agglomerative дал лучший результат на Digits")