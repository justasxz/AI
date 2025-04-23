import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Apply PCA (we’ll keep all 4 components for the variance plot)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_std)

# 1) Original feature space (first two features)
plt.figure()
for target in np.unique(y):
    plt.scatter(
        X_std[y == target, 0],
        X_std[y == target, 1],
        label=target_names[target]
    )
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Original standardized features")
plt.legend()
plt.show()

# 2) PCA projection to 2D
plt.figure()
for target in np.unique(y):
    plt.scatter(
        X_pca[y == target, 0],
        X_pca[y == target, 1],
        label=target_names[target]
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA projection (2 components)")
plt.legend()
plt.show()

# 3) Explained variance ratio
plt.figure()
components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, pca.explained_variance_ratio_)
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("Variance explained by each component")
plt.xticks(components)
plt.show()

# --------------------------------------------------
# ---- New: train RandomForest on full vs. PCA data
# --------------------------------------------------

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.3, random_state=42
)

# 4) Classification WITHOUT PCA
rf_full = RandomForestClassifier(random_state=42)
rf_full.fit(X_train, y_train)
y_pred_full = rf_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)
print(f"RandomForest accuracy without PCA: {acc_full:.3f}")

# 5) Classification WITH PCA (2 components)
n_pcs = 2
pca2 = PCA(n_components=n_pcs)
X_train_pca = pca2.fit_transform(X_train)
X_test_pca  = pca2.transform(X_test)

rf_pca = RandomForestClassifier(random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"RandomForest accuracy with {n_pcs} PCs: {acc_pca:.3f}")

# Optional: decision boundary plot in PC1–PC2 space
# (only makes sense for the PCA model)
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
Z = rf_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
for target in np.unique(y):
    plt.scatter(
        X_train_pca[y_train == target, 0],
        X_train_pca[y_train == target, 1],
        label=target_names[target],
        edgecolor='k'
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("RF decision boundary on first 2 PCs (train set)")
plt.legend()
plt.show()
