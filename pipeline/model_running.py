from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import pathlib
import ast

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pickle

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def xg_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    plt.barh(feature_cols, model.feature_importances_)
    plt.xlabel("Importance")
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Fraud']).plot()
    plt.title("Fraud Detection Confusion Matrix")
    plt.tight_layout()
    # plt.savefig("confusion_matrix.png", dpi=100)
    plt.show()

    with open(DATA_DIR / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return accuracy, precision, recall, f1, model


def pca_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):

    # Standardize features first (PCA needs it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Fit PCA to 3D
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                          c=y_train, cmap='coolwarm', alpha=0.6, s=10)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA: Fraud vs Legitimate")
    plt.colorbar(scatter, label='Fraud')
    plt.tight_layout()
    plt.savefig("pca_visualization.png", dpi=100)
    plt.show()

    return pca, X_train_pca


def kclustering_model(pca, X_train_pca, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    # Fit K-means on ORIGINAL features (full dimensionality)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train)  # Original features, not PCA
    val_clusters = kmeans.predict(X_val)
    test_clusters = kmeans.predict(X_test)

    # But VISUALIZE using PCA (for interpretability)
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                          c=train_clusters, cmap='viridis', alpha=0.6, s=20)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("K-Means Clusters (Fitted on Full Features, Shown in PCA)")
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=100)
    plt.show()

    # Cluster composition
    for i in range(4):
        fraud_count = (y_train[train_clusters == i] == 1).sum()
        legit_count = (y_train[train_clusters == i] == 0).sum()
        print(f"Cluster {i}: {fraud_count} fraud, {legit_count} legit")

def main():
    # Read and convert string back to dict
    train_df = pd.read_csv(DATA_DIR / "label_train.csv")
    train_df["Features"] = train_df["Features"].apply(ast.literal_eval)
    train_features = pd.json_normalize(train_df["Features"])
    train_df = pd.concat([train_df, train_features], axis=1)

    val_df = pd.read_csv(DATA_DIR / "label_val.csv")
    val_df["Features"] = val_df["Features"].apply(ast.literal_eval)
    val_features = pd.json_normalize(val_df["Features"])
    val_df = pd.concat([val_df, val_features], axis=1)

    test_df = pd.read_csv(DATA_DIR / "label_test.csv")
    test_df["Features"] = test_df["Features"].apply(ast.literal_eval)
    test_features = pd.json_normalize(test_df["Features"])
    test_df = pd.concat([test_df, test_features], axis=1)

    # Extract numeric features (drop text_combined, label, etc.)
    feature_cols = [col for col in train_features.columns]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    data = [X_train, y_train, X_val, y_val, X_test, y_test, feature_cols]

    xg_model(*data)
    pca, X_train_pca = pca_model(*data)
    kclustering_model(pca, X_train_pca, *data)

if __name__ == "__main__":
    main()