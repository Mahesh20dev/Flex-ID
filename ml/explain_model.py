import shap
import numpy as np
import pandas as pd
import pickle
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import create_dnn_model

# ================================
# FEDERATED SHAP EXPLAINER
# ================================

class FederatedSHAP:
    def __init__(self, weights_path, data_path, le_path, num_clients=4):
        self.weights_path = weights_path
        self.data_path = data_path
        self.le_path = le_path
        self.num_clients = num_clients

        self.model = self._load_model()
        self.feature_names = self._load_feature_names()

    # ------------------------------
    def _load_model(self):
        print(f"[XAI] Loading model from {self.weights_path}")

        with open(self.le_path, "rb") as f:
            le = pickle.load(f)

        df = pd.read_csv(self.data_path)
        target_col = [c for c in df.columns if c.lower() in ["label", "class", "attack_cat"]][0]
        X_dim = df.drop(columns=[target_col]).shape[1]

        model = create_dnn_model(X_dim, len(le.classes_))

        with open(self.weights_path, "rb") as f:
            weights = pickle.load(f)
        model.set_weights(weights)

        self.le = le
        return model

    # ------------------------------
    def _load_feature_names(self):
        df = pd.read_csv(self.data_path, nrows=1)
        return [c for c in df.columns if c.lower() not in ["label", "class", "attack_cat"]]

    # ------------------------------
    def load_client_data(self, cid):
        with open(f"client_partition_{cid}.pkl", "rb") as f:
            (_, _), (X_test, y_test) = pickle.load(f)
        return X_test.astype(np.float32), y_test

    # ------------------------------
    def explain_client(self, cid, bg_size=50, explain_size=10, prefix=""):
        print(f"[Client {cid}] Running SHAP explanation")

        X_test, _ = self.load_client_data(cid)

        bg = X_test[np.random.choice(len(X_test), bg_size, replace=False)]
        samples = X_test[:explain_size]

        def predict_fn(x):
            return self.model.predict(x, verbose=0)

        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_values = explainer.shap_values(samples)

        # Use attack class
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        # Save local plot
        os.makedirs("results", exist_ok=True)
        out_path = f"results/{prefix}_client_{cid}_shap.png"

        plt.figure()
        shap.summary_plot(
            shap_vals,
            samples,
            feature_names=self.feature_names,
            show=False
        )
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"[Client {cid}] Saved local SHAP → {out_path}")

        # Return importance vector only (privacy preserving)
        return np.mean(np.abs(shap_vals), axis=0)

    # ------------------------------
    def aggregate_global(self, client_vectors, prefix=""):
        print("\n[Server] Aggregating federated SHAP values")

        global_imp = np.mean(client_vectors, axis=0)
        global_imp = 100 * global_imp / np.max(global_imp)

        idx = np.argsort(global_imp)[::-1][:15]
        labels = [self.feature_names[i] for i in idx]
        values = global_imp[idx]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(values)), values)
        plt.yticks(range(len(values)), labels)
        plt.xlabel("Aggregated SHAP Importance (%)")
        plt.title("Federated Global Feature Importance")
        plt.gca().invert_yaxis()

        out_path = f"results/{prefix}_global_shap.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"[Server] Saved global SHAP → {out_path}")

# ================================
# RUN DEMO (FedAvg / FedProx)
# ================================

def run():
    models = {
        "FedAvg": "results/fedavgeachround/round-10-weights.pkl",
        "FedProx": "results/fedproxeachround/round-10-weights.pkl"
    }

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"[WARN] {name} weights not found, skipping.")
            continue

        print(f"\n==== Explaining {name} ====")
        explainer = FederatedSHAP(
            weights_path=path,
            data_path="data/processed_data.csv",
            le_path="data/label_encoder.pkl"
        )

        client_updates = []
        for cid in range(4):
            vec = explainer.explain_client(cid, prefix=name)
            client_updates.append(vec)

        explainer.aggregate_global(client_updates, prefix=name)

if __name__ == "__main__":
    run()
