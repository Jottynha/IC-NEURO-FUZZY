"""
Estudo sistemático de hiperparâmetros do ANFIS simplificado.
Testa variações univariadas e salva resumos em `resultados[2]/anfis_hyperparameter_study/`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from experiment_utils import DATASETS, evaluate_model, load_dataset

# Valores base conservadores para evitar explosão de regras
BASE_PARAMS = {
    "n_membership_functions": 2,
    "learning_rate": 0.01,
    "n_epochs": 10,
    "pca_components": 4,
    "max_train_samples": 300,
}

# Seeds reduzidas para estudo rápido
STUDY_SEEDS = list(range(1, 6))

# Variações a testar (um parâmetro por vez)
PARAMETER_STUDIES = {
    "n_membership_functions": [2, 3, 4],
    "learning_rate": [0.001, 0.01, 0.05],
    "n_epochs": [10, 25, 50],
    "pca_components": [2, 4, 6, 8],
    "max_train_samples": [150, 300, 500],
}


class ANFISClassifier:
    def __init__(self, n_membership_functions: int = 2, learning_rate: float = 0.01, n_epochs: int = 10, random_state: int = 42):
        self.n_mf = n_membership_functions
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.mf_params = None
        self.weights = None
        self.classes_ = None

    def _gaussian_mf(self, x: float, mean: float, sigma: float) -> float:
        sigma = max(float(sigma), 1e-6)
        return float(np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)))

    def _init_parameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y_train)
        self.mf_params = {}
        for feature_idx in range(X_train.shape[1]):
            values = X_train[:, feature_idx]
            min_val = float(values.min())
            max_val = float(values.max())
            if np.isclose(min_val, max_val):
                means = np.array([min_val] * self.n_mf, dtype=np.float32)
                sigmas = np.ones(self.n_mf, dtype=np.float32)
            else:
                means = np.linspace(min_val, max_val, self.n_mf).astype(np.float32)
                sigmas = np.ones(self.n_mf, dtype=np.float32) * ((max_val - min_val) / max(self.n_mf, 1))
            self.mf_params[feature_idx] = {"means": means, "sigmas": sigmas}

        n_rules = self.n_mf ** X_train.shape[1]
        self.weights = rng.normal(loc=0.0, scale=0.01, size=(len(self.classes_), n_rules)).astype(np.float32)

    def _fuzzify(self, x: np.ndarray):
        fuzzified = []
        for feature_idx, feature_val in enumerate(x):
            activations = []
            for i in range(self.n_mf):
                mean = self.mf_params[feature_idx]["means"][i]
                sigma = self.mf_params[feature_idx]["sigmas"][i]
                activations.append(self._gaussian_mf(float(feature_val), float(mean), float(sigma)))
            fuzzified.append(np.array(activations, dtype=np.float32))
        return fuzzified

    def _generate_rules(self, fuzzified_input) -> np.ndarray:
        activations = np.array([1.0], dtype=np.float32)
        for feature_activations in fuzzified_input:
            activations = (activations[:, None] * feature_activations[None, :]).ravel()
        total = activations.sum()
        if total > 0:
            activations = activations / total
        return activations.astype(np.float32)

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        return exp_scores / (exp_scores.sum() + 1e-12)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ANFISClassifier":
        self._init_parameters(X_train, y_train)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        rng = np.random.default_rng(self.random_state)

        for _epoch in range(self.n_epochs):
            order = rng.permutation(X_train.shape[0])
            for i in order:
                rules = self._generate_rules(self._fuzzify(X_train[i]))
                scores = self.weights @ rules
                probs = self._softmax(scores)
                target = np.zeros(len(self.classes_), dtype=np.float32)
                target[class_to_idx[y_train[i]]] = 1.0
                error = target - probs
                self.weights += self.lr * np.outer(error, rules).astype(np.float32)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X_test:
            rules = self._generate_rules(self._fuzzify(x))
            scores = self.weights @ rules
            predictions.append(self.classes_[int(np.argmax(scores))])
        return np.array(predictions)


def prepare_data(data: Tuple[np.ndarray, ...], seed: int, params: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    X_train, X_val, X_test, y_train, y_val, y_test = data

    n_components = min(int(params["pca_components"]), X_train.shape[1])
    if X_train.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

    max_samples = int(params["max_train_samples"])
    if X_train.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_train.shape[0], size=max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(params: Dict[str, Any], random_state: int) -> ANFISClassifier:
    return ANFISClassifier(
        n_membership_functions=params["n_membership_functions"],
        learning_rate=params["learning_rate"],
        n_epochs=params["n_epochs"],
        random_state=random_state,
    )


def test_single_parameter(
    dataset_name: str,
    data: Tuple[np.ndarray, ...],
    param_name: str,
    param_values: List[Any],
) -> List[Dict[str, Any]]:
    X_train, X_val, X_test, y_train, y_val, y_test = data
    labels = np.unique(np.concatenate([y_train, y_val, y_test])).tolist()
    results = []

    print(f"\n  Testando parâmetro: {param_name}")
    for param_value in param_values:
        print(f"    Valor: {param_value}...", end=" ", flush=True)
        test_params = BASE_PARAMS.copy()
        test_params[param_name] = param_value

        val_f1_scores = []
        test_f1_scores = []
        times = []

        for seed in STUDY_SEEDS:
            current_data = prepare_data((X_train, X_val, X_test, y_train, y_val, y_test), seed, test_params)
            start = time.perf_counter()
            model = build_model(test_params, seed)
            model.fit(current_data[0], current_data[3])
            metrics = evaluate_model(model, current_data, labels)
            elapsed = time.perf_counter() - start

            val_f1_scores.append(metrics["val"]["f1"])
            test_f1_scores.append(metrics["test"]["f1"])
            times.append(elapsed)

        result = {
            "dataset": dataset_name,
            "param_name": param_name,
            "param_value": str(param_value),
            "val_f1_mean": float(np.mean(val_f1_scores)),
            "val_f1_std": float(np.std(val_f1_scores, ddof=1)) if len(val_f1_scores) > 1 else 0.0,
            "test_f1_mean": float(np.mean(test_f1_scores)),
            "test_f1_std": float(np.std(test_f1_scores, ddof=1)) if len(test_f1_scores) > 1 else 0.0,
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "n_seeds": len(STUDY_SEEDS),
        }
        results.append(result)
        print(f"val_f1={result['val_f1_mean']:.4f}±{result['val_f1_std']:.4f}, test_f1={result['test_f1_mean']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudo de hiperparâmetros ANFIS")
    parser.add_argument("--dataset", default="adult", help="Dataset para estudo (padrão: adult)")
    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in DATASETS:
        print(f"Dataset inválido: {dataset_name}")
        return

    dataset_path = Path("datasets/processed") / dataset_name
    if not dataset_path.exists():
        print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
        return

    print(f"\n[Estudo de Hiperparâmetros - ANFIS no dataset {dataset_name}]")
    print(f"Valores base: {json.dumps(BASE_PARAMS, ensure_ascii=False)}")
    data = load_dataset(dataset_path)

    output_dir = Path("resultados[2]/anfis_hyperparameter_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for param_name, param_values in PARAMETER_STUDIES.items():
        results = test_single_parameter(dataset_name, data, param_name, param_values)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    csv_path = output_dir / f"hyperparameter_study_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    summary = {}
    for param_name in PARAMETER_STUDIES.keys():
        param_results = df[df["param_name"] == param_name].sort_values("test_f1_mean", ascending=False)
        best = param_results.iloc[0]
        summary[param_name] = {
            "best_value": best["param_value"],
            "best_test_f1": float(best["test_f1_mean"]),
            "best_val_f1": float(best["val_f1_mean"]),
            "top_3": [
                {
                    "value": str(row["param_value"]),
                    "test_f1": float(row["test_f1_mean"]),
                    "val_f1": float(row["val_f1_mean"]),
                }
                for _, row in param_results.head(3).iterrows()
            ],
        }

    summary_path = output_dir / f"summary_{dataset_name}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sumário salvo em: {summary_path}")


if __name__ == "__main__":
    main()
