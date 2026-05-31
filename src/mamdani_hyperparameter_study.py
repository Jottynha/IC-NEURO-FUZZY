"""
Estudo sistemático de hiperparâmetros do Sistema Fuzzy de Mamdani.
Testa variações univariadas e salva resumos em `resultados[2]/mamdani_hyperparameter_study/`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from experiment_utils import DATASETS, evaluate_model, load_dataset

# Valores base conservadores.
BASE_PARAMS = {
    "n_membership_functions": 3,
    "max_train_samples": 300,
}

STUDY_SEEDS = list(range(1, 6))

PARAMETER_STUDIES = {
    "n_membership_functions": [2, 3, 5, 7],
    "max_train_samples": [150, 300, 500, 800],
}


class MamdaniFuzzyClassifier:
    def __init__(self, n_membership_functions: int = 3, random_state: int = 42):
        self.n_mf = n_membership_functions
        self.random_state = random_state
        self.membership_params = None
        self.X_train_fuzzified = None
        self.y_train = None

    def _triangular_mf(self, x: float, a: float, b: float, c: float) -> float:
        if b == a and x == b:
            return 1.0
        if c == b and x == b:
            return 1.0
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return float((x - a) / (b - a + 1e-12))
        return float((c - x) / (c - b + 1e-12))

    def _create_membership_functions(self, X_train: np.ndarray) -> None:
        self.membership_params = {}
        for feature_idx in range(X_train.shape[1]):
            feature_values = X_train[:, feature_idx]
            min_val = float(feature_values.min())
            max_val = float(feature_values.max())
            if np.isclose(min_val, max_val):
                self.membership_params[feature_idx] = [(min_val - 1.0, min_val, min_val + 1.0)] * self.n_mf
                continue

            centers = np.linspace(min_val, max_val, self.n_mf)
            step = centers[1] - centers[0] if self.n_mf > 1 else max_val - min_val
            params = []
            for center in centers:
                params.append((float(center - step), float(center), float(center + step)))
            self.membership_params[feature_idx] = params

    def _fuzzify(self, x: np.ndarray) -> np.ndarray:
        fuzzified = []
        for feature_idx, feature_val in enumerate(x):
            for a, b, c in self.membership_params[feature_idx]:
                fuzzified.append(self._triangular_mf(float(feature_val), a, b, c))
        return np.array(fuzzified, dtype=np.float32)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MamdaniFuzzyClassifier":
        self._create_membership_functions(X_train)
        self.X_train_fuzzified = np.array([self._fuzzify(x) for x in X_train], dtype=np.float32)
        self.y_train = y_train
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X_test:
            x_fuzzified = self._fuzzify(x)
            intersection = np.minimum(self.X_train_fuzzified, x_fuzzified).sum(axis=1)
            union = np.maximum(self.X_train_fuzzified, x_fuzzified).sum(axis=1) + 1e-12
            similarities = intersection / union
            if float(np.max(similarities)) <= 1e-12:
                distances = np.linalg.norm(self.X_train_fuzzified - x_fuzzified, axis=1)
                nearest_idx = int(np.argmin(distances))
            else:
                nearest_idx = int(np.argmax(similarities))
            predictions.append(self.y_train[nearest_idx])
        return np.array(predictions)


def subsample_training(data: Tuple[np.ndarray, ...], seed: int, max_train_samples: int) -> Tuple[np.ndarray, ...]:
    X_train, X_val, X_test, y_train, y_val, y_test = data
    if X_train.shape[0] <= max_train_samples:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_train.shape[0], size=max_train_samples, replace=False)
    return X_train[idx], X_val, X_test, y_train[idx], y_val, y_test


def build_model(params: Dict[str, Any], random_state: int) -> MamdaniFuzzyClassifier:
    return MamdaniFuzzyClassifier(
        n_membership_functions=int(params["n_membership_functions"]),
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
            current_data = subsample_training((X_train, X_val, X_test, y_train, y_val, y_test), seed, int(test_params["max_train_samples"]))
            current_labels = np.unique(np.concatenate([current_data[3], current_data[4], current_data[5]])).tolist()
            start = time.perf_counter()
            model = build_model(test_params, seed)
            model.fit(current_data[0], current_data[3])
            metrics = evaluate_model(model, current_data, current_labels)
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
    parser = argparse.ArgumentParser(description="Estudo de hiperparâmetros Mamdani")
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

    print(f"\n[Estudo de Hiperparâmetros - Mamdani no dataset {dataset_name}]")
    print(f"Valores base: {json.dumps(BASE_PARAMS, ensure_ascii=False)}")
    data = load_dataset(dataset_path)

    output_dir = Path("resultados[2]/mamdani_hyperparameter_study")
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
