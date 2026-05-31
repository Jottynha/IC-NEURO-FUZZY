"""
Estudo sistemático de hiperparâmetros para Bernoulli RBM + Logistic Regression.
Testa variações univariadas e salva resumos em `resultados[2]/rbm_hyperparameter_study/`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from experiment_utils import DATASETS, load_dataset, evaluate_model

# Valores base
BASE_PARAMS = {
    "n_components": 64,
    "n_iter": 10,
    "learning_rate": 0.01,
    "logistic_C": 1.0,
}

# Seeds reduzidas para estudo rápido
STUDY_SEEDS = list(range(1, 6))  # 5 seeds

# Variações a testar (um parâmetro por vez)
PARAMETER_STUDIES = {
    "n_components": [8, 16, 32, 64, 128],
    "n_iter": [5, 10, 20],
    "learning_rate": [0.001, 0.01, 0.05],
    "logistic_C": [0.1, 1.0, 10.0],
}


def scale_data(data: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    X_train, X_val, X_test, y_train, y_val, y_test = data
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test


def build_pipeline(params: Dict[str, Any], random_state: int) -> Pipeline:
    rbm = BernoulliRBM(
        n_components=int(params["n_components"]),
        n_iter=int(params["n_iter"]),
        learning_rate=float(params["learning_rate"]),
        random_state=random_state,
        verbose=0,
    )
    lr = LogisticRegression(C=float(params.get("logistic_C", 1.0)), max_iter=1000, random_state=random_state)
    return Pipeline([("rbm", rbm), ("logistic", lr)])


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

        # Escalar dados antes de treinar RBM
        scaled_data = scale_data((X_train, X_val, X_test, y_train, y_val, y_test))

        for seed in STUDY_SEEDS:
            start = time.perf_counter()
            model = build_pipeline(test_params, seed)
            model.fit(scaled_data[0], scaled_data[3])
            metrics = evaluate_model(model, scaled_data, labels)
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
    parser = argparse.ArgumentParser(description="Estudo de hiperparâmetros RBM + Logistic")
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

    print(f"\n[Estudo de Hiperparâmetros - RBM+Logistic no dataset {dataset_name}]")
    print(f"Valores base: {json.dumps(BASE_PARAMS, ensure_ascii=False)}")
    data = load_dataset(dataset_path)

    output_dir = Path("resultados[2]/rbm_hyperparameter_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for param_name, param_values in PARAMETER_STUDIES.items():
        res = test_single_parameter(dataset_name, data, param_name, param_values)
        all_results.extend(res)

    df = pd.DataFrame(all_results)
    csv_path = output_dir / f"hyperparameter_study_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    # Gerar sumário
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

    print("\nEstudo concluído.")


if __name__ == "__main__":
    main()
