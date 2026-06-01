"""Estudo unificado de hiperparâmetros (variação univariada)."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from experiment_utils import DATASETS, evaluate_model, load_dataset
from anfis_classifier import ANFISClassifier, prepare_data as anfis_prepare_data
from mamdani_fuzzy_classifier import MamdaniFuzzyClassifier, subsample_training as mamdani_subsample


STUDY_CONFIGS: dict[str, dict[str, Any]] = {
    "mlp": {
        "base_params": {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "alpha": 0.0001,
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "n_iter_no_change": 50,
            "batch_size": "auto",
        },
        "parameter_studies": {
            "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (250,), (100, 50)],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "alpha": [0.0, 0.0001, 0.001, 0.01],
        },
        "output_dir": "mlp_hyperparameter_study",
    },
    "rbm": {
        "base_params": {"n_components": 64, "n_iter": 10, "learning_rate": 0.01, "logistic_C": 1.0},
        "parameter_studies": {
            "n_components": [8, 16, 32, 64, 128],
            "n_iter": [5, 10, 20],
            "learning_rate": [0.001, 0.01, 0.05],
            "logistic_C": [0.1, 1.0, 10.0],
        },
        "output_dir": "rbm_hyperparameter_study",
    },
    "mamdani": {
        "base_params": {"n_membership_functions": 3, "max_train_samples": 300},
        "parameter_studies": {
            "n_membership_functions": [2, 3, 5, 7],
            "max_train_samples": [150, 300, 500, 800],
        },
        "output_dir": "mamdani_hyperparameter_study",
    },
    "anfis": {
        "base_params": {
            "n_membership_functions": 2,
            "learning_rate": 0.01,
            "n_epochs": 10,
            "pca_components": 4,
            "max_train_samples": 300,
        },
        "parameter_studies": {
            "n_membership_functions": [2, 3, 4],
            "learning_rate": [0.001, 0.01, 0.05],
            "n_epochs": [10, 25, 50],
            "pca_components": [2, 4, 6, 8],
            "max_train_samples": [150, 300, 500],
        },
        "output_dir": "anfis_hyperparameter_study",
    },
}


def build_mlp_model(params: Dict[str, Any], seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        learning_rate_init=params["learning_rate_init"],
        alpha=params["alpha"],
        max_iter=params["max_iter"],
        random_state=seed,
        early_stopping=params["early_stopping"],
        validation_fraction=params["validation_fraction"],
        n_iter_no_change=params["n_iter_no_change"],
        batch_size=params["batch_size"],
    )


def build_rbm_model(params: Dict[str, Any], seed: int) -> Pipeline:
    rbm = BernoulliRBM(
        n_components=int(params["n_components"]),
        n_iter=int(params["n_iter"]),
        learning_rate=float(params["learning_rate"]),
        random_state=seed,
        verbose=0,
    )
    lr = LogisticRegression(C=float(params.get("logistic_C", 1.0)), max_iter=1000, random_state=seed)
    return Pipeline([("rbm", rbm), ("logistic", lr)])


def scale_data(data: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    X_train, X_val, X_test, y_train, y_val, y_test = data
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test


def run_single_fit(
    algorithm: str,
    params: Dict[str, Any],
    seed: int,
    data: Tuple[np.ndarray, ...],
    labels: list[Any],
) -> tuple[Dict[str, Any], float]:
    start = time.perf_counter()

    if algorithm == "mlp":
        model = build_mlp_model(params, seed)
        model.fit(data[0], data[3])
        metrics = evaluate_model(model, data, labels)
        return metrics, time.perf_counter() - start

    if algorithm == "rbm":
        scaled = scale_data(data)
        model = build_rbm_model(params, seed)
        model.fit(scaled[0], scaled[3])
        metrics = evaluate_model(model, scaled, labels)
        return metrics, time.perf_counter() - start

    if algorithm == "mamdani":
        current_data = mamdani_subsample(data, seed, int(params["max_train_samples"]))
        current_labels = np.unique(np.concatenate([current_data[3], current_data[4], current_data[5]])).tolist()
        model = MamdaniFuzzyClassifier(n_membership_functions=int(params["n_membership_functions"]), random_state=seed)
        model.fit(current_data[0], current_data[3])
        metrics = evaluate_model(model, current_data, current_labels)
        return metrics, time.perf_counter() - start

    current_data = anfis_prepare_data(data, seed, params)
    current_labels = np.unique(np.concatenate([current_data[3], current_data[4], current_data[5]])).tolist()
    model = ANFISClassifier(
        n_membership_functions=int(params["n_membership_functions"]),
        learning_rate=float(params["learning_rate"]),
        n_epochs=int(params["n_epochs"]),
        random_state=seed,
    )
    model.fit(current_data[0], current_data[3])
    metrics = evaluate_model(model, current_data, current_labels)
    return metrics, time.perf_counter() - start


def test_single_parameter(
    algorithm: str,
    dataset_name: str,
    data: Tuple[np.ndarray, ...],
    param_name: str,
    param_values: List[Any],
    base_params: Dict[str, Any],
    seeds: list[int],
) -> List[Dict[str, Any]]:
    labels = np.unique(np.concatenate([data[3], data[4], data[5]])).tolist()
    results = []

    print(f"\n  Testando parâmetro: {param_name}")
    for value in param_values:
        print(f"    Valor: {value}...", end=" ", flush=True)
        params = base_params.copy()
        params[param_name] = value

        val_f1_scores = []
        test_f1_scores = []
        times = []

        for seed in seeds:
            metrics, elapsed = run_single_fit(algorithm, params, seed, data, labels)
            val_f1_scores.append(metrics["val"]["f1"])
            test_f1_scores.append(metrics["test"]["f1"])
            times.append(elapsed)

        result = {
            "dataset": dataset_name,
            "algorithm": algorithm,
            "param_name": param_name,
            "param_value": str(value),
            "val_f1_mean": float(np.mean(val_f1_scores)),
            "val_f1_std": float(np.std(val_f1_scores, ddof=1)) if len(val_f1_scores) > 1 else 0.0,
            "test_f1_mean": float(np.mean(test_f1_scores)),
            "test_f1_std": float(np.std(test_f1_scores, ddof=1)) if len(test_f1_scores) > 1 else 0.0,
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "n_seeds": len(seeds),
        }
        results.append(result)
        print(f"val_f1={result['val_f1_mean']:.4f}±{result['val_f1_std']:.4f}, test_f1={result['test_f1_mean']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudo incremental de hiperparâmetros")
    parser.add_argument("--algorithm", required=True, choices=list(STUDY_CONFIGS.keys()))
    parser.add_argument("--dataset", default="adult")
    parser.add_argument("--seeds", type=int, default=5, help="Quantidade de seeds para estudo rápido")
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        raise ValueError(f"Dataset inválido: {args.dataset}")

    config = STUDY_CONFIGS[args.algorithm]
    seeds = list(range(1, max(args.seeds, 1) + 1))

    dataset_path = Path("datasets/processed") / args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {args.dataset} não encontrado em {dataset_path}")

    print(f"\n[Estudo de Hiperparâmetros - {args.algorithm.upper()} no dataset {args.dataset}]")
    print(f"Valores base: {json.dumps(config['base_params'], ensure_ascii=False)}")

    data = load_dataset(dataset_path)

    all_results = []
    for param_name, param_values in config["parameter_studies"].items():
        all_results.extend(
            test_single_parameter(
                args.algorithm,
                args.dataset,
                data,
                param_name,
                param_values,
                config["base_params"],
                seeds,
            )
        )

    output_dir = Path("resultados[2]") / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    csv_path = output_dir / f"hyperparameter_study_{args.dataset}.csv"
    df.to_csv(csv_path, index=False)

    summary = {}
    for param_name in config["parameter_studies"].keys():
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

    summary_path = output_dir / f"summary_{args.dataset}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nResultados salvos em: {csv_path}")
    print(f"Sumário salvo em: {summary_path}")


if __name__ == "__main__":
    main()
