"""
Runner único para executar versões otimizadas dos algoritmos.
Substitui scripts *_optimized.py individuais.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from experiment_utils import (
    DATASETS,
    DEFAULT_RANDOM_STATES,
    aggregate_runs,
    evaluate_model,
    flatten_params,
    load_dataset,
    run_parameter_search_experiment,
    save_global_summary,
    write_report,
)
from anfis_classifier import ANFISClassifier, prepare_data as anfis_prepare_data
from mamdani_fuzzy_classifier import MamdaniFuzzyClassifier, subsample_training as mamdani_subsample


OPTIMIZED_CONFIGS: dict[str, dict[str, Any]] = {
    "mlp": {
        "algorithm_name": "MLP (Perceptron Multicamadas) - Otimizado",
        "param_grid": [
            {"hidden_layer_sizes": (50,), "activation": "relu", "learning_rate_init": 0.0001, "alpha": 0.01}
        ],
        "output_prefix": "mlp_optimized",
    },
    "rbm": {
        "algorithm_name": "Bernoulli RBM + Regressão Logística - Otimizado",
        "param_grid": [
            {"n_components": 128, "n_iter": 20, "learning_rate": 0.05, "logistic_C": 10.0}
        ],
        "output_prefix": "rbm_optimized",
    },
    "mamdani": {
        "algorithm_name": "Sistema Fuzzy de Mamdani - Otimizado",
        "param_grid": [
            {"n_membership_functions": 5, "max_train_samples": 800}
        ],
        "output_prefix": "mamdani_optimized",
    },
    "anfis": {
        "algorithm_name": "ANFIS (Adaptive Neuro-Fuzzy Inference System) - Otimizado",
        "param_grid": [
            {
                "n_membership_functions": 4,
                "learning_rate": 0.05,
                "n_epochs": 50,
                "pca_components": 2,
                "max_train_samples": 150,
            }
        ],
        "output_prefix": "anfis_optimized",
    },
}


def parse_csv_argument(value: str | None, choices: list[str]) -> list[str]:
    if value is None:
        return list(choices)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in requested if item not in choices]
    if invalid:
        raise ValueError(f"Valores inválidos: {', '.join(invalid)}")
    return requested


def parse_datasets(value: str | None) -> list[str]:
    return parse_csv_argument(value, list(DATASETS))


def build_mlp_model(params: Dict[str, Any], random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        learning_rate_init=params["learning_rate_init"],
        alpha=params["alpha"],
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=50,
        batch_size="auto",
    )


def build_rbm_model(params: Dict[str, Any], random_state: int) -> Pipeline:
    rbm = BernoulliRBM(
        n_components=int(params["n_components"]),
        n_iter=int(params["n_iter"]),
        learning_rate=float(params["learning_rate"]),
        random_state=random_state,
        verbose=0,
    )
    lr = LogisticRegression(C=float(params.get("logistic_C", 1.0)), max_iter=1000, random_state=random_state)
    return Pipeline([("rbm", rbm), ("logistic", lr)])


def scale_to_unit_interval(data: Tuple[np.ndarray, ...], random_state: int) -> Tuple[np.ndarray, ...]:
    X_train, X_val, X_test, y_train, y_val, y_test = data
    scaler = MinMaxScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        y_train,
        y_val,
        y_test,
    )


def run_simple_search(
    algorithm_key: str,
    dataset_name: str,
    data: Tuple[np.ndarray, ...],
) -> Dict[str, Any]:
    config = OPTIMIZED_CONFIGS[algorithm_key]
    if algorithm_key == "mlp":
        return run_parameter_search_experiment(
            algorithm_name=config["algorithm_name"],
            dataset_name=dataset_name,
            data=data,
            param_grid=config["param_grid"],
            random_states=DEFAULT_RANDOM_STATES,
            model_builder=build_mlp_model,
        )
    return run_parameter_search_experiment(
        algorithm_name=config["algorithm_name"],
        dataset_name=dataset_name,
        data=data,
        param_grid=config["param_grid"],
        random_states=DEFAULT_RANDOM_STATES,
        model_builder=build_rbm_model,
        fit_data_transformer=scale_to_unit_interval,
    )


def run_mamdani_optimized(dataset_name: str, data: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    config = OPTIMIZED_CONFIGS["mamdani"]
    labels = np.unique(np.concatenate([data[3], data[4], data[5]])).tolist()
    run_results = []
    tried_results = []

    for run_idx, seed in enumerate(DEFAULT_RANDOM_STATES, start=1):
        best_candidate = None
        print(f"  Execução {run_idx:02d}/{len(DEFAULT_RANDOM_STATES)} | seed={seed}")
        for params in config["param_grid"]:
            current_data = mamdani_subsample(data, seed, int(params["max_train_samples"]))
            current_labels = np.unique(np.concatenate([current_data[3], current_data[4], current_data[5]])).tolist()
            start = time.perf_counter()
            model = MamdaniFuzzyClassifier(n_membership_functions=int(params["n_membership_functions"]), random_state=seed)
            model.fit(current_data[0], current_data[3])
            metrics = evaluate_model(model, current_data, current_labels)
            elapsed = time.perf_counter() - start
            candidate = {
                "dataset": dataset_name,
                "algorithm": config["algorithm_name"],
                "run": run_idx,
                "random_state": seed,
                "params": params,
                "params_text": flatten_params(params),
                "elapsed_seconds": float(elapsed),
                **metrics,
            }
            tried_results.append(candidate)
            if best_candidate is None or metrics["val"]["f1"] > best_candidate["val"]["f1"]:
                best_candidate = candidate
        run_results.append(best_candidate)

    return {
        "dataset": dataset_name,
        "algorithm": config["algorithm_name"],
        "labels": labels,
        "param_grid": config["param_grid"],
        "runs": run_results,
        "tried": tried_results,
        "summary": aggregate_runs(run_results),
    }


def run_anfis_optimized(dataset_name: str, data: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    config = OPTIMIZED_CONFIGS["anfis"]
    labels = np.unique(np.concatenate([data[3], data[4], data[5]])).tolist()
    run_results = []
    tried_results = []

    for run_idx, seed in enumerate(DEFAULT_RANDOM_STATES, start=1):
        best_candidate = None
        print(f"  Execução {run_idx:02d}/{len(DEFAULT_RANDOM_STATES)} | seed={seed}")
        for params in config["param_grid"]:
            current_data = anfis_prepare_data(data, seed, params)
            current_labels = np.unique(np.concatenate([current_data[3], current_data[4], current_data[5]])).tolist()
            start = time.perf_counter()
            model = ANFISClassifier(
                n_membership_functions=int(params["n_membership_functions"]),
                learning_rate=float(params["learning_rate"]),
                n_epochs=int(params["n_epochs"]),
                random_state=seed,
            )
            model.fit(current_data[0], current_data[3])
            metrics = evaluate_model(model, current_data, current_labels)
            elapsed = time.perf_counter() - start
            candidate = {
                "dataset": dataset_name,
                "algorithm": config["algorithm_name"],
                "run": run_idx,
                "random_state": seed,
                "params": params,
                "params_text": flatten_params(params),
                "elapsed_seconds": float(elapsed),
                **metrics,
            }
            tried_results.append(candidate)
            if best_candidate is None or metrics["val"]["f1"] > best_candidate["val"]["f1"]:
                best_candidate = candidate
        run_results.append(best_candidate)

    return {
        "dataset": dataset_name,
        "algorithm": config["algorithm_name"],
        "labels": labels,
        "param_grid": config["param_grid"],
        "runs": run_results,
        "tried": tried_results,
        "summary": aggregate_runs(run_results),
    }


def run_algorithm(algorithm_key: str, dataset_name: str, data: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    if algorithm_key in {"mlp", "rbm"}:
        return run_simple_search(algorithm_key, dataset_name, data)
    if algorithm_key == "mamdani":
        return run_mamdani_optimized(dataset_name, data)
    return run_anfis_optimized(dataset_name, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa versões otimizadas dos algoritmos")
    parser.add_argument("--algorithm", required=True, choices=list(OPTIMIZED_CONFIGS.keys()))
    parser.add_argument("--dataset", default=None, help="Dataset único ou lista separada por vírgula")
    args = parser.parse_args()

    config = OPTIMIZED_CONFIGS[args.algorithm]
    selected_datasets = parse_datasets(args.dataset)
    datasets_root = Path("datasets/processed")
    experiments = []

    for dataset_name in selected_datasets:
        dataset_path = datasets_root / dataset_name
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
            continue
        print(f"\n[Treinando {config['algorithm_name']} para {dataset_name}]")
        data = load_dataset(dataset_path)
        experiments.append(run_algorithm(args.algorithm, dataset_name, data))

    output_root = Path("resultados[2]") / config["output_prefix"]
    output_root.mkdir(parents=True, exist_ok=True)
    short_name = args.algorithm

    write_report(
        title=f"RESULTADOS - {config['algorithm_name'].upper()}",
        experiments=experiments,
        output_txt=output_root / f"resultados_{short_name}_otimizado.txt",
        output_csv=output_root / f"resultados_{short_name}_otimizado_melhores.csv",
        output_all_params_csv=output_root / f"resultados_{short_name}_otimizado_todos.csv",
        output_json=output_root / f"resultados_{short_name}_otimizado_detalhado.json",
    )
    save_global_summary(experiments, output_root / f"resumo_{short_name}_otimizado.csv")
    print(f"\nResultados salvos em {output_root}")


if __name__ == "__main__":
    main()
