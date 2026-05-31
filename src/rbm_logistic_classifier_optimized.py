"""
RBM + Logistic Regression com configuração otimizada baseada no estudo.
Executa a rotina final com 21 seeds e salva resultados em `resultados[2]/rbm_optimized/`.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from experiment_utils import (
    DATASETS,
    DEFAULT_RANDOM_STATES,
    load_dataset,
    run_parameter_search_experiment,
    save_global_summary,
    write_report,
)

ALGORITHM_NAME = "Bernoulli RBM + Regressão Logística - Otimizado"

# Parametro otimizado extraído do estudo em resultados[2]/rbm_hyperparameter_study
PARAM_GRID = [
    {"n_components": 128, "n_iter": 20, "learning_rate": 0.05, "logistic_C": 10.0},
]


def build_model(params: Dict[str, Any], random_state: int) -> Pipeline:
    rbm = BernoulliRBM(
        n_components=int(params["n_components"]),
        n_iter=int(params["n_iter"]),
        learning_rate=float(params["learning_rate"]),
        random_state=random_state,
        verbose=0,
    )
    lr = LogisticRegression(C=float(params.get("logistic_C", 1.0)), max_iter=1000, random_state=random_state)
    return Pipeline([("rbm", rbm), ("logistic", lr)])


def parse_datasets(value: str | None) -> list[str]:
    if value is None:
        return list(DATASETS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in DATASETS]
    if invalid:
        raise ValueError(f"Datasets inválidos: {', '.join(invalid)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa RBM+Logistic (Otimizado)")
    parser.add_argument("--dataset", default=None, help="Dataset único ou lista separada por vírgula")
    args = parser.parse_args()

    datasets_root = Path("datasets/processed")
    experiments = []
    selected_datasets = parse_datasets(args.dataset)

    for dataset_name in selected_datasets:
        dataset_path = datasets_root / dataset_name
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
            continue
        print(f"\n[Treinando RBM+Logistic (Otimizado) para {dataset_name}]")
        data = load_dataset(dataset_path)
        # Nota: scaling para RBM é aplicado internamente via fit_data_transformer no original;
        # usamos run_parameter_search_experiment com fit_data_transformer se necessário.
        from sklearn.preprocessing import MinMaxScaler

        def scale_to_unit_interval(data_tuple, random_state):
            X_train, X_val, X_test, y_train, y_val, y_test = data_tuple
            scaler = MinMaxScaler()
            return (
                scaler.fit_transform(X_train),
                scaler.transform(X_val),
                scaler.transform(X_test),
                y_train,
                y_val,
                y_test,
            )

        experiment = run_parameter_search_experiment(
            algorithm_name=ALGORITHM_NAME,
            dataset_name=dataset_name,
            data=data,
            param_grid=PARAM_GRID,
            random_states=DEFAULT_RANDOM_STATES,
            model_builder=build_model,
            fit_data_transformer=scale_to_unit_interval,
        )
        experiments.append(experiment)

    write_report(
        title="RESULTADOS - BERNOULLI RBM + REGRESSÃO LOGÍSTICA - OTIMIZADO",
        experiments=experiments,
        output_txt=Path("resultados[2]/rbm_optimized/resultados_rbm_optimized.txt"),
        output_csv=Path("resultados[2]/rbm_optimized/resultados_rbm_optimized_melhores.csv"),
        output_all_params_csv=Path("resultados[2]/rbm_optimized/resultados_rbm_optimized_todos.csv"),
        output_json=Path("resultados[2]/rbm_optimized/resultados_rbm_optimized_detalhado.json"),
    )
    save_global_summary(experiments, Path("resultados[2]/rbm_optimized/resumo_rbm_optimized.csv"))
    print("\nResultados da RBM+Logistic (Otimizado) salvos em resultados[2]/rbm_optimized/")


if __name__ == "__main__":
    main()
