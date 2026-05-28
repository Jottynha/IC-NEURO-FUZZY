# Experimentos com Bernoulli RBM + Regressão Logística: grade, 21 execuções e matriz de confusão.

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from experiment_utils import (
    DATASETS,
    DEFAULT_RANDOM_STATES,
    load_dataset,
    run_parameter_search_experiment,
    save_global_summary,
    write_report,
)

ALGORITHM_NAME = "Bernoulli RBM + Regressão Logística"

PARAM_GRID = [
    {"n_components": 32, "n_iter": 10, "learning_rate": 0.01},
    {"n_components": 64, "n_iter": 10, "learning_rate": 0.01},
    {"n_components": 64, "n_iter": 20, "learning_rate": 0.01},
    {"n_components": 128, "n_iter": 20, "learning_rate": 0.01},
    {"n_components": 64, "n_iter": 20, "learning_rate": 0.05},
]


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


def build_model(params: Dict[str, Any], random_state: int) -> Pipeline:
    rbm = BernoulliRBM(
        n_components=params["n_components"],
        n_iter=params["n_iter"],
        learning_rate=params["learning_rate"],
        random_state=random_state,
        verbose=0,
    )
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    return Pipeline([("rbm", rbm), ("logistic", lr)])


def main() -> None:
    datasets_root = Path("datasets/processed")
    experiments = []

    for dataset_name in DATASETS:
        dataset_path = datasets_root / dataset_name
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
            continue
        print(f"\n[Treinando RBM + Regressão Logística para {dataset_name}]")
        data = load_dataset(dataset_path)
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
        title="RESULTADOS - BERNOULLI RBM + REGRESSÃO LOGÍSTICA",
        experiments=experiments,
        output_txt=Path("resultados/resultados_rbm_logistic.txt"),
        output_csv=Path("resultados/resultados_rbm_logistic_melhores_execucoes.csv"),
        output_all_params_csv=Path("resultados/resultados_rbm_logistic_todos_parametros.csv"),
        output_json=Path("resultados/resultados_rbm_logistic_detalhado.json"),
    )
    save_global_summary(experiments, Path("resultados/resumo_rbm_logistic.csv"))
    print("\nResultados da RBM + Regressão Logística salvos em resultados/")


if __name__ == "__main__":
    main()
