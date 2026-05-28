# Experimentos com MLP: grade de parâmetros, 21 execuções e matriz de confusão.

from pathlib import Path
from typing import Any, Dict

from sklearn.neural_network import MLPClassifier

from experiment_utils import (
    DATASETS,
    DEFAULT_RANDOM_STATES,
    load_dataset,
    run_parameter_search_experiment,
    save_global_summary,
    write_report,
)

ALGORITHM_NAME = "MLP (Perceptron Multicamadas)"

PARAM_GRID = [
    {"hidden_layer_sizes": (50,), "activation": "relu", "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100,), "activation": "relu", "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100, 50), "activation": "relu", "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100,), "activation": "tanh", "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100, 50), "activation": "tanh", "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (100,), "activation": "relu", "learning_rate_init": 0.01},
]


def build_model(params: Dict[str, Any], random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=50,
    )


def main() -> None:
    datasets_root = Path("datasets/processed")
    experiments = []

    for dataset_name in DATASETS:
        dataset_path = datasets_root / dataset_name
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
            continue
        print(f"\n[Treinando MLP para {dataset_name}]")
        data = load_dataset(dataset_path)
        experiment = run_parameter_search_experiment(
            algorithm_name=ALGORITHM_NAME,
            dataset_name=dataset_name,
            data=data,
            param_grid=PARAM_GRID,
            random_states=DEFAULT_RANDOM_STATES,
            model_builder=build_model,
        )
        experiments.append(experiment)

    write_report(
        title="RESULTADOS - MLP (PERCEPTRON MULTICAMADAS)",
        experiments=experiments,
        output_txt=Path("resultados/resultados_mlp.txt"),
        output_csv=Path("resultados/resultados_mlp_melhores_execucoes.csv"),
        output_all_params_csv=Path("resultados/resultados_mlp_todos_parametros.csv"),
        output_json=Path("resultados/resultados_mlp_detalhado.json"),
    )
    save_global_summary(experiments, Path("resultados/resumo_mlp.csv"))
    print("\nResultados da MLP salvos em resultados/")


if __name__ == "__main__":
    main()
