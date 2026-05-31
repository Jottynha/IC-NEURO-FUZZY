"""
MLP com configuração otimizada (baseada no estudo incremental).
Testa em todos os 4 datasets com 21 seeds (setup final).
"""

import argparse
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

ALGORITHM_NAME = "MLP (Perceptron Multicamadas) - Otimizado"

# Configuração otimizada com base no estudo incremental
PARAM_GRID = [
    {
        "hidden_layer_sizes": (50,),
        "activation": "relu",
        "learning_rate_init": 0.0001,
        "alpha": 0.01,
    }
]


def build_model(params: Dict[str, Any], random_state: int) -> MLPClassifier:
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


def parse_datasets(value: str | None) -> list[str]:
    if value is None:
        return list(DATASETS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in DATASETS]
    if invalid:
        raise ValueError(f"Datasets inválidos: {', '.join(invalid)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa MLP com configuração otimizada")
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
        print(f"\n[Treinando MLP (Otimizado) para {dataset_name}]")
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
        title="RESULTADOS - MLP (PERCEPTRON MULTICAMADAS) - OTIMIZADO",
        experiments=experiments,
        output_txt=Path("resultados[2]/mlp_optimized/resultados_mlp_otimizado.txt"),
        output_csv=Path("resultados[2]/mlp_optimized/resultados_mlp_otimizado_melhores.csv"),
        output_all_params_csv=Path("resultados[2]/mlp_optimized/resultados_mlp_otimizado_todos.csv"),
        output_json=Path("resultados[2]/mlp_optimized/resultados_mlp_otimizado_detalhado.json"),
    )
    save_global_summary(experiments, Path("resultados[2]/mlp_optimized/resumo_mlp_otimizado.csv"))
    print("\nResultados da MLP (Otimizado) salvos em resultados[2]/mlp_optimized/")


if __name__ == "__main__":
    main()
