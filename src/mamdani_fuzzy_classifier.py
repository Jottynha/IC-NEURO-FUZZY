# Experimentos com Sistema Fuzzy de Mamdani: grade, 21 execuções e matriz de confusão.

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from experiment_utils import (
    DATASETS,
    DEFAULT_RANDOM_STATES,
    load_dataset,
    run_parameter_search_experiment,
    save_global_summary,
    write_report,
)

ALGORITHM_NAME = "Sistema Fuzzy de Mamdani"

PARAM_GRID = [
    {"n_membership_functions": 2, "max_train_samples": 300},
    {"n_membership_functions": 3, "max_train_samples": 300},
    {"n_membership_functions": 5, "max_train_samples": 300},
    {"n_membership_functions": 3, "max_train_samples": 500},
]


class MamdaniFuzzyClassifier:
    """Classificador fuzzy Mamdani simplificado baseado em similaridade fuzzy.

    A implementação cria funções de pertinência triangulares por atributo e usa
    uma estratégia de vizinho mais similar no espaço fuzificado. É simples, mas
    suficiente para comparação experimental com os demais classificadores.
    """

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
            # Similaridade fuzzy por interseção/união, com fallback por distância.
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
        n_membership_functions=params["n_membership_functions"],
        random_state=random_state,
    )


def run_mamdani_for_params(data: Tuple[np.ndarray, ...], seed: int, params: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    return subsample_training(data, seed, int(params["max_train_samples"]))


def parse_datasets(value: str | None) -> list[str]:
    if value is None:
        return list(DATASETS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in DATASETS]
    if invalid:
        raise ValueError(f"Datasets inválidos: {', '.join(invalid)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa experimentos do Sistema Fuzzy de Mamdani")
    parser.add_argument("--dataset", default=None, help="Dataset único ou lista separada por vírgula")
    args = parser.parse_args()

    datasets_root = Path("datasets/processed")
    experiments = []
    selected_datasets = parse_datasets(args.dataset)

    # Como o tamanho do subconjunto de treino faz parte dos parâmetros, precisamos
    # aplicar a transformação dentro do model_builder. Para isso, usamos uma função
    # de experimento específica abaixo.
    from experiment_utils import evaluate_model, aggregate_runs, flatten_params
    import time

    for dataset_name in selected_datasets:
        dataset_path = datasets_root / dataset_name
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
            continue
        print(f"\n[Treinando Fuzzy Mamdani para {dataset_name}]")
        original_data = load_dataset(dataset_path)
        labels = np.unique(np.concatenate([original_data[3], original_data[4], original_data[5]])).tolist()
        run_results = []
        tried_results = []

        for run_idx, seed in enumerate(DEFAULT_RANDOM_STATES, start=1):
            best_candidate = None
            print(f"  Execução {run_idx:02d}/{len(DEFAULT_RANDOM_STATES)} | seed={seed}")
            for params in PARAM_GRID:
                data = run_mamdani_for_params(original_data, seed, params)
                current_labels = np.unique(np.concatenate([data[3], data[4], data[5]])).tolist()
                start = time.perf_counter()
                model = build_model(params, seed)
                model.fit(data[0], data[3])
                metrics = evaluate_model(model, data, current_labels)
                elapsed = time.perf_counter() - start
                candidate = {
                    "dataset": dataset_name,
                    "algorithm": ALGORITHM_NAME,
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
            print(
                "    melhor F1_val="
                f"{best_candidate['val']['f1']:.4f} | F1_test={best_candidate['test']['f1']:.4f} "
                f"| params={best_candidate['params_text']}"
            )

        experiments.append({
            "dataset": dataset_name,
            "algorithm": ALGORITHM_NAME,
            "labels": labels,
            "param_grid": PARAM_GRID,
            "runs": run_results,
            "tried": tried_results,
            "summary": aggregate_runs(run_results),
        })

    write_report(
        title="RESULTADOS - SISTEMA FUZZY DE MAMDANI",
        experiments=experiments,
        output_txt=Path("resultados/resultados_mamdani_fuzzy.txt"),
        output_csv=Path("resultados/resultados_mamdani_fuzzy_melhores_execucoes.csv"),
        output_all_params_csv=Path("resultados/resultados_mamdani_fuzzy_todos_parametros.csv"),
        output_json=Path("resultados/resultados_mamdani_fuzzy_detalhado.json"),
    )
    save_global_summary(experiments, Path("resultados/resumo_mamdani_fuzzy.csv"))
    print("\nResultados do Mamdani salvos em resultados/")


if __name__ == "__main__":
    main()
