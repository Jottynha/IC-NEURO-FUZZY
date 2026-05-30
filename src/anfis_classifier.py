# Experimentos com ANFIS simplificado: grade, 21 execuções e matriz de confusão.

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA

from experiment_utils import (
    DATASETS,
    DEFAULT_RANDOM_STATES,
    evaluate_model,
    aggregate_runs,
    flatten_params,
    load_dataset,
    save_global_summary,
    write_report,
)

ALGORITHM_NAME = "ANFIS (Adaptive Neuro-Fuzzy Inference System)"

PARAM_GRID = [
    {"n_membership_functions": 2, "learning_rate": 0.01, "n_epochs": 10, "pca_components": 6, "max_train_samples": 300},
    {"n_membership_functions": 2, "learning_rate": 0.01, "n_epochs": 25, "pca_components": 6, "max_train_samples": 300},
    {"n_membership_functions": 2, "learning_rate": 0.05, "n_epochs": 25, "pca_components": 6, "max_train_samples": 300},
    {"n_membership_functions": 3, "learning_rate": 0.01, "n_epochs": 10, "pca_components": 6, "max_train_samples": 300},
    {"n_membership_functions": 2, "learning_rate": 0.01, "n_epochs": 25, "pca_components": 8, "max_train_samples": 300},
]


class ANFISClassifier:
    """ANFIS simplificado para classificação multiclasse.

    A camada fuzzy usa funções gaussianas por atributo. As ativações das regras
    alimentam uma camada consequente treinada por atualização tipo gradiente em
    classificação one-vs-rest/softmax.
    """

    def __init__(
        self,
        n_membership_functions: int = 2,
        learning_rate: float = 0.01,
        n_epochs: int = 10,
        random_state: int = 42,
    ):
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
        # Produto das pertinências, calculado por produto cartesiano iterativo.
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


def parse_datasets(value: str | None) -> list[str]:
    if value is None:
        return list(DATASETS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in DATASETS]
    if invalid:
        raise ValueError(f"Datasets inválidos: {', '.join(invalid)}")
    return selected


def build_model(params: Dict[str, Any], random_state: int) -> ANFISClassifier:
    return ANFISClassifier(
        n_membership_functions=params["n_membership_functions"],
        learning_rate=params["learning_rate"],
        n_epochs=params["n_epochs"],
        random_state=random_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa experimentos do ANFIS")
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
        print(f"\n[Treinando ANFIS para {dataset_name}]")
        original_data = load_dataset(dataset_path)
        labels = np.unique(np.concatenate([original_data[3], original_data[4], original_data[5]])).tolist()
        run_results = []
        tried_results = []

        import time

        for run_idx, seed in enumerate(DEFAULT_RANDOM_STATES, start=1):
            best_candidate = None
            print(f"  Execução {run_idx:02d}/{len(DEFAULT_RANDOM_STATES)} | seed={seed}")
            for params in PARAM_GRID:
                data = prepare_data(original_data, seed, params)
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
        title="RESULTADOS - ANFIS (ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM)",
        experiments=experiments,
        output_txt=Path("resultados/resultados_anfis.txt"),
        output_csv=Path("resultados/resultados_anfis_melhores_execucoes.csv"),
        output_all_params_csv=Path("resultados/resultados_anfis_todos_parametros.csv"),
        output_json=Path("resultados/resultados_anfis_detalhado.json"),
    )
    save_global_summary(experiments, Path("resultados/resumo_anfis.csv"))
    print("\nResultados do ANFIS salvos em resultados/")


if __name__ == "__main__":
    main()
