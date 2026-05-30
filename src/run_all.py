#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_SCRIPTS = {
    "mlp": "src/mlp_classifier.py",
    "rbm": "src/rbm_logistic_classifier.py",
    "mamdani": "src/mamdani_fuzzy_classifier.py",
    "anfis": "src/anfis_classifier.py",
}
DATASET_FRACTIONS = {
    "adult": 0.30,
    "bank_marketing": 0.30,
    "heart_disease": 1.00,
    "mushroom": 0.50,
}


def parse_csv_argument(value: str | None, choices: Iterable[str]) -> list[str]:
    choices_list = list(choices)
    if value is None:
        return choices_list
    requested = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in requested if item not in choices_list]
    if invalid:
        raise ValueError(f"Valores inválidos: {', '.join(invalid)}")
    return requested


def run_script(path: str, extra_args=None) -> int:
    print(f"\n>>> Executando: {path}")
    start = time.time()
    cmd = [sys.executable, path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start
    print(f"<<< Concluído: {path} (tempo {elapsed:.1f}s)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Executa pré-processamento e algoritmos selecionados")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Se informado, usa a mesma fração para todos os datasets",
    )
    parser.add_argument("--datasets", default=None, help="Lista de datasets separados por vírgula")
    parser.add_argument("--algorithms", default=None, help="Lista de algoritmos separados por vírgula")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Mapa dataset=alg1,alg2. Pode ser usado várias vezes.",
    )
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)

    selected_datasets = parse_csv_argument(args.datasets, DATASET_FRACTIONS.keys())
    selected_algorithms = parse_csv_argument(args.algorithms, ALGORITHM_SCRIPTS.keys())

    dataset_algorithm_map: dict[str, list[str]] = {
        dataset: list(selected_algorithms) for dataset in selected_datasets
    }
    if args.run:
        dataset_algorithm_map = {}
        for item in args.run:
            if "=" not in item:
                raise ValueError(f"Formato inválido em --run: {item}. Use dataset=alg1,alg2")
            dataset_name, algorithms_text = item.split("=", 1)
            dataset_name = dataset_name.strip()
            if dataset_name not in DATASET_FRACTIONS:
                raise ValueError(f"Dataset inválido em --run: {dataset_name}")
            algorithms = parse_csv_argument(algorithms_text, ALGORITHM_SCRIPTS.keys())
            dataset_algorithm_map[dataset_name] = algorithms
        selected_datasets = list(dataset_algorithm_map.keys())

    for dataset_name in selected_datasets:
        default_fraction = DATASET_FRACTIONS[dataset_name]
        sample_fraction = args.sample_fraction if args.sample_fraction is not None else default_fraction
        rc = run_script(
            "src/preprocessing.py",
            [
                "--dataset",
                dataset_name,
                "--random-state",
                str(args.random_state),
                "--sample-fraction",
                str(sample_fraction),
                "--skip-exploratory",
            ],
        )
        if rc != 0:
            print(f"Pré-processamento de {dataset_name} retornou código {rc}. Interrompendo.")
            return

    for dataset_name, algorithms in dataset_algorithm_map.items():
        for algorithm in algorithms:
            script = ALGORITHM_SCRIPTS[algorithm]
            rc = run_script(script, ["--dataset", dataset_name])
            if rc != 0:
                print(f"Script {script} retornou código {rc}. Interrompendo.")
                return


if __name__ == "__main__":
    main()
