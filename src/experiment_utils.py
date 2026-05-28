"""Funções comuns para experimentos com múltiplas execuções.

Este módulo centraliza carregamento dos dados, cálculo de métricas,
agregação de resultados e escrita dos relatórios em TXT/CSV/JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATASETS = ["adult", "bank_marketing", "heart_disease", "mushroom"]
DEFAULT_RANDOM_STATES = list(range(1, 22))
SPLITS = ["train", "val", "test"]
METRICS = ["accuracy", "precision", "recall", "f1"]


def load_dataset(dataset_path: Path):
    X_train = np.load(dataset_path / "X_train.npy")
    X_val = np.load(dataset_path / "X_val.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_train = np.load(dataset_path / "y_train.npy")
    y_val = np.load(dataset_path / "y_val.npy")
    y_test = np.load(dataset_path / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def evaluate_model(model: Any, data: Tuple[np.ndarray, ...], labels: Sequence[Any]) -> Dict[str, Any]:
    X_train, X_val, X_test, y_train, y_val, y_test = data
    predictions = {
        "train": model.predict(X_train),
        "val": model.predict(X_val),
        "test": model.predict(X_test),
    }
    targets = {"train": y_train, "val": y_val, "test": y_test}
    return {split: metric_dict(targets[split], predictions[split], labels) for split in SPLITS}


def flatten_params(params: Dict[str, Any]) -> str:
    if not params:
        return "{}"
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


def aggregate_runs(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrega métricas e matrizes de confusão de várias execuções."""
    summary: Dict[str, Any] = {}
    for split in SPLITS:
        split_summary: Dict[str, Any] = {}
        for metric in METRICS:
            values = np.array([r[split][metric] for r in run_results], dtype=float)
            split_summary[f"{metric}_mean"] = float(values.mean())
            split_summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        matrices = np.array([r[split]["confusion_matrix"] for r in run_results], dtype=int)
        split_summary["confusion_matrix_sum"] = matrices.sum(axis=0).astype(int).tolist()
        summary[split] = split_summary
    times = np.array([r["elapsed_seconds"] for r in run_results], dtype=float)
    summary["time_seconds_mean"] = float(times.mean())
    summary["time_seconds_std"] = float(times.std(ddof=1)) if len(times) > 1 else 0.0
    return summary


def run_parameter_search_experiment(
    *,
    algorithm_name: str,
    dataset_name: str,
    data: Tuple[np.ndarray, ...],
    param_grid: List[Dict[str, Any]],
    random_states: Sequence[int],
    model_builder: Callable[[Dict[str, Any], int], Any],
    fit_data_transformer: Callable[[Tuple[np.ndarray, ...], int], Tuple[np.ndarray, ...]] | None = None,
) -> Dict[str, Any]:
    """Executa busca em grade com seleção por F1 de validação em cada semente.

    Para cada random_state:
      1. testa todos os parâmetros da grade;
      2. seleciona a configuração com maior F1 na validação;
      3. registra métricas completas em treino, validação e teste;
      4. salva matriz de confusão de cada split.
    """
    labels = np.unique(np.concatenate([data[3], data[4], data[5]])).tolist()
    run_results: List[Dict[str, Any]] = []
    tried_results: List[Dict[str, Any]] = []

    for run_idx, seed in enumerate(random_states, start=1):
        current_data = fit_data_transformer(data, seed) if fit_data_transformer else data
        current_labels = np.unique(np.concatenate([current_data[3], current_data[4], current_data[5]])).tolist()
        best_candidate: Dict[str, Any] | None = None

        print(f"  Execução {run_idx:02d}/{len(random_states)} | seed={seed}")
        for params in param_grid:
            start = time.perf_counter()
            model = model_builder(params, seed)
            model.fit(current_data[0], current_data[3])
            metrics = evaluate_model(model, current_data, current_labels)
            elapsed = time.perf_counter() - start

            candidate = {
                "dataset": dataset_name,
                "algorithm": algorithm_name,
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

        assert best_candidate is not None
        run_results.append(best_candidate)
        print(
            "    melhor F1_val="
            f"{best_candidate['val']['f1']:.4f} | F1_test={best_candidate['test']['f1']:.4f} "
            f"| params={best_candidate['params_text']}"
        )

    return {
        "dataset": dataset_name,
        "algorithm": algorithm_name,
        "labels": labels,
        "param_grid": param_grid,
        "runs": run_results,
        "tried": tried_results,
        "summary": aggregate_runs(run_results),
    }


def rows_from_runs(experiment: Dict[str, Any], include_all_tried: bool = False) -> List[Dict[str, Any]]:
    source = experiment["tried"] if include_all_tried else experiment["runs"]
    rows: List[Dict[str, Any]] = []
    for r in source:
        row = {
            "dataset": r["dataset"],
            "algorithm": r["algorithm"],
            "run": r["run"],
            "random_state": r["random_state"],
            "params": r["params_text"],
            "elapsed_seconds": r["elapsed_seconds"],
        }
        for split in SPLITS:
            for metric in METRICS:
                row[f"{split}_{metric}"] = r[split][metric]
        rows.append(row)
    return rows


def summary_rows(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        row = {
            "dataset": exp["dataset"],
            "algorithm": exp["algorithm"],
            "n_runs": len(exp["runs"]),
            "param_grid_size": len(exp["param_grid"]),
            "time_seconds_mean": exp["summary"]["time_seconds_mean"],
            "time_seconds_std": exp["summary"]["time_seconds_std"],
        }
        for split in SPLITS:
            for metric in METRICS:
                row[f"{split}_{metric}_mean"] = exp["summary"][split][f"{metric}_mean"]
                row[f"{split}_{metric}_std"] = exp["summary"][split][f"{metric}_std"]
        # parâmetro mais frequente entre as melhores execuções
        params = [r["params_text"] for r in exp["runs"]]
        row["most_frequent_best_params"] = max(set(params), key=params.count) if params else ""
        rows.append(row)
    return rows


def write_report(
    *,
    title: str,
    experiments: List[Dict[str, Any]],
    output_txt: Path,
    output_csv: Path,
    output_all_params_csv: Path,
    output_json: Path,
) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    # CSV com melhores execuções e CSV com todas as tentativas da grade.
    best_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for exp in experiments:
        best_rows.extend(rows_from_runs(exp, include_all_tried=False))
        all_rows.extend(rows_from_runs(exp, include_all_tried=True))
    pd.DataFrame(best_rows).to_csv(output_csv, index=False)
    pd.DataFrame(all_rows).to_csv(output_all_params_csv, index=False)

    # JSON detalhado, mantendo matrizes de confusão.
    output_json.write_text(json.dumps(experiments, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # TXT legível.
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append(title)
    lines.append("=" * 90)
    lines.append("Metodologia experimental:")
    lines.append("- Para cada dataset, são feitas 21 execuções independentes por random_state.")
    lines.append("- Em cada execução, todos os parâmetros da grade são testados.")
    lines.append("- A melhor configuração é escolhida pelo maior F1-score no conjunto de validação.")
    lines.append("- As métricas finais são calculadas em treino, validação e teste.")
    lines.append("- As matrizes de confusão abaixo são somas das 21 execuções selecionadas.")
    lines.append("")

    for exp in experiments:
        lines.append("=" * 90)
        lines.append(f"Dataset: {exp['dataset'].upper()}")
        lines.append(f"Algoritmo: {exp['algorithm']}")
        lines.append(f"Classes: {exp['labels']}")
        lines.append(f"Tamanho da grade de parâmetros: {len(exp['param_grid'])}")
        lines.append("Parâmetros testados:")
        for i, params in enumerate(exp["param_grid"], start=1):
            lines.append(f"  {i}. {flatten_params(params)}")
        lines.append("")
        lines.append("Resumo das 21 execuções selecionadas:")
        for split in SPLITS:
            lines.append(f"\n{split.upper()}:")
            for metric in METRICS:
                mean = exp["summary"][split][f"{metric}_mean"]
                std = exp["summary"][split][f"{metric}_std"]
                lines.append(f"  {metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")
            lines.append("  Matriz de confusão agregada:")
            matrix = exp["summary"][split]["confusion_matrix_sum"]
            for row in matrix:
                lines.append(f"    {row}")
        lines.append("")
        lines.append(
            f"Tempo médio por execução selecionada: {exp['summary']['time_seconds_mean']:.2f}s "
            f"± {exp['summary']['time_seconds_std']:.2f}s"
        )
        lines.append("Melhores parâmetros por execução:")
        for r in exp["runs"]:
            lines.append(
                f"  run={r['run']:02d}, seed={r['random_state']}, "
                f"F1_val={r['val']['f1']:.4f}, F1_test={r['test']['f1']:.4f}, "
                f"params={r['params_text']}"
            )
        lines.append("")

    lines.append("Arquivos complementares gerados:")
    lines.append(f"- {output_csv}")
    lines.append(f"- {output_all_params_csv}")
    lines.append(f"- {output_json}")
    output_txt.write_text("\n".join(lines), encoding="utf-8")


def save_global_summary(experiments: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows(experiments)).to_csv(output_csv, index=False)
