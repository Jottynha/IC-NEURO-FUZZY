"""
Estudo sistemático de hiperparâmetros da MLP.
Testa cada parâmetro isoladamente com valores base fixos, mostrando impacto.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from experiment_utils import (
    DATASETS,
    load_dataset,
    evaluate_model,
    flatten_params,
)

# Valores base (conservadores e bem-comportados)
BASE_PARAMS = {
    "hidden_layer_sizes": (100,),
    "activation": "relu",
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "max_iter": 1000,
    "early_stopping": True,
    "validation_fraction": 0.2,
    "n_iter_no_change": 50,
    "batch_size": "auto",
}

# Seeds reduzidas para estudo rápido
STUDY_SEEDS = list(range(1, 6))  # 5 seeds apenas

# Variações a testar (um parâmetro por vez)
PARAMETER_STUDIES = {
    "hidden_layer_sizes": [
        (50,),
        (100,),
        (150,),
        (200,),
        (250,),
        (100, 50),
        (150, 75),
        (200, 100),
    ],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
    "alpha": [0.0, 0.00001, 0.0001, 0.001, 0.01],
}


def build_model_from_params(params: Dict[str, Any], random_state: int) -> MLPClassifier:
    """Constrói MLPClassifier com parâmetros dados."""
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        learning_rate_init=params["learning_rate_init"],
        alpha=params["alpha"],
        max_iter=params["max_iter"],
        random_state=random_state,
        early_stopping=params["early_stopping"],
        validation_fraction=params["validation_fraction"],
        n_iter_no_change=params["n_iter_no_change"],
        batch_size=params["batch_size"],
    )


def test_single_parameter(
    dataset_name: str,
    data: Tuple[np.ndarray, ...],
    param_name: str,
    param_values: List[Any],
) -> List[Dict[str, Any]]:
    """
    Testa um parâmetro variando seus valores, mantendo os demais fixos.
    Retorna lista de dicionários com resultados agregados.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = data
    labels = np.unique(np.concatenate([y_train, y_val, y_test])).tolist()
    results = []

    print(f"\n  Testando parâmetro: {param_name}")
    for param_value in param_values:
        print(f"    Valor: {param_value}...", end=" ", flush=True)
        
        # Criar parâmetros para este teste
        test_params = BASE_PARAMS.copy()
        test_params[param_name] = param_value
        
        # Agregar métricas sobre múltiplas seeds
        val_f1_scores = []
        test_f1_scores = []
        times = []
        
        for seed in STUDY_SEEDS:
            start = time.perf_counter()
            model = build_model_from_params(test_params, seed)
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, data, labels)
            elapsed = time.perf_counter() - start
            
            val_f1_scores.append(metrics["val"]["f1"])
            test_f1_scores.append(metrics["test"]["f1"])
            times.append(elapsed)
        
        # Agregar resultados
        result = {
            "dataset": dataset_name,
            "param_name": param_name,
            "param_value": str(param_value),
            "val_f1_mean": float(np.mean(val_f1_scores)),
            "val_f1_std": float(np.std(val_f1_scores, ddof=1)) if len(val_f1_scores) > 1 else 0.0,
            "test_f1_mean": float(np.mean(test_f1_scores)),
            "test_f1_std": float(np.std(test_f1_scores, ddof=1)) if len(test_f1_scores) > 1 else 0.0,
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "n_seeds": len(STUDY_SEEDS),
        }
        results.append(result)
        print(f"val_f1={result['val_f1_mean']:.4f}±{result['val_f1_std']:.4f}, test_f1={result['test_f1_mean']:.4f}")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudo sistemático de hiperparâmetros da MLP")
    parser.add_argument("--dataset", default="adult", help="Dataset para estudo (padrão: adult)")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    if dataset_name not in DATASETS:
        print(f"Dataset inválido: {dataset_name}")
        return
    
    # Carregar dados
    dataset_path = Path("datasets/processed") / dataset_name
    if not dataset_path.exists():
        print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
        return
    
    print(f"\n[Estudo de Hiperparâmetros - MLP no dataset {dataset_name}]")
    print(f"Valores base: {json.dumps({k: str(v) for k, v in BASE_PARAMS.items()}, ensure_ascii=False, indent=2)}")
    data = load_dataset(dataset_path)
    
    # Criar diretório de saída
    output_dir = Path("resultados[2]/mlp_hyperparameter_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Testar cada parâmetro
    all_results = []
    for param_name, param_values in PARAMETER_STUDIES.items():
        results = test_single_parameter(dataset_name, data, param_name, param_values)
        all_results.extend(results)
    
    # Salvar resultados em CSV para análise rápida
    df = pd.DataFrame(all_results)
    csv_path = output_dir / f"hyperparameter_study_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")
    
    # Gerar resumo por parâmetro
    summary = {}
    for param_name in PARAMETER_STUDIES.keys():
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
    
    summary_path = output_dir / f"summary_{dataset_name}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sumário salvo em: {summary_path}")
    
    # Exibir sumário
    print("\n" + "=" * 80)
    print("SUMÁRIO DE IMPACTOS POR PARÂMETRO")
    print("=" * 80)
    for param_name, info in summary.items():
        print(f"\n{param_name.upper()}:")
        print(f"  Melhor valor: {info['best_value']}")
        print(f"  Test F1: {info['best_test_f1']:.4f}")
        print(f"  Val F1: {info['best_val_f1']:.4f}")
        print(f"  Top 3:")
        for i, top in enumerate(info["top_3"], 1):
            print(f"    {i}. {top['value']:30s} → test_f1={top['test_f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("Análise completa. Próximo passo: combinar melhores valores de cada parâmetro.")
    print("=" * 80)


if __name__ == "__main__":
    main()
