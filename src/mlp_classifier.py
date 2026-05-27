# MLP classifier
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_dataset(dataset_path: Path):
    X_train = np.load(dataset_path / "X_train.npy")
    X_val = np.load(dataset_path / "X_val.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_train = np.load(dataset_path / "y_train.npy")
    y_val = np.load(dataset_path / "y_val.npy")
    y_test = np.load(dataset_path / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate_mlp(dataset_name: str, dataset_path: Path):
    print(f"\n[Treinando MLP para {dataset_name}]")
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(dataset_path)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42,
                        early_stopping=True, validation_fraction=0.2, n_iter_no_change=50)
    mlp.fit(X_train, y_train)
    y_pred_train = mlp.predict(X_train)
    y_pred_val = mlp.predict(X_val)
    y_pred_test = mlp.predict(X_test)
    results = {
        "dataset": dataset_name,
        "algoritmo": "MLP (Perceptron Multicamadas)",
        "train": {"accuracy": float(accuracy_score(y_train, y_pred_train)),
                  "precision": float(precision_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                  "recall": float(recall_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                  "f1": float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0))},
        "val": {"accuracy": float(accuracy_score(y_val, y_pred_val)),
                "precision": float(precision_score(y_val, y_pred_val, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_val, y_pred_val, average='weighted', zero_division=0)),
                "f1": float(f1_score(y_val, y_pred_val, average='weighted', zero_division=0))},
        "test": {"accuracy": float(accuracy_score(y_test, y_pred_test)),
                 "precision": float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                 "recall": float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                 "f1": float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0))}
    }
    return results

def main():
    datasets_root = Path("datasets/processed")
    datasets = ["adult", "bank_marketing", "heart_disease", "mushroom"]
    all_results = []
    for dataset_name in datasets:
        dataset_path = datasets_root / dataset_name
        if dataset_path.exists():
            result = train_and_evaluate_mlp(dataset_name, dataset_path)
            all_results.append(result)
        else:
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
    output_file = Path("resultados/resultados_mlp.txt")
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("RESULTADOS - MLP (PERCEPTRON MULTICAMADAS)\n")
        f.write("="*70 + "\n\n")
        for result in all_results:
            f.write(f"Dataset: {result['dataset'].upper()}\n")
            f.write(f"Algoritmo: {result['algoritmo']}\n")
            f.write("-"*70 + "\n")
            for split in ["train", "val", "test"]:
                metrics = result[split]
                f.write(f"\n{split.upper()}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
            f.write("\n" + "="*70 + "\n\n")
    print(f"\nResultados salvos em {output_file}")

if __name__ == "__main__":
    main()
