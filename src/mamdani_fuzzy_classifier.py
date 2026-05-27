# Carrega dataset, aplica Mamdani, resultados_mamdani_fuzzy.txt

from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

class MamdaniFuzzyClassifier:
    def __init__(self, n_membership_functions: int = 3, random_state: int = 42):
        self.n_mf = n_membership_functions
        self.random_state = random_state
        self.membership_params = None  # Parâmetros das funções de pertinência
        self.class_rules = None  # Regras fuzzy por classe        
    def _triangular_mf(self, x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    def _create_membership_functions(self, X_train):
        self.membership_params = {}
        for feature_idx in range(X_train.shape[1]):
            feature_values = X_train[:, feature_idx]
            min_val = feature_values.min()
            max_val = feature_values.max()
            range_val = max_val - min_val + 1e-6
            params = []
            for i in range(self.n_mf):
                a = min_val + (range_val * i / (self.n_mf - 1)) if self.n_mf > 1 else min_val
                b = min_val + (range_val * (i + 0.5) / self.n_mf)
                c = min_val + (range_val * (i + 1) / (self.n_mf - 1)) if self.n_mf > 1 else max_val
                params.append((a, b, c))
            self.membership_params[feature_idx] = params
    def _fuzzify(self, x):
        fuzzified = []
        for feature_idx, feature_val in enumerate(x):
            mf_values = []
            for a, b, c in self.membership_params[feature_idx]:
                mf_val = self._triangular_mf(feature_val, a, b, c)
                mf_values.append(mf_val)
            fuzzified.extend(mf_values)
        return np.array(fuzzified)
    def _defuzzify(self, fuzzy_rules):
        if len(fuzzy_rules) == 0 or np.sum(fuzzy_rules) == 0:
            return 0.0
        return np.average(np.arange(len(fuzzy_rules)), weights=fuzzy_rules + 1e-10)
    def fit(self, X_train, y_train):
        self._create_membership_functions(X_train)
        # Armazenar classe para cada amostra fuzificada (simples estratégia de regras)
        self.X_train_fuzzified = np.array([self._fuzzify(x) for x in X_train])
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            x_fuzzified = self._fuzzify(x)
            similarities = []
            for x_train_fuzz in self.X_train_fuzzified:
                # mínimo
                sim = np.min(np.maximum(x_fuzzified, x_train_fuzz))
                similarities.append(sim)
            similarities = np.array(similarities)
            # Se todas as similaridades são muito baixas, usar vizinho mais próximo
            if np.max(similarities) < 0.1:
                distances = np.linalg.norm(self.X_train_fuzzified - x_fuzzified, axis=1)
                nearest_idx = np.argmin(distances)
                pred = self.y_train[nearest_idx]
            else:
                # Predizer a classe do vizinho mais similar
                nearest_idx = np.argmax(similarities)
                pred = self.y_train[nearest_idx]
            predictions.append(pred)
        return np.array(predictions)

def load_dataset(dataset_path: Path):
    X_train = np.load(dataset_path / "X_train.npy")
    X_val = np.load(dataset_path / "X_val.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_train = np.load(dataset_path / "y_train.npy")
    y_val = np.load(dataset_path / "y_val.npy")
    y_test = np.load(dataset_path / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate_mamdani(dataset_name: str, dataset_path: Path):
    print(f"\n[Treinando Fuzzy Mamdani para {dataset_name}]")
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(dataset_path)
    # Para datasets muito grandes, usar subsample
    if X_train.shape[0] > 500:
        indices = np.random.choice(X_train.shape[0], size=500, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"  (Dataset reduzido para 500 amostras para treinamento mais rápido)")
    classifier = MamdaniFuzzyClassifier(n_membership_functions=3, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred_train = classifier.predict(X_train)
    y_pred_val = classifier.predict(X_val)
    y_pred_test = classifier.predict(X_test)
    results = {
        "dataset": dataset_name,
        "algoritmo": "Sistema Fuzzy de Mamdani",
        "train": {
            "accuracy": float(accuracy_score(y_train, y_pred_train)),
            "precision": float(precision_score(y_train, y_pred_train, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_train, y_pred_train, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0))
        },
        "val": {
            "accuracy": float(accuracy_score(y_val, y_pred_val)),
            "precision": float(precision_score(y_val, y_pred_val, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_val, y_pred_val, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_val, y_pred_val, average='weighted', zero_division=0))
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "precision": float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0))
        }
    }
    return results

def main():
    datasets_root = Path("datasets/processed")
    datasets = ["adult", "bank_marketing", "heart_disease", "mushroom"]
    all_results = []
    for dataset_name in datasets:
        dataset_path = datasets_root / dataset_name
        if dataset_path.exists():
            result = train_and_evaluate_mamdani(dataset_name, dataset_path)
            all_results.append(result)
        else:
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
    output_file = Path("resultados/resultados_mamdani_fuzzy.txt")
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("RESULTADOS - SISTEMA FUZZY DE MAMDANI\n")
        f.write("="*70 + "\n\n")
        for result in all_results:
            f.write(f"Dataset: {result['dataset'].upper()}\n")
            f.write(f"Algoritmo: {result['algoritmo']}\n")
            f.write("-"*70 + "\n")
            for split in ["train", "val", "test"]:
                metrics = result[split]
                f.write(f"\n{split.upper()}:\n")
                f.write(f"-> Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"-> Precision: {metrics['precision']:.4f}\n")
                f.write(f"-> Recall:    {metrics['recall']:.4f}\n")
                f.write(f"-> F1-Score:  {metrics['f1']:.4f}\n")
            f.write("\n" + "="*70 + "\n\n")
    print(f"\nResultados salvos em {output_file}")
if __name__ == "__main__":
    main()
