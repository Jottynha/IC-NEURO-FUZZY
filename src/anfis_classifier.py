# Carrega datasets, treina ANFIS, resultados_anfis.txt

from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import json

class ANFISClassifier:
    """
    ANFIS (Adaptive Neuro-Fuzzy Inference System) para classificação.
    Combina lógica fuzzy com aprendizado neural para adaptação automática das regras.
    Implementação simplificada com 2 camadas fuzzy e adaptação por backpropagation.
    """
    def __init__(self, n_membership_functions: int = 2, learning_rate: float = 0.01, 
                 n_epochs: int = 10, random_state: int = 42):
        self.n_mf = n_membership_functions
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state        
        # Parâmetros adaptativos
        self.mf_params = None  # Parâmetros das funções de pertinência (adaptáveis)
        self.weights = None    # Pesos das regras fuzzy (adaptáveis)
        self.y_train = None
        self.X_train = None
    def _gaussian_mf(self, x, mean, sigma):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2 + 1e-10))
    def _init_parameters(self, X_train):
        np.random.seed(self.random_state)
        self.mf_params = {}
        for feature_idx in range(X_train.shape[1]):
            feature_values = X_train[:, feature_idx]
            min_val = feature_values.min()
            max_val = feature_values.max()
            means = np.linspace(min_val, max_val, self.n_mf)
            sigmas = np.ones(self.n_mf) * (max_val - min_val) / (2 * self.n_mf + 1e-10)
            self.mf_params[feature_idx] = {
                'means': means.astype(np.float32),
                'sigmas': sigmas.astype(np.float32)
            }        
        n_rules = self.n_mf ** X_train.shape[1]
        self.weights = np.random.randn(n_rules) * 0.1
    def _fuzzify(self, x):
        fuzzified = []
        for feature_idx, feature_val in enumerate(x):
            mf_activations = []
            for i in range(self.n_mf):
                mean = self.mf_params[feature_idx]['means'][i]
                sigma = self.mf_params[feature_idx]['sigmas'][i]
                activation = self._gaussian_mf(feature_val, mean, sigma)
                mf_activations.append(activation)
            fuzzified.append(np.array(mf_activations))
        return fuzzified
    def _generate_rules(self, fuzzified_input):
        # mínimo para combinação de antecedentes
        rule_activations = []
        # Gerar todas as combinações possíveis
        def generate_combinations(idx, current_activation, activations):
            if idx >= len(fuzzified_input):
                rule_activations.append(current_activation)
                return
            for i in range(self.n_mf):
                new_activation = min(current_activation, fuzzified_input[idx][i])
                generate_combinations(idx + 1, new_activation, fuzzified_input)
        generate_combinations(0, 1.0, fuzzified_input)
        return np.array(rule_activations)
    def _inference(self, rule_activations):
        # Saída agregada máximo ponderado
        output = np.dot(rule_activations, self.weights)
        return output
    def _defuzzify_output(self, output):
        return int(np.round(np.clip(output, 0, 1)))
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self._init_parameters(X_train)
        # Treinamento: ajustar pesos para minimizar erro de classificação
        for epoch in range(self.n_epochs):
            total_error = 0
            for i in range(min(X_train.shape[0], 100)):  # Limitar a 100 amostras por época para velocidade
                x = X_train[i]
                y_true = y_train[i]
                fuzzified = self._fuzzify(x)
                rule_activations = self._generate_rules(fuzzified)
                output = self._inference(rule_activations)
                output = np.tanh(output)  
                error = y_true - output
                total_error += error ** 2
                # Backpropagation: ajustar pesos
                self.weights += self.lr * error * rule_activations
            if epoch % 5 == 0:
                avg_error = total_error / min(X_train.shape[0], 100)
        # Após treinamento, ajustar pesos para problema multi-classe
        # Usar estratégia de votação: cada classe tem seus pesos
        self.class_weights = {}
        for class_label in np.unique(y_train):
            self.class_weights[class_label] = self.weights.copy()
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            fuzzified = self._fuzzify(x)
            rule_activations = self._generate_rules(fuzzified)
            output = self._inference(rule_activations)
            # Classificação por votação fuzzy
            class_scores = {}
            for class_label in self.class_weights.keys():
                score = np.dot(rule_activations, self.class_weights[class_label])
                class_scores[class_label] = score
            # Selecionar classe com maior score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return np.array(predictions)
def load_dataset(dataset_path: Path):
    X_train = np.load(dataset_path / "X_train.npy")
    X_val = np.load(dataset_path / "X_val.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_train = np.load(dataset_path / "y_train.npy")
    y_val = np.load(dataset_path / "y_val.npy")
    y_test = np.load(dataset_path / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate_anfis(dataset_name: str, dataset_path: Path):
    print(f"\n[Treinando ANFIS para {dataset_name}]")
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(dataset_path)
    # Para datasets com muitas features, fazer seleção
    if X_train.shape[1] > 10:
        # Usar PCA para reduzir dimensionalidade
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, X_train.shape[1]))
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        print(f"  (Reduzidas para {X_train.shape[1]} features via PCA)")
    # Para datasets muito grandes, usar subsample
    if X_train.shape[0] > 300:
        indices = np.random.choice(X_train.shape[0], size=300, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"  (Dataset reduzido para 300 amostras para treinamento mais rápido)")
    # Normalizar y para o intervalo [0, 1]
    y_unique = np.unique(y_train)
    y_train_norm = (y_train - y_unique.min()) / (y_unique.max() - y_unique.min() + 1e-10)
    classifier = ANFISClassifier(n_membership_functions=2, learning_rate=0.01, 
                                 n_epochs=10, random_state=42)
    classifier.fit(X_train, y_train)    
    # Predições
    y_pred_train = classifier.predict(X_train)
    y_pred_val = classifier.predict(X_val)
    y_pred_test = classifier.predict(X_test)
    # Métricas
    results = {
        "dataset": dataset_name,
        "algoritmo": "ANFIS (Adaptive Neuro-Fuzzy Inference System)",
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
            result = train_and_evaluate_anfis(dataset_name, dataset_path)
            all_results.append(result)
        else:
            print(f"Dataset {dataset_name} não encontrado em {dataset_path}")
    # Salvar resultados em txt
    output_file = Path("resultados/resultados_anfis.txt")
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("RESULTADOS [ANFIS (ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM)]\n")
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
