# Pré-processamento de 4 bases de classificação do UCI:
#1) Adult (id=2)
#2) Bank Marketing (id=222)
#3) Heart Disease (id=45)
#4) Mushroom (id=73)
# O script realiza:
# -> Carregamento das bases via ucimlrepo
# -> Tratamento de valores faltantes (forward fill, depois modo)
# -> One-hot encoding de variáveis categóricas
# -> Normalização com StandardScaler (fit em treino)
# -> Split estratificado 60/20/20 (treino/validação/teste)
# -> Salvamento em 'datasets/processed/<dataset>/'

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo
def preprocess_dataset(
    name: str,
    dataset_id: int,
    output_root: Path,
    random_state: int = 42,
) -> None:
    print(f"\n{'='*60}")
    print(f"Processando: {name.upper()}")
    print(f"{'='*60}")
    # Carregamento
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy().iloc[:, 0]
    print(f"Original: {X.shape[0]} amostras, {X.shape[1]} atributos")
    print(f"Classes: {y.nunique()}")
    # Tratamento de valores faltantes
    X = X.ffill().fillna(X.mode().iloc[0])
    # Codificando variáveis categóricas
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    else:
        y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=np.float32)
    print(f"Após pré-proc: {X.shape[0]} amostras, {X.shape[1]} atributos")
    # Split estratificado 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    print(f"Split: treino {X_train.shape[0]}, val {X_val.shape[0]}, test {X_test.shape[0]}")
    # Normalização (fit em treino)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    dataset_dir = output_root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "X_train.npy", X_train)
    np.save(dataset_dir / "X_val.npy", X_val)
    np.save(dataset_dir / "X_test.npy", X_test)
    np.save(dataset_dir / "y_train.npy", y_train)
    np.save(dataset_dir / "y_val.npy", y_val)
    np.save(dataset_dir / "y_test.npy", y_test)
    np.save(dataset_dir / "scaler_mean.npy", scaler.mean_)
    np.save(dataset_dir / "scaler_scale.npy", scaler.scale_)
    # Metadados
    meta = {
        "dataset": name,
        "uci_id": dataset_id,
        "n_samples": int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(np.unique(y).shape[0]),
        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Salvo em: {dataset_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pré-processa 4 bases de classificação do UCI"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/processed",
        help="Diretório de saída",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente para reprodutibilidade",
    )
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    datasets = {
        "adult": 2,
        "bank_marketing": 222,
        "heart_disease": 45,
        "mushroom": 73,
    }
    for name, dataset_id in datasets.items():
        preprocess_dataset(name, dataset_id, output_root, args.random_state)
    print(f"\n{'='*60}")
    print("Pré-processamento concluído para todas as 4 bases")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

