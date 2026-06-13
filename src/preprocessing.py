"""Pré-processamento dos datasets UCI usados no experimento.

Gera splits 60/20/20, aplica one-hot em categorias e StandardScaler.
Salva arrays numpy em `datasets/processed/<dataset>/`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo


DATASETS = {
    "adult": 2,
    "bank_marketing": 222,
    "heart_disease": 45,
    "mushroom": 73,
}


def parse_dataset_argument(value: str | None) -> list[str]:
    """Recebe None, um dataset ou uma lista separada por vírgula."""
    if value is None:
        return list(DATASETS.keys())

    requested = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in requested if item not in DATASETS]
    if invalid:
        valid = ", ".join(DATASETS.keys())
        raise ValueError(f"Dataset(s) inválido(s): {', '.join(invalid)}. Válidos: {valid}")
    return requested


def maybe_sample_dataset(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    sample_fraction: float | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series | np.ndarray]:
    """Aplica amostragem estratificada opcional antes do split treino/val/test."""
    if sample_fraction is None or sample_fraction >= 1.0:
        return X, y

    if sample_fraction <= 0:
        raise ValueError("--sample-fraction precisa ser maior que 0 e menor ou igual a 1.")

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_fraction,
        random_state=random_state,
        stratify=y,
    )
    return X_sample, y_sample


def preprocess_dataset(
    name: str,
    dataset_id: int,
    output_root: Path,
    random_state: int = 42,
    sample_fraction: float | None = None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Processando: {name.upper()}")
    print(f"{'=' * 60}")

    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy().iloc[:, 0]

    # Corrige variações textuais como '<=50K' e '<=50K.' na base Adult.
    if y.dtype == "object":
        y = y.astype(str).str.strip().str.rstrip(".")

    print(f"Original: {X.shape[0]} amostras, {X.shape[1]} atributos")

    X, y = maybe_sample_dataset(X, y, sample_fraction, random_state)
    if sample_fraction is not None and sample_fraction < 1.0:
        print(f"Após amostragem: {X.shape[0]} amostras ({sample_fraction:.2%})")

    # Tratamento de valores faltantes e codificação das variáveis explicativas.
    # Evita chained assignment e deixa o preenchimento compatível com pandas 3.x.
    X = X.replace("?", np.nan)
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                fill_value = X[col].median()
            else:
                mode_values = X[col].mode(dropna=True)
                fill_value = mode_values.iloc[0] if not mode_values.empty else ""
            X.loc[:, col] = X[col].fillna(fill_value)
    X = pd.get_dummies(X, drop_first=True)

    # Codificação da variável-alvo.
    if getattr(y, "dtype", None) == "object":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    else:
        y = np.asarray(y, dtype=int)

    X = np.asarray(X, dtype=np.float32)

    print(f"Após pré-proc: {X.shape[0]} amostras, {X.shape[1]} atributos")

    # Split estratificado 60/20/20.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    print(f"Split: treino {X_train.shape[0]}, val {X_val.shape[0]}, test {X_test.shape[0]}")

    # Normalização com ajuste apenas no treino, para evitar vazamento de informação.
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

    meta = {
        "dataset": name,
        "uci_id": dataset_id,
        "n_samples": int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(np.unique(y).shape[0]),
        "sample_fraction": sample_fraction,
        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Salvo em: {dataset_dir}")


def write_exploratory_analysis(output_root: Path, selected_datasets: list[str]) -> None:
    """Gera uma análise exploratória simples dos datasets processados."""
    results_dir = Path("resultados")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name in selected_datasets:
        ds_dir = output_root / name
        if not ds_dir.exists():
            print(f"Pular {name}: não encontrado em {ds_dir}")
            continue

        try:
            X_train = np.load(ds_dir / "X_train.npy")
            y_train = np.load(ds_dir / "y_train.npy")
            X_val = np.load(ds_dir / "X_val.npy")
            y_val = np.load(ds_dir / "y_val.npy")
            X_test = np.load(ds_dir / "X_test.npy")
            y_test = np.load(ds_dir / "y_test.npy")
        except Exception as e:
            print(f"Erro carregando arrays de {name}: {e}")
            continue

        lines = []
        lines.append("=" * 60)
        lines.append(f"Análise exploratória - {name.upper()}")
        lines.append("=" * 60)

        meta_file = ds_dir / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                lines.append(f"Descrição: {meta.get('dataset', '')}")
                lines.append(f"UCI id: {meta.get('uci_id', '')}")
                lines.append(f"Fração amostral: {meta.get('sample_fraction', '')}")
            except Exception:
                pass

        total_samples = int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])
        lines.append(
            f"Amostras (treino/val/test): "
            f"{X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]} "
            f"(total {total_samples})"
        )
        lines.append(f"Atributos (features): {X_train.shape[1]}")

        classes, counts = np.unique(
            np.concatenate([y_train, y_val, y_test]),
            return_counts=True,
        )
        lines.append(f"Classes: {classes.tolist()}")
        lines.append(f"Contagem por classe: {counts.tolist()}")

        props = (counts / counts.sum()).tolist()
        lines.append(f"Proporção por classe: {[round(p, 4) for p in props]}")

        n_features = X_train.shape[1]
        show_n = min(10, n_features)
        lines.append(f"Estatísticas (train) - primeiras {show_n} features:")

        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        mins = np.min(X_train, axis=0)
        maxs = np.max(X_train, axis=0)

        for i in range(show_n):
            lines.append(
                f"  feat_{i}: "
                f"mean={means[i]:.4f}, std={stds[i]:.4f}, "
                f"min={mins[i]:.4f}, max={maxs[i]:.4f}"
            )

        majority_prop = max(props)
        lines.append(f"Maioria de classe (proporção): {majority_prop:.4f}")

        txt_path = results_dir / f"exploratory_{name}.txt"
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Análise exploratória salva em: {txt_path}")

    print("\nAnálises exploratórias concluídas.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-processa bases de classificação do UCI")
    parser.add_argument("--dataset", default=None, help="Dataset único ou lista separada por vírgula")
    parser.add_argument("--output-dir", default="datasets/processed", help="Diretório de saída")
    parser.add_argument("--random-state", type=int, default=42, help="Semente para reprodutibilidade")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Fração amostral opcional. Ex.: 0.3 usa 30 por cento da base.",
    )
    parser.add_argument(
        "--skip-exploratory",
        action="store_true",
        help="Pula a geração dos arquivos de análise exploratória.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    selected_datasets = parse_dataset_argument(args.dataset)

    for name in selected_datasets:
        preprocess_dataset(
            name=name,
            dataset_id=DATASETS[name],
            output_root=output_root,
            random_state=args.random_state,
            sample_fraction=args.sample_fraction,
        )

    print(f"\n{'=' * 60}")
    print(f"Pré-processamento concluído para: {', '.join(selected_datasets)}")
    print(f"{'=' * 60}\n")

    if not args.skip_exploratory:
        write_exploratory_analysis(output_root, selected_datasets)


if __name__ == "__main__":
    main()
