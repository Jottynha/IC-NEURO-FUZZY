#!/usr/bin/env python3
"""Gera gráficos resumo a partir dos arquivos de resultados em `resultados[2]/`.

Exemplos de uso:

python3 src/plot_results.py --results-dir resultados[2] --out-dir imgs/plots_results
python3 src/plot_results.py --results-dir resultados[2] --out-dir imgs/plots_results --show

O script tenta carregar arquivos `resumo_*.csv` e arquivos `resumo_*_otimizado.csv` e agrupar
as métricas por `dataset` e `algorithm` (inferido a partir do nome do arquivo).
Se existirem CSVs por-seed, também tenta gerar boxplots.
"""
import argparse
import os
import glob
import re
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def infer_algorithm_from_path(path):
    name = path.lower()
    for alg in ["mlp", "rbm", "mamdani", "mamdani_fuzzy", "anfis", "rbm_logistic"]:
        if alg in name:
            return alg.replace("_fuzzy", "")
    # fallback: try regex resumo_<alg>
    m = re.search(r"resumo_([a-z0-9]+)", os.path.basename(name))
    if m:
        return m.group(1)
    return "unknown"


def collect_summary_csvs(results_dir):
    # Use os.walk + fnmatch to avoid issues with special characters (e.g. colchetes) in paths
    files = []
    for root, _, filenames in os.walk(results_dir):
        for name in filenames:
            if fnmatch.fnmatch(name.lower(), "resumo*.csv"):
                files.append(os.path.join(root, name))
    files = sorted(list(set(files)))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            alg = infer_algorithm_from_path(f)
            df["__source_file"] = os.path.basename(f)
            df["algorithm"] = alg
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True, sort=False)


def plot_bar_by_dataset(df, metric, out_dir, show=False):
    os.makedirs(out_dir, exist_ok=True)
    # Grouped bar: x = dataset, hue = algorithm
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    algs = list(df["algorithm"].unique())
    palette = sns.color_palette("tab10", n_colors=max(1, len(algs)))
    order = sorted(df["dataset"].unique())
    sns.barplot(data=df, x="dataset", y=metric, hue="algorithm", errorbar=None, palette=palette)
    plt.title(f"{metric} por dataset e algoritmo")
    plt.ylabel(metric)
    plt.xlabel("Dataset")
    plt.legend(title="Algoritmo", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    out = os.path.join(out_dir, f"{metric}_by_dataset.png")
    plt.savefig(out, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_time_vs_f1(df, out_dir, show=False):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(9, 6))
    if "time_seconds_mean" in df.columns and "test_f1_mean" in df.columns:
        sns.set_theme(style="whitegrid")
        algs = list(df["algorithm"].unique())
        palette = sns.color_palette("tab10", n_colors=max(1, len(algs)))
        ax = sns.scatterplot(data=df, x="time_seconds_mean", y="test_f1_mean", hue="algorithm", style="dataset", s=120, palette=palette)
        for i, row in df.iterrows():
            ax.text(row["time_seconds_mean"] * 1.05, row["test_f1_mean"], f"{row['algorithm']}\n{row['dataset']}", fontsize=8)
        ax.set_xscale('log')
        plt.xlabel("Tempo médio (s) [escala log]")
        plt.ylabel("Test F1 (média)")
        plt.title("Tempo x F1 por algoritmo e dataset")
        plt.tight_layout()
        out = os.path.join(out_dir, "time_vs_test_f1.png")
        plt.savefig(out, dpi=200)
        if show:
            plt.show()
        plt.close()


def try_boxplots_from_detailed(results_dir, out_dir, show=False):
    # Procura por arquivos CSV que contenham coluna 'test_f1' e uma coluna indicativa de seed (seed|random_state)
    candidates = []
    for root, _, filenames in os.walk(results_dir):
        for name in filenames:
            if name.lower().endswith('.csv'):
                candidates.append(os.path.join(root, name))
    rows = []
    for c in candidates:
        try:
            d = pd.read_csv(c)
            cols = [cname.lower() for cname in d.columns]
            if any('test_f1' in cname for cname in cols) and any(k in cols for k in ('seed', 'random_state')):
                metric_col = [cname for cname in d.columns if 'test_f1' in cname.lower()][0]
                alg = infer_algorithm_from_path(c)
                # tenta inferir dataset do caminho
                ds_name = os.path.basename(c).split('_')[0]
                d = d.copy()
                d['algorithm'] = alg
                d['dataset'] = ds_name
                d['__metric_col'] = metric_col
                rows.append(d)
        except Exception:
            continue
    if not rows:
        return False
    combined = pd.concat(rows, ignore_index=True, sort=False)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style='whitegrid')
    algs = list(combined['algorithm'].unique())
    palette = sns.color_palette('tab10', n_colors=max(1, len(algs)))
    metric_col = combined['__metric_col'].iloc[0]
    sns.boxplot(data=combined, x='algorithm', y=metric_col, palette=palette)
    plt.title('Distribuição de Test F1 por algoritmo (por-seed)')
    plt.tight_layout()
    out = os.path.join(out_dir, 'boxplot_test_f1_by_algorithm.png')
    plt.savefig(out, dpi=200)
    if show:
        plt.show()
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="resultados[2]", help="Diretório raiz dos resultados (padrão: resultados[2])")
    parser.add_argument("--out-dir", default="imgs/plots_results", help="Diretório de saída para gráficos")
    parser.add_argument("--show", action="store_true", help="Mostrar os gráficos na tela")
    args = parser.parse_args()

    df = collect_summary_csvs(args.results_dir)
    if df is None or df.empty:
        print("Nenhum arquivo resumo encontrado em:", args.results_dir)
        return

    # Normalizar nomes de colunas esperadas
    # Alguns arquivos usam val_f1_mean/test_f1_mean
    if "test_f1_mean" not in df.columns and "test_f1" in df.columns:
        df = df.rename(columns={"test_f1": "test_f1_mean"})
    if "val_f1_mean" not in df.columns and "val_f1" in df.columns:
        df = df.rename(columns={"val_f1": "val_f1_mean"})

    out_dir = args.out_dir
    # Plots: bar por dataset (test_f1_mean e val_f1_mean)
    if "test_f1_mean" in df.columns:
        plot_bar_by_dataset(df, "test_f1_mean", out_dir, show=args.show)
    if "val_f1_mean" in df.columns:
        plot_bar_by_dataset(df, "val_f1_mean", out_dir, show=args.show)

    # Scatter tempo x f1
    plot_time_vs_f1(df, out_dir, show=args.show)

    # Tentar boxplots por-seed se houver dados
    ok = try_boxplots_from_detailed(args.results_dir, out_dir, show=args.show)
    if not ok:
        print("Nenhum arquivo por-seed/detailed encontrado — boxplots ignorados.")

    print("Gráficos salvos em:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
