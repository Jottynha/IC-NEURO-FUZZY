from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATASETS = ["adult", "bank_marketing", "heart_disease", "mushroom"]
ROOT = Path('datasets/processed')
RESULTS = Path('analise_exploratoria')
PLOTS = RESULTS / 'plots'
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
LITERATURE = {
    'mlp': 'Bishop, C. M. (1995). Neural Networks for Pattern Recognition.',
    'rbm': 'Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence (RBMs).',
    'mamdani': 'Mamdani, E. H. (1975). Application of fuzzy algorithms for control of simple dynamic plant.',
    'anfis': 'Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system.'
}
EXPECTATIONS_TEMPLATE = {
    'adult': (
        "Adult: alto número de features (one-hot), mistura contínuo/categórica e desbalanceamento.\n"
        "Expectativa: MLP bem ajustado pode performar razoavelmente; RBM pode ajudar na representação; "
        "Mamdani/ANFIS precisam de redução dimensional e seleção de features para generalizar.\n"
    ),
    'bank_marketing': (
        "Bank Marketing: problema de negócio binário, dados estruturados, desbalanceado.\n"
        "Expectativa: MLP e modelos lineares ou RBM+Logistic tendem a performar bem; fuzzy/ANFIS podem fornecer interpretabilidade.\n"
    ),
    'heart_disease': (
        "Heart Disease: conjunto pequeno (~300 amostras), poucas features.\n"
        "Expectativa: Algoritmos complexos (muitos parâmetros) podem overfit; preferir modelos simples/regularizados e validação robusta.\n"
    ),
    'mushroom': (
        "Mushroom: relativamente fácil, detectável com regras simples; balanceado.\n"
        "Expectativa: Todos os métodos (MLP, RBM+Log, Mamdani) devem alcançar alta acurácia; ANFIS também se beneficiar de regras.\n"
    )
}

def plot_class_distribution(y, out_path: Path):
    classes, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([str(c) for c in classes], counts, color='C0')
    ax.set_title('Distribuição de classes')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Contagem')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
def plot_feature_correlation(X, out_path: Path, n_features: int = 15):
    # Plota matriz de correlação das top N features com variância mais alta
    from numpy.linalg import svd
    n_features = min(n_features, X.shape[1])  # Não exceder número de features
    # Selecionar features com maior variância
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[::-1][:n_features]
    X_subset = X[:, top_indices]
    corr = np.corrcoef(X_subset.T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels([f'f{top_indices[i]}' for i in range(n_features)], rotation=45, ha='right')
    ax.set_yticklabels([f'f{top_indices[i]}' for i in range(n_features)])
    ax.set_title(f'Correlação entre top {n_features} features (maior variância)')
    fig.colorbar(im, ax=ax, label='Correlação')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
def plot_pca_scree(X, out_path: Path, n_components: int = 20):
    n_components = min(n_components, X.shape[1], X.shape[0])  # Não pode exceder min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(1, len(evr)+1), evr, '-o')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Explained variance ratio')
    ax.set_title('Scree plot (PCA)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
def analyze_dataset(name: str):
    ds_dir = ROOT / name
    if not ds_dir.exists():
        print(f'Pular {name}: pasta não existe ({ds_dir})')
        return
    X_train = np.load(ds_dir / 'X_train.npy')
    y_train = np.load(ds_dir / 'y_train.npy')
    X_val = np.load(ds_dir / 'X_val.npy')
    y_val = np.load(ds_dir / 'y_val.npy')
    X_test = np.load(ds_dir / 'X_test.npy')
    y_test = np.load(ds_dir / 'y_test.npy')
    out_plot_dir = PLOTS / name
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(np.concatenate([y_train, y_val, y_test]), out_plot_dir / 'class_distribution.png')
    plot_feature_correlation(X_train, out_plot_dir / 'feature_correlation.png', n_features=15)
    plot_pca_scree(X_train, out_plot_dir / 'pca_scree.png', n_components=20)
    meta_file = ds_dir / 'metadata.json'
    meta = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
    lines = []
    lines.append('='*80)
    lines.append(f'Análise exploratória detalhada - {name.upper()}')
    lines.append('='*80)
    lines.append('')
    lines.append('Metadados:')
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
    lines.append('')
    n_total = int(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])
    lines.append(f'Amostras (treino/val/test): {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]} (total={n_total})')
    lines.append(f'Número de features: {X_train.shape[1]}')
    classes, counts = np.unique(np.concatenate([y_train, y_val, y_test]), return_counts=True)
    props = counts / counts.sum()
    lines.append(f'Classes encontradas: {classes.tolist()}')
    lines.append(f'Contagens por classe: {counts.tolist()}')
    lines.append(f'Proporções por classe: {[round(float(p),4) for p in props]}')
    lines.append('')
    lines.append('Estatísticas (primeiras 10 features do treino):')
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    mins = np.min(X_train, axis=0)
    maxs = np.max(X_train, axis=0)
    show_n = min(10, X_train.shape[1])
    for i in range(show_n):
        lines.append(f'  feat_{i}: mean={means[i]:.4f}, std={stds[i]:.4f}, min={mins[i]:.4f}, max={maxs[i]:.4f}')
    lines.append('')
    lines.append('Gráficos gerados:')
    lines.append(f' - {out_plot_dir / "class_distribution.png"}')
    lines.append(f' - {out_plot_dir / "feature_correlation.png"}')
    lines.append(f' - {out_plot_dir / "pca_scree.png"}')
    lines.append('')
    lines.append('O que esperar de cada algoritmo (breve, com referências):')
    lines.append('')
    lines.append('MLP: ' + LITERATURE['mlp'])
    lines.append('RBM: ' + LITERATURE['rbm'])
    lines.append('Mamdani: ' + LITERATURE['mamdani'])
    lines.append('ANFIS: ' + LITERATURE['anfis'])
    lines.append('')
    lines.append('Observações específicas para este dataset:')
    lines.append(EXPECTATIONS_TEMPLATE.get(name, ''))
    out_txt = RESULTS / f'exploratory_{name}_detailed.txt'
    out_txt.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Gerado: {out_txt} e plots em {out_plot_dir}')
if __name__ == '__main__':
    for ds in DATASETS:
        analyze_dataset(ds)
    print('\nAnálise exploratória detalhada concluída.')
