<p align="center"> 
  <img src="imgs/logo_azul.png" alt="CEFET-MG" width="100px" height="100px">
</p>

<h1 align="center">
Redes Neurais Artificiais e Sistemas Neuro-Fuzzy
</h1>

<h3 align="center">
Avaliação e comparação de diferentes algoritmos de Inteligência Computacional em tarefas de classificação.
</h3>

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

</div>

---

<div align="justify">
<p><strong>Disciplina:</strong> Inteligência Computacional<br>
<strong>Instituição:</strong> Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG) - Campus V Divinópolis<br>
<strong>Professor:</strong> Alisson Marques da Silva<br>
<strong>Projeto:</strong> "Atividade 03"<br>
<strong>Alunos:</strong> Jader Oliveira Silva e João Pedro Rodrigues Silva <br>
</div>

## Visão geral

Este repositório implementa uma avaliação e comparação de algoritmos de **Redes Neurais Artificiais (RNAs)** e **Sistemas Neuro-Fuzzy** em tarefas de **classificação** utilizando 4 bases de dados reais do UCI Machine Learning Repository.

### Bases de Dados

| Base | ID UCI | Amostras | Atributos | Tarefa |
|------|--------|----------|-----------|--------|
| **Adult** | 2 | ~48,842 | 100* | Multiclasse (4 classes) |
| **Bank Marketing** | 222 | ~45,211 | 38 | Classificação binária desbalanceada |
| **Heart Disease** | 45 | 303 | 13 | Multiclasse (5 classes) pequeno dataset |
| **Mushroom** | 73 | 8,124 | 94* | Classificação binária balanceada |

*Após one-hot encoding de variáveis categóricas.

#### Justificativa das Bases

As quatro bases foram selecionadas para compor um conjunto experimental heterogêneo, variando em tamanho amostral, dimensionalidade, balanceamento e separabilidade. Essa composição permite avaliar o comportamento dos algoritmos sob cenários distintos de classificação tabular.

- **Adult:** base de grande porte, com alta dimensionalidade após codificação categórica e multiclasse desbalanceada. É adequada para examinar escalabilidade e robustez a heterogeneidade dos atributos.
- **Bank Marketing:** problema binário de grande porte e fortemente desbalanceado, útil para analisar sensibilidade a assimetria de classes em dados estruturados de negócio.
- **Heart Disease:** base pequena, com poucas observações e múltiplas classes, apropriada para investigar generalização em regime de escassez de dados e risco de sobreajuste.
- **Mushroom:** base balanceada e altamente estruturada, com padrões relativamente separáveis. Serve como cenário de alta previsibilidade para comparar o desempenho máximo dos métodos.

Em conjunto, essas bases cobrem diferentes combinações de complexidade, desbalanceamento e quantidade de dados, favorecendo uma comparação metodologicamente mais abrangente.

## Análise Exploratória

**Arquivo:** `src/exploratory_analysis.py`

Gera gráficos e relatórios detalhados para cada dataset:

```bash
python3 src/exploratory_analysis.py
```

Saída (em `analise_exploratoria/`):
- `exploratory_<dataset>_detailed.txt` - Relatório completo com estatísticas
- `plots/<dataset>/class_distribution.png` - Distribuição de classes
- `plots/<dataset>/feature_correlation.png` - Matriz de correlação (top 15 features)
- `plots/<dataset>/pca_scree.png` - Variância explicada (PCA)

Os relatórios sintetizam estatísticas descritivas e uma leitura preliminar da adequação de cada base aos algoritmos avaliados.

## Dependências

Instale os requisitos:

```bash
pip install ucimlrepo pandas numpy scikit-learn scipy matplotlib
```

## Algoritmos Avaliados

Quatro algoritmos de Inteligência Computacional foram implementados e comparados:

### 1. MLP (Perceptron Multicamadas)
- **Implementação:** `src/mlp_classifier.py`
- **Referência:** Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
- **Síntese:** rede feedforward treinada por retropropagação, capaz de modelar relações não lineares entre atributos.
- **Justificativa:** constitui um baseline neural supervisionado apropriado para dados tabulares.

### 2. RBM + Regressão Logística
- **Implementação:** `src/rbm_logistic_classifier.py`
- **Referência:** Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence.
- **Síntese:** combina extração não supervisionada de representações por RBM com um classificador linear na saída.
- **Justificativa:** permite avaliar se uma etapa de representação melhora a separabilidade das classes.

### 3. Sistema Fuzzy de Mamdani
- **Implementação:** `src/mamdani_fuzzy_classifier.py`
- **Referência:** Mamdani, E. H. (1975). Application of fuzzy algorithms for control of simple dynamic plant.
- **Síntese:** sistema baseado em regras linguísticas, com fuzzificação, inferência e defuzzificação.
- **Justificativa:** oferece interpretabilidade explícita, útil para comparar desempenho e transparência.

### 4. ANFIS (Adaptive Neuro-Fuzzy Inference System)
- **Implementação:** `src/anfis_classifier.py`
- **Referência:** Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system.
- **Síntese:** modelo neuro-fuzzy que ajusta parâmetros de pertinência e regras por aprendizagem adaptativa.
- **Justificativa:** combina interpretabilidade da lógica fuzzy com capacidade de ajuste de modelos neurais.

**Arquivo:** `src/preprocessing.py`

O script realiza para cada base:

1. **Carregamento** via `ucimlrepo`
2. **Tratamento de valores faltantes** (forward fill + modo)
3. **One-hot encoding** de variáveis categóricas
4. **Normalização** com `StandardScaler` (fit em treino)
5. **Split estratificado** em 60% treino / 20% validação / 20% teste
6. **Salvamento** em `datasets/processed/<dataset>/`

### Executar

```bash
python3 src/preprocessing.py
```

Com parâmetros customizados:

```bash
python3 src/preprocessing.py --output-dir datasets/processed --random-state 42
```

### Estrutura de saída

Para cada base (`adult`, `bank_marketing`, `heart_disease`, `mushroom`):

```
datasets/processed/<dataset>/
├── X_train.npy       # Features de treino (normalizado)
├── X_val.npy         # Features de validação (normalizado)
├── X_test.npy        # Features de teste (normalizado)
├── y_train.npy       # Rótulos de treino
├── y_val.npy         # Rótulos de validação
├── y_test.npy        # Rótulos de teste
├── scaler_mean.npy   # Média do escaler
├── scaler_scale.npy  # Desvio padrão do escaler
└── metadata.json     # Informações do dataset
```

## Execução dos Algoritmos

**Runner único (pré-processamento + todos os 4 algoritmos):**

```bash
python3 src/run_all.py
```

Ou executar individualmente:

```bash
# Pré-processamento (gera datasets processados e análise exploratória básica)
python3 src/preprocessing.py

# Análise exploratória detalhada (gráficos + relatórios completos)
python3 src/exploratory_analysis.py

# Algoritmos individuais
python3 src/mlp_classifier.py
python3 src/rbm_logistic_classifier.py
python3 src/mamdani_fuzzy_classifier.py
python3 src/anfis_classifier.py
```

**Saídas geradas:**
- `resultados/resultados_mlp.txt` - Resultados MLP
- `resultados/resultados_rbm_logistic.txt` - Resultados RBM + Logistic
- `resultados/resultados_mamdani_fuzzy.txt` - Resultados Mamdani
- `resultados/resultados_anfis.txt` - Resultados ANFIS

## Referências Bibliográficas

- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 1771-1800.
- Mamdani, E. H. (1975). Application of fuzzy algorithms for control of simple dynamic plant. *Proceedings of the Institution of Electrical Engineers*, 121(12), 1585-1588.
- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665-685.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
