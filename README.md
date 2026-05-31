<p align="center">
  <img src="imgs/logo_azul.png" alt="CEFET-MG" width="100px" height="100px">
</p>

# Redes Neurais Artificiais e Sistemas Neuro-Fuzzy

Avaliação comparativa de algoritmos de aprendizado supervisionado e sistemas neuro-fuzzy em tarefas de classificação tabular.

<div align="justify">
<p><strong>Disciplina:</strong> Inteligência Computacional<br>
<strong>Instituição:</strong> Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG) - Campus V Divinópolis<br>
<strong>Professor:</strong> Alisson Marques da Silva<br>
<strong>Projeto:</strong> "Atividade 03"<br>
<strong>Alunos:</strong> Jader Oliveira Silva e João Pedro Rodrigues Silva
</p>
</div>

<div align="center">
  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
  [![UCI ML Repo](https://img.shields.io/badge/UCI-Machine%20Learning%20Repo-0052CC?style=for-the-badge)](https://archive.ics.uci.edu/)
</div>

## Resumo 

Este repositório implementa e compara quatro abordagens para classificação: MLP, RBM seguido de regressão logística, um sistema fuzzy de Mamdani simplificado e um ANFIS simplificado. A avaliação é realizada em quatro bases públicas com distintos perfis de dimensão e balanceamento (`adult`, `bank_marketing`, `heart_disease`, `mushroom`). Adotamos uma metodologia experimental com 60% treino / 20% validação / 20% teste e seleção por F1 no conjunto de validação, repetida em 21 sementes independentes para estimar variabilidade.

## 1) Introdução

O objetivo do estudo é comparar desempenho e custo computacional de abordagens neurais e neuro-fuzzy em problemas tabulares realistas. Além disso, investigamos o impacto de hiperparâmetros para cada algoritmo através de uma abordagem incremental (variação univariada) e de uma grade fina limitada para evitar explosão combinatória.

## 2) Bases de dados

- `adult`: ~48k amostras, alta dimensionalidade pós one-hot, multiclasse desbalanceada.
- `bank_marketing`: ~45k amostras, problema binário desbalanceado.
- `heart_disease`: ~303 amostras, baixa dimensão, multiclasses.
- `mushroom`: ~8k amostras, alta dimensionalidade, binário balanceado.

Os dados processados (one-hot, normalização e splits estratificados) estão em `datasets/processed/<dataset>/`.

## 3) Metodologia experimental

- Pré-processamento: `src/preprocessing.py` (one-hot, StandardScaler, split 60/20/20).
- Para cada algoritmo: executar busca em grade limitada; para cada `random_state` testar toda a grade e escolher a configuração com maior F1 na validação. Repetir para 21 seeds (arquivo `experiment_utils.py`).
- Métricas: Accuracy, Precision (weighted), Recall (weighted) e F1 (weighted). Matrizes de confusão agregadas são salvas.

## 4) Algoritmos (implementação e parâmetros)

4.1. MLP (Perceptron Multicamadas)
- Implementação: `src/mlp_classifier.py` (base)
- Execução otimizada unificada: `src/optimized_experiments.py --algorithm mlp`
- Estudo univariado unificado: `src/hyperparameter_study.py --algorithm mlp`
- Hiperparâmetros controlados no estudo: número de camadas ocultas e neurônios, `activation`, `learning_rate_init`, `alpha` (L2), `max_iter`, `early_stopping`.

Resumo das conclusões para MLP (estudo incremental e grade fina)
- Variação univariada das camadas mostrou ganho gradual de F1 até 200 neurônios em `adult`.
- `learning_rate_init=1e-3` é robusto; `1e-2` acelera, mas aumenta variância.
- `alpha=1e-4` (L2) melhorou generalização levemente no `adult`.

Resultados agregados e arquivos de estudo estão em `resultados[2]/` (ex.: `resumo_mlp.csv`, `comparacao_mlp_adult.json`).

4.2. RBM + Regressão Logística
- Implementação: `src/rbm_logistic_classifier.py`.
- Execução otimizada unificada: `src/optimized_experiments.py --algorithm rbm`
- Estudo univariado unificado: `src/hyperparameter_study.py --algorithm rbm`

Versão otimizada e estudo concluído
- Estudo univariado em `resultados[2]/rbm_hyperparameter_study/`.
- Configuração final adotada: `n_components=128`, `n_iter=20`, `learning_rate=0.05`, `logistic_C=10.0`.
- Execução final salva em `resultados[2]/rbm_optimized/`.

Resumo comparativo da RBM otimizada

| Dataset | F1 validação (orig.) | F1 validação (otim.) | F1 teste (orig.) | F1 teste (otim.) | Tempo médio orig. | Tempo médio otim. |
|---|---:|---:|---:|---:|---:|---:|
| Adult | 0.4402 | 0.4403 | 0.4449 | 0.4456 | 69.33s | 45.70s |
| Bank Marketing | 0.8557 | 0.8596 | 0.8573 | 0.8621 | 8.46s | 19.29s |
| Heart Disease | 0.4514 | 0.4911 | 0.3844 | 0.4775 | 0.05s | 0.15s |
| Mushroom | 0.9967 | 0.9993 | 0.9990 | 0.9997 | 1.98s | 2.56s |

Observações
- Em `adult`, o ganho em F1 foi muito pequeno, mas com redução do tempo médio por execução.
- Em `bank_marketing`, houve melhora moderada de F1, com maior custo computacional.
- Em `heart_disease`, a otimização trouxe o maior ganho relativo de generalização.
- Em `mushroom`, o desempenho já era muito alto; a melhoria foi marginal, como esperado.

4.3. Sistema Fuzzy de Mamdani
- Implementação: `src/mamdani_fuzzy_classifier.py` (classificador por similaridade com MFs triangulares).
- Execução otimizada unificada: `src/optimized_experiments.py --algorithm mamdani`
- Estudo univariado unificado: `src/hyperparameter_study.py --algorithm mamdani`

Versão otimizada e estudo concluído
- Estudo univariado em `resultados[2]/mamdani_hyperparameter_study/`.
- Configuração final adotada: `n_membership_functions=5`, `max_train_samples=800`.
- Execução final salva em `resultados[2]/mamdani_optimized/`.

Resumo comparativo do Mamdani otimizado

| Dataset | F1 validação (orig.) | F1 validação (otim.) | F1 teste (orig.) | F1 teste (otim.) | Tempo médio orig. | Tempo médio otim. |
|---|---:|---:|---:|---:|---:|---:|
| Adult | 0.4360 | 0.4355 | 0.4349 | 0.4378 | 4.44s | 44.35s |
| Bank Marketing | 0.8438 | 0.8441 | 0.8442 | 0.8450 | 1.90s | 15.90s |
| Heart Disease | 0.5842 | 0.5842 | 0.5191 | 0.5191 | 0.03s | 0.09s |
| Mushroom | 0.9966 | 0.9982 | 0.9983 | 0.9990 | 0.83s | 5.32s |

Observações
- Em `adult` e `bank_marketing`, os ganhos em F1 foram pequenos.
- Em `heart_disease`, os resultados permaneceram essencialmente iguais ao baseline.
- Em `mushroom`, houve ganho leve em validação e teste.
- O custo computacional aumentou significativamente com `max_train_samples=800`.

4.4. ANFIS (Adaptive Neuro-Fuzzy Inference System)
- Implementação: `src/anfis_classifier.py` (versão simplificada com MFs gaussianas e PCA prévio opcional).
- Execução otimizada unificada: `src/optimized_experiments.py --algorithm anfis`
- Estudo univariado unificado: `src/hyperparameter_study.py --algorithm anfis`

Versão otimizada e estudo concluído
- Estudo univariado em `resultados[2]/anfis_hyperparameter_study/`.
- Configuração final adotada: `n_membership_functions=4`, `learning_rate=0.05`, `n_epochs=50`, `pca_components=2`, `max_train_samples=150`.
- Execução final salva em `resultados[2]/anfis_optimized/`.

Resumo comparativo do ANFIS otimizado

| Dataset | F1 validação (orig.) | F1 validação (otim.) | F1 teste (orig.) | F1 teste (otim.) | Tempo médio orig. | Tempo médio otim. |
|---|---:|---:|---:|---:|---:|---:|
| Adult | 0.3409 | 0.4208 | 0.3408 | 0.4220 | 0.85s | 2.46s |
| Bank Marketing | 0.8281 | 0.8281 | 0.8281 | 0.8281 | 0.71s | 2.34s |
| Heart Disease | 0.3798 | 0.5788 | 0.3798 | 0.5322 | 0.10s | 0.87s |
| Mushroom | 0.8954 | 0.8839 | 0.9004 | 0.8965 | 0.44s | 1.00s |

Observações
- Em `adult`, o ANFIS otimizado trouxe ganho claro de generalização, com aumento substancial de F1.
- Em `heart_disease`, a melhoria foi ainda mais expressiva, indicando que a redução de dimensão e o aumento de regras ajudaram na separação.
- Em `bank_marketing`, os resultados ficaram praticamente estáveis.
- Em `mushroom`, houve pequena queda em validação e leve perda em teste, sugerindo que a versão original já estava bem ajustada para essa base.

## 5) Estudo de Hiperparâmetros (procedimento)

Procedimento adotado:
- Fixar uma configuração base (valores default ou heurísticos).
- Variar apenas um hiperparâmetro por experimento (univariado) em uma grade reduzida.
- Executar com 5 seeds para identificar tendências e, quando promissor, expandir para 21 seeds.
- Manter registros compactos em `resultados[2]/` com resumos por dataset e por parâmetro.

Arquivos de resultados resumidos gerados pelo estudo:
- `resultados[2]/resumo_mlp.csv` — resumo por dataset para MLP (médias, desvios, tempo).
- `resultados[2]/comparacao_mlp_adult.json` — comparação direta entre `resultados[1]` (original) e `resultados[2]` (fino) para `adult`.

## 6) Resultados e análise (sumário)

- Exemplo (MLP, dataset `adult`): grade fina apresentou melhoria marginal em F1 de validação (+0.0038) e de teste (+0.0026) com tempo médio inalterado.
- Interpretação: pequenas melhorias indicam que a configuração original estava próxima de um ótimo local; a adição de L2 e teste de arquiteturas maiores forneceu ganho de generalização moderado.
- RBM + Logistic (estudo otimizado): o comportamento foi dependente do dataset. Houve ganhos modestos em `adult`, ganhos moderados em `bank_marketing` e `mushroom`, e ganho expressivo em `heart_disease`, com aumento de custo apenas nos datasets maiores e redução de tempo em `adult`.
- ANFIS (estudo otimizado): houve ganho expressivo em `adult` e `heart_disease`, estabilidade em `bank_marketing` e pequena perda em `mushroom`; o custo computacional aumentou em todos os datasets devido à maior profundidade do modelo e ao número de épocas.
- Mamdani (estudo otimizado): desempenho muito próximo ao baseline na maioria das bases, com ganhos marginais em `mushroom`; aumento de custo relevante devido ao maior número de amostras de treino usadas na similaridade fuzzy.

Seção completa de resultados (tabelas e gráficos) será preenchida no relatório final a partir de arquivos em `resultados/`.

## 7) Reprodutibilidade

Passos principais para reproduzir os experimentos:

1. Instalar dependências:

```bash
pip install -r requirements.txt
```

2. Gerar dados processados (one-hot, normalização):

```bash
python3 src/preprocessing.py
```

3. Executar um algoritmo específico (ex.: MLP no dataset `adult`):

```bash
python3 src/mlp_classifier.py --dataset adult
```

4. Rodar estudo incremental (scripts específicos salvo em `src/`): consultar `resultados[2]/` para saídas resumidas.

```bash
# Exemplo: estudo rápido de hiperparâmetros (5 seeds) para ANFIS no Adult
python3 src/hyperparameter_study.py --algorithm anfis --dataset adult --seeds 5

# Exemplo: execução otimizada (21 seeds) para Mamdani em todos os datasets
python3 src/optimized_experiments.py --algorithm mamdani
```

## 8) Organização dos diretórios e arquivos importantes

- `src/` — código-fonte dos algoritmos e utilitários.
- `src/hyperparameter_study.py` — ponto único para estudos univariados.
- `src/optimized_experiments.py` — ponto único para execuções otimizadas.
- `datasets/processed/` — datasets processados (npy + metadata).
- `resultados/` — resultados originais e detalhados (`resultados[1]/`, `resultados[2]/`).

## Referências

- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8), 1771-1800.
- Mamdani, E. H. (1975). Application of fuzzy algorithms for control of simple dynamic plant. *Proceedings of the Institution of Electrical Engineers*, 121(12), 1585-1588.
- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665-685.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
