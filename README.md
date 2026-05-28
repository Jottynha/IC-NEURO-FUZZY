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
| **Adult** | 2 | ~48,842 | 14 | Classificação binária: renda > 50K |
| **Bank Marketing** | 222 | ~45,211 | 20 | Classificação binária: subscrição a depósito |
| **Heart Disease** | 45 | 303 | 13 | Classificação binária: presença de doença |
| **Mushroom** | 73 | 8,124 | 22 | Classificação binária: comestível/venenoso |

## Dependências

Instale os requisitos:

```bash
pip install ucimlrepo pandas numpy scikit-learn
```

## Pré-processamento

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

---

## Atualização experimental

Esta versão inclui comparação de parâmetros, 21 execuções independentes por algoritmo/dataset e matrizes de confusão. Para detalhes, consulte o arquivo `COMO_RODAR_EXPERIMENTOS.md`.
