# Como rodar os experimentos implementados

Esta versão implementa os itens 3, 4 e 5 combinados:

1. comparação de diferentes parâmetros por algoritmo;
2. 21 execuções independentes por dataset e algoritmo;
3. matriz de confusão para treino, validação e teste.

## Instalação

```bash
pip install -r requirements.txt
```

## Execução completa

```bash
python src/main.py
```

A execução completa chama, nesta ordem:

```text
1. src/preprocessing.py
2. src/mlp_classifier.py
3. src/rbm_logistic_classifier.py
4. src/mamdani_fuzzy_classifier.py
5. src/anfis_classifier.py
```

## Execução por algoritmo

Também é possível rodar um algoritmo por vez:

```bash
python src/mlp_classifier.py
python src/rbm_logistic_classifier.py
python src/mamdani_fuzzy_classifier.py
python src/anfis_classifier.py
```

## O que cada script faz agora

Para cada dataset, o script:

1. carrega os dados processados de `datasets/processed/<dataset>/`;
2. testa uma grade de configurações de parâmetros;
3. seleciona a melhor configuração pelo maior F1-score na validação;
4. repete o processo para 21 sementes diferentes;
5. calcula média e desvio-padrão das métricas;
6. salva as matrizes de confusão agregadas das 21 execuções.

## Arquivos de saída

Cada algoritmo gera arquivos em `resultados/`, por exemplo para a MLP:

```text
resultados_mlp.txt
resultados_mlp_melhores_execucoes.csv
resultados_mlp_todos_parametros.csv
resultados_mlp_detalhado.json
resumo_mlp.csv
```

O arquivo `.txt` é o relatório legível.
O arquivo `melhores_execucoes.csv` mostra a configuração escolhida em cada uma das 21 execuções.
O arquivo `todos_parametros.csv` registra todas as configurações testadas.
O arquivo `.json` mantém os resultados completos, incluindo as matrizes de confusão de cada execução.

## Observação sobre tempo de execução

A nova versão faz muito mais testes do que a anterior. Portanto, é normal demorar mais, principalmente em MLP, RBM e nos modelos fuzzy.
