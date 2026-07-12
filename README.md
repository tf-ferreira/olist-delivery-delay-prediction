# Previsão de atraso de entrega no e-commerce brasileiro (Olist)

> **Status: em construção.** Este README será expandido ao longo do projeto com resultados, instruções completas de reprodução e a arquitetura de produção proposta.

Projeto de ciência de dados sobre o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, ~100 mil pedidos, 2016 a 2018). O objetivo é prever, **no momento da compra**, a probabilidade de um pedido ser entregue com atraso em relação à data prometida ao cliente, permitindo notificação proativa e priorização logística.

O projeto cobre o ciclo completo:

1. **Entendimento dos dados** com auditoria explícita de vazamento (quais colunas são conhecidas no momento da compra e quais não são).
2. **EDA orientada a perguntas de negócio** e segmentação de vendedores em personas via K-Means.
3. **Modelagem** com split temporal (Regressão Logística → Random Forest → LightGBM), tratamento de desbalanceamento por pesos de classe e ajuste de threshold pela curva precision-recall.
4. **Avaliação com métricas estatísticas e de negócio**, traduzindo precision/recall em um cenário operacional de notificação proativa de clientes.
5. **Entrega**: app interativo em Streamlit e relatório executivo em PDF.

## Dados

Os CSVs brutos não são versionados. Para reproduzir:

1. Baixe o dataset no Kaggle: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>
2. Extraia os 9 arquivos CSV em `data/raw/`

O dicionário de dados construído por inspeção direta está em [`data/data_dictionary.md`](data/data_dictionary.md).

## Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estrutura planejada

```
├── data/
│   ├── raw/                  # CSVs do Kaggle (não versionados)
│   ├── processed/            # parquets intermediários (não versionados)
│   └── data_dictionary.md    # dicionário dos 9 arquivos brutos
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda_insights.ipynb
│   └── 03_modeling.ipynb
├── src/                      # código reutilizável (data prep, features, treino, avaliação)
├── app/                      # aplicativo Streamlit
└── reports/                  # figuras e relatório executivo
```

## Autor

Thiago Ferreira, cientista de dados, bacharel em Matemática Aplicada e Computação Científica (ICMC/USP), MBA em Data Science em andamento (ESALQ/USP).
