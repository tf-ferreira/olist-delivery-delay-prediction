# Previsão de atraso de entrega no e-commerce brasileiro (Olist)

Prever, **no momento em que o cliente conclui a compra**, a probabilidade de o pedido ser entregue depois da data prometida, e transformar essa probabilidade em uma política operacional de notificação proativa.

**🔗 App interativo:** <https://olist-delivery-delay-prediction.streamlit.app/> · **📄 Relatório executivo:** [`reports/relatorio_executivo.pdf`](reports/relatorio_executivo.pdf)

Construído sobre o [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle): 96.470 pedidos entregues, 2016–2018, taxa de atraso de 6,77%.

## Resultados-chave

| O que | Número | Contexto |
|---|---|---|
| Precisão dos alertas no ponto default | **2,2x o acaso** | notificar aleatoriamente acertaria 7%; o modelo acerta 15,7% por alerta |
| Atrasos capturados no ponto default | **36%** | notificando 16% dos pedidos (~1.054 clientes/mês típico) |
| Robustez entre regimes | **AUC-ROC 0,61–0,79/mês** | teste de 6 meses cobrindo crise logística, greve dos caminhoneiros e recalibração de promessas |
| Prova anti-vazamento | **AUC 0,500 com alvo embaralhado** | verificação executável no notebook 03, com `assert` |

## O que diferencia este projeto

1. **Anti-vazamento como tese, não como checklist.** Auditoria coluna a coluna (o que era conhecido no momento da compra?), uma candidata a feature validada nos dados e mesmo assim excluída por prudência (`shipping_limit_date`), e a feature central, o histórico do vendedor, construída com **relógios de disponibilidade**: o desfecho de um pedido só entra no histórico quando aquele pedido é entregue, e o tempo de postagem quando o pacote chega à transportadora. O `expanding().shift(1)` clássico usaria desfechos que ainda não existiam. O contrato é executável: [6 testes pytest](tests/test_features.py) quebram se alguém violá-lo.
2. **Split temporal com o teste tocado uma única vez.** Treino até fev/2018, teste de mar–ago/2018 aberto uma vez ao final. O corte foi fixado por volumetria priorizando manter os choques reais (greve, crise) dentro do teste, com o trade-off documentado.
3. **Honestidade auditável.** O LightGBM venceu a passada única do teste (AUC-PR 0,181 vs 0,138) e a campeã selecionada na validação, a Regressão Logística, **não foi trocada**: trocar depois de ver o teste é usar o teste para selecionar. A lição (seleção em janela única tem variância) está documentada e virou próximo passo. Hipóteses desmentidas pela EDA (frete/preço sem sinal, multi-vendedor atrasando MENOS) estão relatadas, não escondidas.
4. **Métricas de negócio em contagens.** O ponto de operação é um **orçamento de alertas** ("notificar os k% pedidos mais arriscados"), imune ao deslocamento da escala de probabilidades entre retreinos. Cada orçamento vira contagens: notificados, atrasos capturados, falsos alarmes. O app recalcula o cenário ao vivo num slider.

## Como reproduzir

```bash
# 1. Ambiente (Python 3.11+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Dados: baixe o dataset no Kaggle e extraia os 9 CSVs em data/raw/
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# 3. Pipeline da tabela mestra (com auditoria de joins impressa)
python -m src.data_prep

# 4. Notebooks, na ordem (executáveis de ponta a ponta)
#    01_data_understanding → 02_eda_insights → 03_modeling

# 5. Testes do contrato anti-vazamento
python -m pytest tests/ -v

# 6. Artefatos do app + app local
python -m src.export_app_artifacts
streamlit run app/streamlit_app.py
```

## Estrutura

```
├── data/
│   ├── raw/                  # CSVs do Kaggle (não versionados)
│   ├── processed/            # parquets intermediários (não versionados)
│   └── data_dictionary.md    # dicionário dos 9 arquivos, por inspeção direta
├── notebooks/
│   ├── 01_data_understanding.ipynb   # alvo, auditoria de vazamento, armadilhas
│   ├── 02_eda_insights.ipynb         # 8 perguntas de negócio + personas (K-Means)
│   └── 03_modeling.ipynb             # features, split, modelos, avaliação, cenário
├── src/
│   ├── data_prep.py          # 9 tabelas → 1 linha por pedido, joins auditados
│   ├── features.py           # histórico do vendedor com relógios de disponibilidade
│   ├── train.py              # whitelist de features, split temporal, pipelines
│   ├── evaluate.py           # passada única, estratificação, orçamento de alertas
│   └── export_app_artifacts.py  # deriva os artefatos pequenos que o app consome
├── tests/                    # o contrato anti-vazamento, executável
├── app/                      # Streamlit (4 páginas) + artefatos derivados (1,8 MB)
└── reports/                  # relatório executivo (PDF) + figuras oficiais
```

Princípio: **notebooks contam a história; `src/` é código de produção** (tipado, docstrings, testado, formatado com ruff, seeds fixadas).

## Principais achados da análise

- **O atraso é relativo à promessa, não só à logística:** a taxa mensal varia de 3% a 19%; após a greve de mai/2018, as promessas foram recalibradas e a taxa caiu a 1,2% sem a logística mudar. Promessas de prazo intermediário (17–22 dias) atrasam mais que as curtas e as longas.
- **Geografia domina:** de 4,5% (SP) a 21,4% (AL); o Rio de Janeiro é o 2º maior mercado com quase 3x a taxa de SP; a distância vendedor→cliente tem gradiente monotônico limpo.
- **O risco de vendedor é concentrado e antecipável:** 10% dos vendedores ("gargalos logísticos", via K-Means) têm 2,4x a taxa média, com tempo de postagem ~3x maior, sintoma visível antes do atraso.

## Como isto iria para produção

Desenho de industrialização (sem código, arquitetura de referência em stack Azure/Databricks):

1. **Ingestão** orquestrada por **Airflow**: pedidos, itens e cadastros aterrissando em camadas bronze de um lakehouse, com backfill idempotente.
2. **Transformação em dbt** reproduzindo o `data_prep`: os disjuntores do pipeline (contagem de linhas por join, whitelist de nulos justificados) viram testes de dados nativos (`dbt test`), rodando a cada carga.
3. **Feature store com relógios de disponibilidade:** as features do vendedor materializadas incrementalmente com a mesma semântica hermética deste repo (desfecho disponível na entrega, postagem no despacho), garantindo paridade treino/serving por construção.
4. **Treino e registro em Databricks/Azure ML:** reseleção periódica de campeão com **validação de origem rolante** (a lição da janela única), registro de modelo versionado, e o orçamento de alertas como configuração de negócio, não constante de código.
5. **Serving** batch diário (escoragem dos pedidos do dia para a fila de notificação) ou endpoint em tempo real no checkout; o app deste repo é o protótipo da camada de consumo.
6. **Monitoramento e retreino:** a estratificação mensal do notebook 03 vira dashboard (AUC-ROC móvel + prevalência); a degradação observada no mês mais distante do treino fundamenta o gatilho de retreino; alarmes de drift de features completam o ciclo.

## Limitações (ditas com franqueza) e próximos passos

- **Grão do pedido vs remessa:** pedidos multi-vendedor (1,3%) são agregados no pedido; peso/volume somados não correspondem a uma remessa física. Redesenho por grão de remessa (padrão gargalo por distância) especificado como iteração futura.
- **Seleção em janela única de validação** tem variância (demonstrado pelo LightGBM no teste) → rolling validation.
- **Probabilidades sem calibração fina** (a política por orçamento contorna; calibração formal é evolução).
- Próximos passos adicionais: piloto operacional com métricas de aceite, risco por rota origem×destino, Optuna com orçamento pequeno, MLflow tracking, CI rodando lint e pytest.

## Créditos e licença dos dados

Dataset por [Olist](https://olist.com/), publicado no Kaggle sob CC BY-NC-SA 4.0. Este repositório não redistribui os dados brutos; o app consome apenas agregados derivados.

## Autor

Thiago Ferreira, cientista de dados, bacharel em Matemática Aplicada e Computação Científica (ICMC/USP), MBA em Data Science em andamento (ESALQ/USP).
