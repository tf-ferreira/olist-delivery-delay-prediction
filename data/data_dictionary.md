# Dicionário de dados — Brazilian E-Commerce Public Dataset by Olist

Descrição detalhada dos 9 arquivos brutos em `data/raw/`, gerada a partir de inspeção direta dos CSVs (pandas). Todos os números abaixo foram medidos nos arquivos, não copiados da documentação do Kaggle.

**Fonte:** [Kaggle — Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
**Período dos pedidos:** 2016-09-04 a 2018-10-17
**Unidade de análise do projeto:** o pedido (`order_id`)
**Alvo do projeto:** atraso na entrega, definido como `date(order_delivered_customer_date) > date(order_estimated_delivery_date)`. Nos 96.470 pedidos entregues com data de entrega registrada, há **6.534 atrasos (6,77%)**, confirmando o desbalanceamento previsto.

---

## Visão geral e relacionamentos

| # | Arquivo | Linhas | Colunas | Granularidade (1 linha =) | Chave |
|---|---|---:|---:|---|---|
| 1 | `olist_orders_dataset.csv` | 99.441 | 8 | um pedido | `order_id` (única) |
| 2 | `olist_order_items_dataset.csv` | 112.650 | 7 | um item de um pedido | `order_id` + `order_item_id` |
| 3 | `olist_customers_dataset.csv` | 99.441 | 5 | um cliente em um pedido | `customer_id` (única) |
| 4 | `olist_sellers_dataset.csv` | 3.095 | 4 | um vendedor | `seller_id` (única) |
| 5 | `olist_products_dataset.csv` | 32.951 | 9 | um produto | `product_id` (única) |
| 6 | `olist_order_payments_dataset.csv` | 103.886 | 5 | um lançamento de pagamento | `order_id` + `payment_sequential` |
| 7 | `olist_order_reviews_dataset.csv` | 99.224 | 7 | uma avaliação | `review_id` (NÃO é única, ver seção 7) |
| 8 | `olist_geolocation_dataset.csv` | 1.000.163 | 5 | uma coordenada observada de um CEP | nenhuma (múltiplas linhas por prefixo) |
| 9 | `product_category_name_translation.csv` | 71 | 2 | uma categoria de produto | `product_category_name` (única) |

```
customers ──(customer_id)── orders ──(order_id)── order_items ──(product_id)── products ──(category)── translation
                              │                        │
                              ├──(order_id)── payments └──(seller_id)── sellers
                              │
                              └──(order_id)── reviews

geolocation ──(zip_code_prefix)── customers e sellers   [agregar para centroide ANTES do join]
```

---

## 1. `olist_orders_dataset.csv` — tabela central

99.441 linhas, 8 colunas. Uma linha por pedido; `order_id` e `customer_id` são ambos únicos aqui (relação 1:1, ver arquivo 3). É a tabela que define o alvo e o eixo temporal do split.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `order_id` | str (hash 32) | 0 | Identificador único do pedido. Chave primária. |
| `customer_id` | str (hash 32) | 0 | FK para `customers`. Único por linha (cada pedido gera um `customer_id` novo). |
| `order_status` | str | 0 | Status do pedido: `delivered` (96.478), `shipped` (1.107), `canceled` (625), `unavailable` (609), `invoiced` (314), `processing` (301), `created` (5), `approved` (2). |
| `order_purchase_timestamp` | datetime | 0 | Momento da compra. **Origem do eixo temporal**: split treino/teste e features de calendário derivam daqui. |
| `order_approved_at` | datetime | 160 (0,16%) | Aprovação do pagamento. **BANIDA como feature** (posterior à compra). |
| `order_delivered_carrier_date` | datetime | 1.783 (1,79%) | Postagem na transportadora. **BANIDA como feature** (posterior à compra). |
| `order_delivered_customer_date` | datetime | 2.965 (2,98%) | Entrega ao cliente. **Componente do alvo**, jamais feature. Nula em pedidos não entregues. |
| `order_estimated_delivery_date` | datetime | 0 | Data estimada de entrega mostrada ao cliente na compra. Vem sempre com hora 00:00:00 (só 459 valores distintos), por isso o alvo compara DATAS, não timestamps. **PERMITIDA como feature** (conhecida na compra); dela deriva a janela prometida em dias. |

Observações:
- Escopo do projeto: apenas `order_status == 'delivered'` com data de entrega não nula (96.470 pedidos). 8 pedidos `delivered` têm `order_delivered_customer_date` nula e saem do escopo; cancelados e nunca entregues são documentados como limitação.
- As três colunas de datas operacionais (aprovação, postagem, entrega) são o núcleo da auditoria anti-vazamento do `notebooks/01`.

## 2. `olist_order_items_dataset.csv` — itens dos pedidos

112.650 linhas, 7 colunas. Uma linha por item; chave composta `order_id` + `order_item_id`. Cobre 98.666 pedidos distintos. Distribuição: 88.863 pedidos com 1 item, 7.516 com 2, máximo de 21 itens. **1.278 pedidos (1,3%) têm mais de um vendedor**, o que justifica a agregação para o nível de pedido e a feature "nº de vendedores distintos".

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `order_id` | str (hash 32) | 0 | FK para `orders`. |
| `order_item_id` | int | 0 | Sequencial do item dentro do pedido (1 a 21). |
| `product_id` | str (hash 32) | 0 | FK para `products` (32.951 produtos distintos). |
| `seller_id` | str (hash 32) | 0 | FK para `sellers` (3.095 vendedores distintos). |
| `shipping_limit_date` | datetime | 0 | Prazo limite para o vendedor postar o item. Definido no checkout, logo em princípio conhecida na compra, mas fica fora da lista principal de features por prudência (validar na auditoria do notebook 01 se é sempre ≥ compra). |
| `price` | float | 0 | Preço do item em R$. Agregar por soma no pedido. |
| `freight_value` | float | 0 | Frete do item em R$ (rateado por item). Agregar por soma; base da razão frete/preço. |

Agregações planejadas por `order_id`: soma de `price` e `freight_value`, contagem de itens, nº de `seller_id` distintos, e atributos físicos do produto via join com `products`.

## 3. `olist_customers_dataset.csv` — clientes

99.441 linhas, 5 colunas. **Pegadinha do schema:** `customer_id` é único por PEDIDO (1:1 com `orders`), enquanto `customer_unique_id` identifica a pessoa (96.096 valores; ~3% dos clientes compraram mais de uma vez). Para este projeto o join é por `customer_id`; `customer_unique_id` só interessaria para análise de recompra, fora do escopo.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `customer_id` | str (hash 32) | 0 | Chave de join com `orders`. Um valor por pedido. |
| `customer_unique_id` | str (hash 32) | 0 | Identificador estável da pessoa através de pedidos. |
| `customer_zip_code_prefix` | int | 0 | 5 primeiros dígitos do CEP (14.994 distintos). Join com `geolocation` agregada; 157 prefixos não existem na geolocation (tratar nulo de distância). |
| `customer_city` | str | 0 | Cidade, minúscula, sem padronização garantida (4.119 valores). |
| `customer_state` | str | 0 | UF (27 valores). **Feature direta** (taxa de atraso varia por UF). |

## 4. `olist_sellers_dataset.csv` — vendedores

3.095 linhas, 4 colunas, `seller_id` único. Menor tabela de entidades; base da segmentação K-Means (features agregadas por vendedor: volume, receita, taxa histórica de atraso, tempo médio de postagem) e origem do cálculo de distância vendedor→cliente.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `seller_id` | str (hash 32) | 0 | Chave primária; FK em `order_items`. |
| `seller_zip_code_prefix` | int | 0 | Prefixo de CEP do vendedor (2.246 distintos; 7 sem correspondência na geolocation). |
| `seller_city` | str | 0 | Cidade (611 valores, com sujeira de digitação conhecida no dataset). |
| `seller_state` | str | 0 | UF (23 valores; forte concentração em SP). |

## 5. `olist_products_dataset.csv` — produtos

32.951 linhas, 9 colunas, `product_id` único. Fornece os atributos físicos usados como feature (peso, dimensões → volume).

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `product_id` | str (hash 32) | 0 | Chave primária; FK em `order_items`. |
| `product_category_name` | str | 610 (1,85%) | Categoria em português (73 valores). **Feature**; nulos viram categoria explícita "desconhecida". |
| `product_name_lenght` | float | 610 | Nº de caracteres do nome do anúncio. Irrelevante para atraso; descartar. (Typo "lenght" é do dataset original.) |
| `product_description_lenght` | float | 610 | Nº de caracteres da descrição. Descartar. |
| `product_photos_qty` | float | 610 | Nº de fotos do anúncio. Descartar. |
| `product_weight_g` | float | 2 (0,01%) | Peso em gramas. **Feature** (agregada por soma no pedido). |
| `product_length_cm` | float | 2 | Comprimento em cm. Compõe o volume (L×A×P). |
| `product_height_cm` | float | 2 | Altura em cm. Compõe o volume. |
| `product_width_cm` | float | 2 | Largura em cm. Compõe o volume. |

Os mesmos 610 registros concentram os nulos das 4 primeiras colunas de atributos (anúncios sem cadastro completo); os nulos físicos são só 2 produtos, imputáveis pela mediana da categoria ou global.

## 6. `olist_order_payments_dataset.csv` — pagamentos

103.886 linhas, 5 colunas. Uma linha por lançamento; um pedido pode ter vários lançamentos (`payment_sequential` até 29, tipicamente combinação com voucher). Cobre 99.440 pedidos, ou seja, 1 pedido de `orders` não tem pagamento registrado.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `order_id` | str (hash 32) | 0 | FK para `orders`. |
| `payment_sequential` | int | 0 | Sequencial do lançamento dentro do pedido. |
| `payment_type` | str | 0 | `credit_card` (76.795), `boleto` (19.784), `voucher` (5.775), `debit_card` (1.529), `not_defined` (3). |
| `payment_installments` | int | 0 | Nº de parcelas (0 a 24; há valores 0, anomalia a registrar). |
| `payment_value` | float | 0 | Valor do lançamento em R$. |

Uso no projeto: secundário. `payment_type` (ex.: boleto atrasa a aprovação) e `payment_installments` são conhecidos na compra e PODEM entrar como feature, mas não estão na lista principal; decidir na EDA. Se usados, agregar para o nível de pedido (tipo dominante por valor, soma de valores).

## 7. `olist_order_reviews_dataset.csv` — avaliações

99.224 linhas, 7 colunas. **Nenhuma coluna deste arquivo pode ser feature: toda avaliação é posterior à entrega, vazamento por construção.** Uso legítimo: validação de negócio na EDA (pedidos atrasados devem ter score menor, o que quantifica o custo do atraso em satisfação).

Integridade digna de nota: `review_id` tem 814 duplicatas (um mesmo review vinculado a mais de um pedido) e 547 pedidos têm mais de uma avaliação. Se usado na EDA, deduplicar por pedido (ex.: avaliação mais recente).

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `review_id` | str (hash 32) | 0 | Identificador da avaliação (não único). |
| `order_id` | str (hash 32) | 0 | FK para `orders`. |
| `review_score` | int | 0 | Nota de 1 a 5. |
| `review_comment_title` | str | 87.656 (88,34%) | Título do comentário, majoritariamente vazio. |
| `review_comment_message` | str | 58.247 (58,70%) | Texto do comentário (NLP fica como próximo passo, fora do escopo). |
| `review_creation_date` | datetime | 0 | Data de envio do questionário ao cliente. |
| `review_answer_timestamp` | datetime | 0 | Momento da resposta do cliente. |

## 8. `olist_geolocation_dataset.csv` — geolocalização

1.000.163 linhas, 5 colunas. Maior arquivo (61 MB). **Não tem chave: são múltiplas observações de coordenadas por prefixo de CEP** (mediana de 29 linhas por prefixo, máximo 1.146). Join direto multiplicaria as linhas da tabela mestra; por decisão já tomada, agregar para centroide (mediana de lat/lng por prefixo) ANTES de qualquer join.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `geolocation_zip_code_prefix` | int | 0 | Prefixo de CEP (19.015 distintos; cobre quase todos os prefixos de clientes e vendedores). |
| `geolocation_lat` | float | 0 | Latitude. **Contém outliers impossíveis** (máx +45,07; Brasil vai de ~+5,3 a ~-33,8). |
| `geolocation_lng` | float | 0 | Longitude. **Contém outliers impossíveis** (mín -101,5 e máx +121,1; Brasil vai de ~-74 a ~-34). |
| `geolocation_city` | str | 0 | Cidade (8.011 valores, com muita sujeira de grafia). Não usar; UF e coordenadas bastam. |
| `geolocation_state` | str | 0 | UF (27 valores). |

Tratamento planejado: filtrar coordenadas para a caixa geográfica do Brasil antes do centroide (a mediana já mitiga, mas o filtro documenta a consciência do problema). Prefixos ausentes (157 de clientes, 7 de vendedores) geram distância nula a imputar/sinalizar.

## 9. `product_category_name_translation.csv` — tradução de categorias

71 linhas, 2 colunas, mapeamento 1:1 português → inglês.

| Coluna | Tipo | Nulos | Descrição |
|---|---|---:|---|
| `product_category_name` | str | 0 | Categoria em português (chave). |
| `product_category_name_english` | str | 0 | Categoria em inglês. |

Atenção: `products` tem 73 categorias, a tradução tem 71. Faltam `pc_gamer` e `portateis_cozinha_e_preparadores_de_alimentos`. Como os textos do projeto são em português, a tradução é dispensável para o pipeline; se usada em alguma figura, tratar as 2 categorias faltantes explicitamente.

---

## Síntese das armadilhas de integridade (checklist para o notebook 01)

1. **Vazamento temporal:** `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date` e TODO o arquivo de reviews são posteriores à compra, banidos como feature. `order_estimated_delivery_date` é a exceção permitida.
2. **Geolocation sem chave:** agregar para centroide por prefixo antes do join, senão a mestra explode de ~96k para milhões de linhas.
3. **Coordenadas absurdas:** filtrar lat/lng para o território brasileiro antes de calcular haversine.
4. **`customer_id` ≠ pessoa:** é um id por pedido; a pessoa é `customer_unique_id`.
5. **Reviews duplicados:** `review_id` repetido e pedidos com múltiplos reviews; deduplicar se usar na EDA.
6. **Estimativa é meia-noite:** o alvo compara datas, não timestamps.
7. **Nulos estruturais:** 610 produtos sem cadastro de categoria; 8 pedidos `delivered` sem data de entrega; 1 pedido sem pagamento; 157+7 prefixos de CEP sem geolocalização; 2 categorias sem tradução.
