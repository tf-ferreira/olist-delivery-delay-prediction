"""Streamlit app: delivery delay prediction on the Olist dataset.

Five pages: context, data (tables and the leakage contract), EDA & personas,
model & live alert budget slider, and the order simulator. The app consumes
ONLY the small derived artifacts in ``app/artifacts/`` (no raw data), so it
deploys to Streamlit Community Cloud straight from the public repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ACCENT, GRAY, INK = "#2a78d6", "#898781", "#0b0b0b"

st.set_page_config(
    page_title="Atraso de entrega · Olist", page_icon="📦", layout="wide"
)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e1e0d9",
        "axes.edgecolor": GRAY,
        "axes.labelcolor": "#52514e",
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
    }
)


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    names = [
        "test_predictions",
        "operating_points",
        "pr_curve_val",
        "monthly_rate",
        "uf_rate",
        "dist_rate",
        "window_rate",
        "personas",
        "route_distance",
    ]
    return {n: pd.read_parquet(ARTIFACTS / f"{n}.parquet") for n in names}


@st.cache_data
def load_meta() -> tuple[dict, dict]:
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    defaults = json.loads((ARTIFACTS / "simulator_defaults.json").read_text())
    return meta, defaults


@st.cache_resource
def load_champion():
    return joblib.load(ARTIFACTS / "champion.joblib")


T = load_tables()
META, DEFAULTS = load_meta()

st.sidebar.title("📦 Atraso de entrega")
st.sidebar.caption("Brazilian E-Commerce Public Dataset (Olist) · 2016–2018")
page = st.sidebar.radio(
    "Páginas",
    [
        "1 · Contexto",
        "2 · Dados",
        "3 · EDA e personas",
        "4 · Modelo e cenário de negócio",
        "5 · Simulador de pedido",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    "Split temporal com corte em mar/2018; teste de mar–ago/2018 tocado uma única vez. "
    "Só entram no modelo informações conhecidas no momento da compra."
)


def rate_bar(
    df: pd.DataFrame, x: str, title: str, xlabel: str, horizontal: bool = False
):
    fig, ax = plt.subplots(figsize=(7.5, 4 if not horizontal else 6))
    if horizontal:
        ax.barh(df[x], 100 * df["rate"], color=ACCENT)
        ax.axvline(100 * META["global_rate"], color=GRAY, linewidth=1, linestyle="--")
        ax.grid(axis="x", color="#e1e0d9")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("taxa de atraso (%)")
    else:
        ax.bar(df[x], 100 * df["rate"], color=ACCENT, width=0.65)
        ax.axhline(100 * META["global_rate"], color=GRAY, linewidth=1, linestyle="--")
        ax.set_ylabel("taxa de atraso (%)")
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    return fig


# ------------------------------------------------------------------ page 1
if page.startswith("1"):
    st.title("Prever atraso de entrega no momento da compra")
    st.markdown(
        """
No instante em que o cliente conclui a compra, o site promete uma data de entrega.
Quando a entrega estoura essa data, a empresa só fica sabendo **depois do dano**.
Este projeto estima, **no momento da compra**, a probabilidade de cada pedido atrasar,
permitindo notificação proativa e priorização logística.
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Pedidos entregues analisados", f"{META['population']:,}".replace(",", ".")
    )
    c2.metric("Taxa de atraso", f"{META['global_rate']:.1%}")
    c3.metric("Período", "2016 – 2018")
    c4.metric("Modelo em produção simulada", "Reg. Logística")

    st.subheader("O alvo é relativo à promessa")
    st.markdown(
        """
Um pedido é **atrasado** quando a data de entrega real supera a data prometida no checkout
(comparação por **datas**: a promessa vem à meia-noite, e uma comparação ingênua por hora
rotularia errado 1.292 pedidos entregues no próprio dia prometido).

**Disciplina anti-vazamento:** o modelo só usa colunas conhecidas no momento da compra
(endereço, preço, frete, janela prometida, atributos do produto e o **histórico do vendedor
até aquele instante**). Datas de aprovação, postagem, entrega e reviews são banidas; a auditoria
completa, coluna a coluna, está no notebook 01 do repositório.
"""
    )
    monthly = T["monthly_rate"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = pd.to_datetime(monthly["purchase_year_month"])
    ax.plot(
        x, 100 * monthly["rate"], color=ACCENT, linewidth=2, marker="o", markersize=4
    )
    ax.axhline(100 * META["global_rate"], color=GRAY, linewidth=1, linestyle="--")
    ax.axvline(pd.Timestamp(META["cut"]), color=INK, linewidth=1, linestyle=":")
    ax.annotate(
        "corte treino | teste",
        xy=(pd.Timestamp(META["cut"]), 17),
        fontsize=8,
        color=INK,
        ha="center",
    )
    ax.set_title("Taxa de atraso por mês de compra (o fenômeno não é estável)")
    ax.set_ylabel("taxa de atraso (%)")
    st.pyplot(fig, width=760)
    st.caption(
        "Black Friday (nov/17, 12,4%), crise logística (mar/18, 19,0%) e a recalibração de "
        "promessas pós-greve (jun/18, 1,2%): o atraso mede a calibração da promessa, não só a logística."
    )

# ------------------------------------------------------------------ page 2
elif page.startswith("2"):
    st.title("Os dados: 9 tabelas e o contrato de uso")
    st.markdown(
        """
Fonte: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle), 9 tabelas relacionais com ~100 mil pedidos reais de 2016 a 2018. Antes de qualquer análise,
cada tabela teve granularidade e chaves **verificadas por código** (notebook 01), não assumidas da documentação.
"""
    )

    st.subheader("As 9 tabelas e o que é uma linha em cada uma")
    tabelas = pd.DataFrame(
        [
            ("orders", "um pedido", "datas de compra, entrega real e data prometida"),
            ("order_items", "um item de um pedido", "preço, frete, produto, vendedor"),
            ("order_payments", "um pagamento de um pedido", "tipo, parcelas, valor"),
            ("order_reviews", "uma avaliação", "nota 1–5, comentário (pós-entrega!)"),
            ("customers", "o cliente de UM pedido", "cidade, UF, prefixo de CEP"),
            ("sellers", "um vendedor", "cidade, UF, prefixo de CEP"),
            ("products", "um produto do catálogo", "categoria, peso, dimensões"),
            (
                "geolocation",
                "uma coordenada de CEP",
                "latitude, longitude (SEM chave!)",
            ),
            ("category_translation", "uma categoria", "nome em inglês"),
        ],
        columns=["tabela", "uma linha é...", "colunas principais"],
    )
    st.dataframe(tabelas, hide_index=True, width=760)
    st.caption(
        "Armadilhas verificadas em código: `customer_id` é único por PEDIDO (a pessoa é "
        "`customer_unique_id`); `geolocation` tem ~1 milhão de linhas sem chave (vira 1 centroide "
        "por prefixo de CEP, filtrado pela caixa do Brasil, ANTES de qualquer join); 814 reviews duplicados."
    )

    st.subheader("A auditoria de vazamento: o contrato do modelo")
    st.markdown(
        "Regra única: **uma coluna só pode ser feature se seu valor é conhecido no momento em que "
        "o cliente conclui a compra.** Cada coluna foi classificada; o resumo:"
    )
    audit = pd.DataFrame(
        [
            (
                "PERMITIDA",
                "data da compra; data PROMETIDA; preço; frete; endereço do cliente; categoria, peso e dimensões; pagamento",
                "existem na tela do checkout",
            ),
            (
                "PERMITIDA (derivada)",
                "janela prometida em dias; distância vendedor→cliente; razão frete/preço; histórico do vendedor ATÉ aquele instante",
                "construídas só com o que existia em t",
            ),
            (
                "BANIDA",
                "data de aprovação; data de postagem; data de entrega; status do pedido; reviews (100% pós-entrega)",
                "nascem DEPOIS da compra; em produção estariam vazias",
            ),
            (
                "ALVO",
                "data de entrega real vs prometida (comparação por DATAS)",
                "define o que prever; jamais entra como insumo",
            ),
            (
                "VALIDADA E EXCLUÍDA",
                "shipping_limit_date (limite de postagem)",
                "passou 100% na validação (0 violações em 112.650 itens) e ficou fora: o snapshot não prova que nunca é atualizada após a compra",
            ),
        ],
        columns=["status", "colunas", "motivo"],
    )
    st.dataframe(audit, hide_index=True, width=760)
    st.caption(
        "Prova executável do contrato: treinar com o alvo embaralhado derruba a AUC para 0,500 "
        "(célula com assert no notebook 03), e 6 testes pytest protegem a feature de histórico do vendedor."
    )

    st.subheader("População e alvo")
    st.markdown(
        """
- **96.470 pedidos** com status entregue e data registrada (8 excluídos por data nula); **6,77% atrasados**.
- O alvo compara **datas**, não timestamps: a promessa é registrada à meia-noite, e a comparação ingênua
  por hora rotularia errado **1.292 pedidos** entregues no próprio dia prometido (daria 8,11% em vez de 6,77%).
- Fora do escopo (limitação declarada): pedidos cancelados ou nunca entregues; e o fim do período é truncado
  em ago/2018, porque perto da borda só os pedidos entregues RÁPIDO já constavam na base (viés de sobrevivência).
"""
    )

# ------------------------------------------------------------------ page 3
elif page.startswith("3"):
    st.title("O que os dados contam")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Geografia", "Janela prometida", "Distância", "Personas de vendedores"]
    )
    with tab1:
        st.pyplot(
            rate_bar(
                T["uf_rate"].sort_values("rate"),
                "customer_state",
                "Taxa de atraso por UF de destino",
                "",
                horizontal=True,
            ),
            width=700,
        )
        st.caption(
            "Amplitude de 4,5% (SP) a 21,4% (AL). O caso de negócio mais forte é o RJ: "
            "2º maior mercado (12.350 pedidos) com taxa quase 3x a de SP."
        )
    with tab2:
        st.pyplot(
            rate_bar(
                T["window_rate"],
                "faixa",
                "Taxa de atraso por janela prometida (dias, quintis)",
                "dias",
            ),
            width=700,
        )
        st.caption(
            "Relação NÃO monotônica: o pior grupo é o intermediário (17–22 dias, 8,2%). "
            "A janela é a própria previsão da plataforma, e ela erra mais no meio."
        )
    with tab3:
        st.pyplot(
            rate_bar(
                T["dist_rate"],
                "faixa",
                "Taxa de atraso por distância vendedor→cliente (km, quintis)",
                "km",
            ),
            width=700,
        )
        st.caption(
            "Gradiente monotônico limpo: de 4,6% a 9,7% do quintil mais próximo ao mais distante."
        )
    with tab4:
        personas = T["personas"]
        profile = (
            personas.groupby("persona")
            .agg(
                vendedores=("seller_id", "size"),
                pedidos_mediana=("n_orders", "median"),
                taxa_atraso=("late_rate", "mean"),
                postagem_dias=("posting_days", "mean"),
            )
            .round(3)
        )
        st.dataframe(profile, width=700)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = {
            "Motores de volume": "#2a78d6",
            "Cauda longa ágil": "#1baf7a",
            "Gargalos logísticos": "#eda100",
        }
        for name, grp in personas.groupby("persona"):
            ax.scatter(
                grp["posting_days"],
                100 * grp["late_rate"],
                s=np.sqrt(grp["n_orders"]) * 3,
                alpha=0.55,
                label=name,
                color=colors.get(name, ACCENT),
                edgecolors="white",
                linewidths=0.3,
            )
        ax.set_xlim(0, 16)
        ax.legend(frameon=False)
        ax.set_title("Personas (K-Means, k=3): postagem × atraso")
        ax.set_xlabel("tempo médio de postagem (dias)")
        ax.set_ylabel("taxa de atraso do vendedor (%)")
        st.pyplot(fig, width=700)
        st.caption(
            "10% dos vendedores (Gargalos logísticos) carregam 2,4x a taxa global de atraso, "
            "com postagem ~3x mais lenta: risco concentrado e visível ANTES do atraso."
        )

# ------------------------------------------------------------------ page 4
elif page.startswith("4"):
    st.title("O modelo e o cenário de negócio ao vivo")
    st.markdown(
        """
Três modelos foram comparados sob protocolo rígido (seleção na validação temporal; teste de
mar–ago/2018 tocado **uma única vez**). A campeã é a **Regressão Logística** (AUC-PR 0,138 no
teste, 2,0x o acaso; AUC-ROC 0,699). O LightGBM venceu a passada única do teste (0,181) e a
campeã **não** foi trocada: trocar após ver o teste seria usar o teste para selecionar. A lição
virou próximo passo (validação em múltiplas janelas antes de produção).
"""
    )
    preds = T["test_predictions"]
    ops = T["operating_points"].set_index("ponto")

    months = META["n_test_months"]
    total_late_month = int(preds["is_late"].sum()) // months
    b1, b2, b3 = st.columns(3)
    b1.metric("Pedidos / mês típico", f"{len(preds) // months:,}".replace(",", "."))
    b2.metric(
        "Atrasos / mês típico",
        f"{total_late_month:,}".replace(",", "."),
        f"{META['test_prevalence']:.1%} dos pedidos",
        delta_color="off",
    )
    b3.metric("Janela do teste", "6 meses (mar–ago/2018)")
    st.caption(
        "Estes são os totais contra os quais o cenário abaixo se compara: sem modelo nenhum, "
        f"a operação enfrenta ~{total_late_month} atrasos por mês sem saber quais pedidos são."
    )
    st.divider()

    st.subheader("Escolha o orçamento de alertas")
    st.markdown(
        "O ponto de operação é um **orçamento**: a fração dos pedidos mais arriscados a notificar "
        "por mês. Thresholds de probabilidade não sobrevivem a retreinos; orçamentos sobrevivem."
    )
    named = {f"{k} ({v:.0%})": v for k, v in META["named_points"].items()}
    rate = (
        st.slider(
            "Orçamento de alertas (% dos pedidos notificados)",
            min_value=2.0,
            max_value=50.0,
            value=100 * META["default_alert_rate"],
            step=0.5,
            format="%.1f%%",
        )
        / 100
    )
    st.caption("Pontos nomeados escolhidos na validação: " + " · ".join(named))

    thr = float(np.quantile(preds["proba"], 1 - rate))
    alert = preds["proba"] >= thr
    months = META["n_test_months"]
    notified = int(alert.sum())
    captured = int((alert & (preds["is_late"] == 1)).sum())
    total_late = int(preds["is_late"].sum())
    false_alarms = notified - captured
    precision = captured / notified if notified else 0
    lift = precision / META["test_prevalence"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notificados / mês", f"{notified // months:,}".replace(",", "."))
    c2.metric(
        "Atrasos capturados / mês",
        f"{captured // months:,}".replace(",", "."),
        f"{captured / total_late:.0%} de todos os atrasos",
    )
    c3.metric("Falsos alarmes / mês", f"{false_alarms // months:,}".replace(",", "."))
    c4.metric(
        "Precisão dos alertas", f"{precision:.1%}", f"lift {lift:.1f}x sobre o acaso"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        # PR curve of the TEST itself: the slider point slides exactly on it
        # by construction (same predictions, same threshold semantics). The
        # validation curve sits higher (prevalence 9.8% vs 7.0%), so the point
        # would never touch it, which reads as a bug.
        from sklearn.metrics import precision_recall_curve

        prec_t, rec_t, _ = precision_recall_curve(preds["is_late"], preds["proba"])
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(rec_t[::20], prec_t[::20], color=ACCENT, linewidth=2)
        ax.axhline(META["test_prevalence"], color=GRAY, linewidth=1, linestyle="--")
        ax.annotate(
            f"acaso = prevalência ({META['test_prevalence']:.1%})",
            xy=(0.98, META["test_prevalence"]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=GRAY,
        )
        ax.scatter([captured / total_late], [precision], s=70, color=INK, zorder=3)
        ax.annotate(
            "você está aqui",
            xy=(captured / total_late, precision),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
        ax.set_ylim(0, 0.6)
        ax.set_title("Curva precision-recall (teste)")
        ax.set_xlabel("recall")
        ax.set_ylabel("precisão")
        st.pyplot(fig, width=560)
        st.caption(
            "O ponto desliza sobre a curva do teste conforme o orçamento. Os pontos "
            "nomeados do menu foram escolhidos na validação, antes de tocar o teste."
        )
    with col_b:
        monthly_roc = []
        from sklearn.metrics import roc_auc_score

        for month, grp in preds.groupby("purchase_year_month"):
            monthly_roc.append(
                {
                    "mês": month,
                    "auc_roc": roc_auc_score(grp["is_late"], grp["proba"]),
                    "taxa": grp["is_late"].mean(),
                }
            )
        mdf = pd.DataFrame(monthly_roc)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(range(len(mdf)), mdf["auc_roc"], color=ACCENT, linewidth=2, marker="o")
        ax.axhline(0.5, color=GRAY, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(mdf)), mdf["mês"], rotation=45)
        ax.set_ylim(0.45, 0.85)
        ax.set_title("Ordenação por mês do teste (AUC-ROC)")
        st.pyplot(fig, width=560)
        st.caption(
            "Estável através de crise, greve e recalibração; o pior mês é o mais distante do treino (drift)."
        )

# ------------------------------------------------------------------ page 4
else:
    st.title("Simulador de pedido")
    st.markdown(
        "Monte um pedido hipotético e receba a probabilidade de atraso da campeã. Campos não "
        "expostos usam a **mediana do treino** (a mesma disciplina do pipeline)."
    )
    champion = load_champion()
    med, modes = DEFAULTS["medians"], DEFAULTS["modes"]

    with st.form("simulador"):
        c1, c2, c3 = st.columns(3)
        with c1:
            uf = st.selectbox(
                "UF de destino",
                DEFAULTS["states"],
                index=DEFAULTS["states"].index("SP"),
            )
            category = st.selectbox("Categoria do produto", DEFAULTS["categories"])
            month = st.select_slider(
                "Mês da compra", options=list(range(1, 13)), value=6
            )
            payment = st.selectbox("Tipo de pagamento", DEFAULTS["payment_types"])
        with c2:
            seller_uf = st.selectbox(
                "UF do vendedor (origem)",
                DEFAULTS["states"],
                index=DEFAULTS["states"].index("SP"),
                help=(
                    "A origem entra no modelo pela DISTÂNCIA da rota e pelo histórico do "
                    "vendedor; a UF do vendedor não é uma feature própria. Selecionando a "
                    "origem, a distância usa a mediana real da rota origem→destino."
                ),
            )
            window = st.slider(
                "Janela prometida (dias)", 5, 60, int(med["promised_window_days"])
            )
            manual_dist = st.checkbox("Ajustar distância manualmente", value=False)
            distance_manual = st.slider(
                "Distância manual (km, usada só se marcado acima)",
                0,
                3200,
                int(med["distance_km"]),
            )
            weight = st.slider("Peso total (g)", 50, 30000, int(med["total_weight_g"]))
            n_items = st.slider("Nº de itens", 1, 6, 1)
        with c3:
            price = st.number_input(
                "Preço total (R$)", 10.0, 5000.0, float(round(med["total_price"], 0))
            )
            freight = st.number_input(
                "Frete total (R$)", 0.0, 500.0, float(round(med["total_freight"], 0))
            )
            seller_rate = (
                st.slider(
                    "Taxa de atraso histórica do vendedor (%)",
                    0.0,
                    50.0,
                    100 * med["seller_late_rate_hist"],
                )
                / 100
            )
            seller_posting = st.slider(
                "Tempo de postagem histórico do vendedor (dias)",
                0.5,
                15.0,
                float(round(med["seller_posting_days_hist"], 1)),
            )
        submitted = st.form_submit_button(
            "Calcular probabilidade de atraso", type="primary"
        )

    if submitted:
        routes = T["route_distance"]
        route = routes[
            (routes["seller_state"] == seller_uf) & (routes["customer_state"] == uf)
        ]
        route_km = (
            float(route["mediana_km"].iloc[0]) if len(route) else med["distance_km"]
        )
        distance = float(distance_manual) if manual_dist else route_km

        row = {c: med[c] for c in med}
        row.update(
            {
                "customer_state": uf,
                "main_category": category,
                "payment_type_primary": payment,
                "purchase_month": month,
                "promised_window_days": window,
                "distance_km": distance,
                "total_weight_g": weight,
                "total_volume_cm3": med["total_volume_cm3"]
                * weight
                / max(med["total_weight_g"], 1),
                "n_items": n_items,
                "total_price": price,
                "total_freight": freight,
                "freight_price_ratio": freight / max(price, 0.01),
                "payment_total": price + freight,
                "seller_late_rate_hist": seller_rate,
                "seller_posting_days_hist": seller_posting,
                "distance_missing": 0,
                "weight_missing": 0,
                "payment_missing": 0,
                "seller_no_history": 0,
            }
        )
        X = pd.DataFrame([row])[META["features"]]
        proba = float(champion.predict_proba(X)[:, 1][0])

        preds = T["test_predictions"]
        default_thr = float(np.quantile(preds["proba"], 1 - META["default_alert_rate"]))
        pct = float((preds["proba"] < proba).mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidade de atraso", f"{proba:.1%}")
        c2.metric("Mais arriscado que", f"{pct:.0%} dos pedidos do teste")
        alerted = proba >= default_thr
        c3.metric(
            "No orçamento default (16%)",
            "SERIA NOTIFICADO" if alerted else "não notificado",
        )
        st.progress(min(proba / 0.5, 1.0))
        origem = (
            "ajustada manualmente"
            if manual_dist
            else f"mediana da rota {seller_uf}→{uf}"
            + ("" if len(route) else " (rota sem dados; usada a mediana geral)")
        )
        st.caption(
            f"Distância usada: **{distance:,.0f} km** ({origem}) · taxa média global "
            f"{META['global_rate']:.1%} · threshold do orçamento default {default_thr:.1%}. "
            "Probabilidades da campeã sem calibração fina (próximo passo registrado); "
            "a leitura recomendada é o percentil de risco."
        )
