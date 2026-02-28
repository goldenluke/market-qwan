import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import webbrowser
from src.data_loader import DataLoader
from src.qwan_regime_model import RegimeAwareQWAN
from src.backtest_institutional import InstitutionalBacktest
from src.attribution import regime_attribution
from src.phi_surface import phi_surface
from src.stress_test import stress_test
from src.montecarlo_equity import monte_carlo_equity
from src.quant_fund_engine import QuantFundEngine
from src.qwan_forecast_model import QWANForecastModel
import numpy as np
import numpy as np
import networkx as nx
from src.qwan_forecast_model import QWANForecastModel
import os
import subprocess
import sys

@st.cache_resource
def load_forecast_model(returns):
    model = QWANForecastModel(horizon=30, input_size=126)
    model.fit(returns)
    return model

def correlation_network(returns_matrix, threshold=0.5):
    """
    Cria uma rede de correlação a partir da matriz de retornos.
    Nós = ativos
    Arestas = correlação acima do threshold
    """

    corr = np.corrcoef(returns_matrix.T)
    n_assets = corr.shape[0]

    G = nx.Graph()

    # adiciona nós
    for i in range(n_assets):
        G.add_node(i)

    # adiciona arestas se |correlação| > threshold
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if abs(corr[i, j]) > threshold:
                G.add_edge(i, j, weight=corr[i, j])

    return G


def network_stress_index(G):
    """
    Mede estresse como densidade da rede.
    Quanto mais conectada, maior o risco sistêmico.
    """

    if len(G.nodes) <= 1:
        return 0.0

    density = nx.density(G)

    return float(density)
def rolling_susceptibility(market_returns, window=126):
    series = pd.Series(market_returns)

    rolling_mean = series.rolling(window).mean()
    rolling_var = series.rolling(window).var()

    chi = rolling_var / (rolling_mean.abs() + 1e-8)

    return chi.values

def eigenvalue_stress(returns_matrix):
    """
    Calcula o estresse sistêmico baseado no autovalor dominante
    da matriz de correlação.
    """

    # Matriz de correlação
    corr = np.corrcoef(returns_matrix.T)

    # Autovalores
    eigenvalues = np.linalg.eigvals(corr)

    lambda_max = np.max(eigenvalues).real
    lambda_mean = np.mean(eigenvalues).real

    stress_index = lambda_max / lambda_mean

    return {
        "lambda_max": float(lambda_max),
        "lambda_mean": float(lambda_mean),
        "stress_index": float(stress_index)
    }
# ==========================================================
# CONFIGURAÇÃO
# ==========================================================

import numpy as np



st.set_page_config(
    page_title="Market-QWAN Institutional Platform",
    layout="wide"
)

st.title("🏦 Market-QWAN Institutional Quant Platform")

st.markdown("""
### 📘 Sobre o Sistema

**Leigo:**  
Este painel mostra como a estratégia tenta crescer o capital controlando riscos e se adaptando a diferentes tipos de mercado.

**Técnico:**  
Engine multi-regime com walk-forward rolling, vol targeting, stop estrutural,
benchmark comparativo, superfície funcional Φ e simulação Monte Carlo regime-switching.
""")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙ Configuração")

tickers_input = st.sidebar.text_input("Tickers", "AAPL,MSFT,GOOGL,AMZN,META")
period = st.sidebar.selectbox("Período", ["2y", "3y", "5y", "10y"], index=2)

n_regimes = st.sidebar.slider("Regimes", 2, 5, 3)
alpha = st.sidebar.slider("Alpha (Entropia)", 0.0, 2.0, 0.6)
gamma = st.sidebar.slider("Gamma (Momentum)", 0.0, 2.0, 0.5)
target_vol = st.sidebar.slider("Target Vol", 0.05, 0.30, 0.15)
stop_dd = st.sidebar.slider("Stop Estrutural", -0.5, -0.05, -0.25)

run = st.sidebar.button("🚀 Rodar Modelo")

# ==========================================================
# EXECUÇÃO
# ==========================================================

if run:

    try:

        tickers = [t.strip() for t in tickers_input.split(",")]

        with st.spinner("🔄 Processando modelo..."):

            progress_bar = st.progress(0)
            status_text = st.empty()

            # 1️⃣ Dados
            status_text.text("📥 Baixando dados...")
            loader = DataLoader(tickers=tickers, period=period)
            returns = loader.load()
            progress_bar.progress(20)

            # 2️⃣ Modelo
            status_text.text("🧠 Treinando modelo multi-regime...")
            model = RegimeAwareQWAN(
                returns=returns,
                n_regimes=n_regimes,
                alpha=alpha,
                gamma=gamma,
                target_vol=target_vol
            )
            progress_bar.progress(40)

            # 3️⃣ Backtest institucional
            status_text.text("📈 Executando walk-forward institucional...")
            bt = InstitutionalBacktest(
                returns=returns,
                model_class=lambda r: RegimeAwareQWAN(
                    r,
                    n_regimes=n_regimes,
                    alpha=alpha,
                    gamma=gamma,
                    target_vol=target_vol
                ),
                lookback=252,
                rebalance_frequency=21,
                target_vol=target_vol,
                stop_drawdown=stop_dd
            )
            progress_bar.progress(70)

            equity = bt.equity_curve
            benchmark = bt.benchmark_equity
            drawdown = bt.drawdown
            metrics = bt.compute_metrics()

            # 4️⃣ Superfície Φ
            status_text.text("🧮 Calculando superfície Φ...")
            alpha_vals, gamma_vals, Z = phi_surface(
                model,
                grid_size=25,
                mode="posterior"
            )
            progress_bar.progress(85)

            # 5️⃣ Monte Carlo
            status_text.text("🎲 Simulando Monte Carlo...")
            mc = monte_carlo_equity(model, T=252, n_sim=200)
            progress_bar.progress(100)

            status_text.text("✅ Execução concluída.")

        progress_bar.empty()
        status_text.empty()

        # ==========================================================
        # ABAS
        # ==========================================================

        tabs = st.tabs([
            "📈 Performance",
            "📊 Rolling Metrics",
            "🧠 Regimes",
            "🧮 Φ Surface",
            "⚠ Stress Test",
            "🎲 Monte Carlo",
            "🌐 Systemic Risk",  # NOVA ABA
            "📈 Forecast"
        ])

        # ==========================================================
        # 📈 PERFORMANCE
        # ==========================================================

        with tabs[0]:

            st.subheader("Equity Curve")

            st.markdown("""
            **Leigo:**  
            Mostra quanto o capital teria crescido ao longo do tempo.

            **Técnico:**  
            Retorno acumulado multiplicativo comparado ao benchmark equal-weight.
            """)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity.index, y=equity, name="QWAN"))
            fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="Benchmark"))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Drawdown")

            st.markdown("""
            **Leigo:**  
            Mostra as maiores quedas temporárias.

            **Técnico:**  
            Drawdown percentual relativo ao máximo histórico.
            """)

            st.plotly_chart(px.line(drawdown), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CAGR", round(metrics.get("cagr", 0), 3))
            col2.metric("Sharpe", round(metrics.get("sharpe", 0), 3))
            col3.metric("Sortino", round(metrics.get("sortino", 0), 3))
            col4.metric("Alpha", round(metrics.get("alpha", 0), 3))


        # ==========================================================
        # 📊 ROLLING METRICS
        # ==========================================================

        with tabs[1]:

            st.subheader("Rolling Sharpe")

            st.markdown("""
            **Leigo:**  
            Mede se a estratégia está funcionando bem recentemente.

            **Técnico:**  
            Sharpe anualizado em janela móvel de 6 meses.
            """)

            rolling_sharpe = (
                bt.portfolio_returns.rolling(126).mean() * 252 /
                (bt.portfolio_returns.rolling(126).std() * np.sqrt(252))
            )

            st.plotly_chart(px.line(rolling_sharpe), use_container_width=True)

            st.subheader("Rolling Alpha")

            st.markdown("""
            **Leigo:**  
            Mostra se estamos realmente batendo o mercado.

            **Técnico:**  
            Alpha anualizado rolling contra benchmark.
            """)

            rolling_alpha = (
                (bt.portfolio_returns - bt.benchmark_returns)
                .rolling(126).mean() * 252
            )

            st.plotly_chart(px.line(rolling_alpha), use_container_width=True)


        # ==========================================================
        # 🧠 REGIMES
        # ==========================================================

        with tabs[2]:

            st.subheader("Regime ao Longo do Tempo")

            st.markdown("""
            **Leigo:**  
            Mostra os tipos de mercado identificados automaticamente.

            **Técnico:**  
            Sequência de estados latentes inferidos via HMM.
            """)

            regime_series = pd.Series(model.hidden_states, index=returns.index)
            st.plotly_chart(px.scatter(regime_series), use_container_width=True)

            st.subheader("Attribution por Regime")

            st.markdown("""
            **Leigo:**  
            Mostra como a estratégia se comporta em cada tipo de mercado.

            **Técnico:**  
            Retorno médio e volatilidade condicional por estado latente.
            """)

            attr = regime_attribution(
                returns,
                model.hidden_states,
                bt.portfolio_returns
            )

            st.dataframe(attr)


        # ==========================================================
        # 🧮 Φ SURFACE
        # ==========================================================

        with tabs[3]:

            st.subheader("Superfície Φ")

            st.markdown("""
            **Leigo:**  
            Mostra como o desempenho muda quando ajustamos parâmetros.

            **Técnico:**  
            Paisagem do funcional Φ no espaço (α, γ).
            """)

            fig_surface = go.Figure(
                data=[go.Surface(z=Z, x=alpha_vals, y=gamma_vals)]
            )

            fig_surface.update_layout(
                scene=dict(
                    xaxis_title="Alpha",
                    yaxis_title="Gamma",
                    zaxis_title="Φ"
                )
            )

            st.plotly_chart(fig_surface, use_container_width=True)


        # ==========================================================
        # ⚠ STRESS TEST
        # ==========================================================

        with tabs[4]:

            st.subheader("Stress Test -20%")

            st.markdown("""
            **Leigo:**  
            Simula uma grande queda repentina.

            **Técnico:**  
            Choque exógeno aplicado à série de retornos.
            """)

            stress_equity = stress_test(bt.portfolio_returns, -0.2)
            st.plotly_chart(px.line(stress_equity), use_container_width=True)


        # ==========================================================
        # 🎲 MONTE CARLO
        # ==========================================================

        with tabs[5]:

            st.subheader("Monte Carlo Regime Switching")

            st.markdown("""
            **Leigo:**  
            Simula vários futuros possíveis.

            **Técnico:**  
            Simulação estocástica baseada na matriz de transição do HMM.
            """)

            fig_mc = go.Figure()
            for i in range(min(50, mc.shape[1])):
                fig_mc.add_trace(go.Scatter(y=mc.iloc[:, i], showlegend=False))

            st.plotly_chart(fig_mc, use_container_width=True)

            st.subheader("Distribuição Final")

            st.markdown("""
            **Leigo:**  
            Mostra os possíveis resultados finais.

            **Técnico:**  
            Distribuição empírica do valor final da equity simulada.
            """)

            st.plotly_chart(
                px.histogram(mc.iloc[-1], nbins=40),
                use_container_width=True
            )

            # ==========================================================
            # 🌐 SYSTEMIC RISK
            # ==========================================================

        with tabs[6]:

            st.subheader("Systemic Risk Monitor")

            st.markdown("""
            **Leigo:**  
            Mede se o mercado está se movendo de forma sincronizada e arriscada.

            **Técnico:**  
            Monitor baseado em autovalor dominante da matriz de correlação
            e densidade da rede de correlação.
            """)

            returns_matrix = returns.values

            # ----------------------------
            # Eigenvalue Stress
            # ----------------------------
            eigen = eigenvalue_stress(returns_matrix)

            # ----------------------------
            # Correlation Heatmap
            # ----------------------------
            st.subheader("Correlation Heatmap")

            corr = np.corrcoef(returns_matrix.T)

            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1
                )
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            # ----------------------------
            # Rolling Susceptibility
            # ----------------------------
            st.subheader("Rolling Market Susceptibility (χ)")

            market_returns = returns.mean(axis=1)
            chi = rolling_susceptibility(market_returns, window=126)

            st.plotly_chart(
                px.line(
                    x=returns.index,
                    y=chi,
                    labels={"x": "Date", "y": "χ"},
                    title="Rolling Susceptibility"
                ),
                use_container_width=True
            )

            # ----------------------------
            # Network Stress
            # ----------------------------
            G = correlation_network(returns_matrix, threshold=0.5)
            net_stress = network_stress_index(G)

            st.metric("Network Stress Index", round(net_stress, 2))

            # ----------------------------
            # Indicador Sistêmico Composto
            # ----------------------------
            systemic_index = (
                eigen["stress_index"] * 0.5 +
                net_stress * 0.3 +
                np.nanmean(chi[-50:]) * 0.2
            )

            st.metric("Composite Systemic Risk", round(systemic_index, 2))

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "λ_max",
                    round(eigen["lambda_max"], 2)
                )

            with col2:
                st.metric(
                    "Stress Index (λ_max / λ_mean)",
                    round(eigen["stress_index"], 2)
                )    

        with tabs[7]:

            st.subheader("Neural Forecast")
            


            st.divider()

            # 🔥 BOTÃO EXTERNO (NÃO USA st.button)
            st.markdown(
                """
                <a href="http://127.0.0.1:5050" target="_blank" style="text-decoration:none;">
                    <div style="
                        background-color:#6366f1;
                        color:white;
                        padding:12px 20px;
                        border-radius:10px;
                        font-weight:600;
                        text-align:center;
                        width:260px;
                        cursor:pointer;">
                        🚀 Abrir Neural Forecast Trainer
                    </div>
                </a>
                """,
                unsafe_allow_html=True
            )

            model_dir = "models"

            if not os.path.exists(model_dir):
                st.warning("Diretório models não encontrado.")
            else:

                model_files = [
                    f for f in os.listdir(model_dir)
                    if f.endswith(".pkl")
                ]

                if not model_files:
                    st.warning("Nenhum modelo encontrado.")
                else:

                    selected_model = st.selectbox(
                        "Selecionar modelo",
                        model_files
                    )

                    model_path = os.path.join(model_dir, selected_model)

                    model = QWANForecastModel.load(model_path)

                    forecast = model.predict()
                    forecast_series = forecast.set_index("ds").iloc[:, -1]

                    st.line_chart(forecast_series)

    except Exception as e:
        st.error(f"Erro: {e}")