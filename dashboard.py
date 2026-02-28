import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import DataLoader
from src.qwan_regime_model import RegimeAwareQWAN
from src.backtest_institutional import InstitutionalBacktest
from src.attribution import regime_attribution
from src.phi_surface import phi_surface
from src.stress_test import stress_test
from src.montecarlo_equity import monte_carlo_equity


# ==========================================================
# CONFIGURAÇÃO
# ==========================================================

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
            "🎲 Monte Carlo"
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

    except Exception as e:
        st.error(f"Erro: {e}")