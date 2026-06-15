"""
Page 5 — Complementos
Greeks (all orders), volatility smile, IV surface, put-call parity analysis.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import BSMEngine, parity_check, intrinsic_time_value

st.set_page_config(page_title="Complementos · VQD", page_icon="∂", layout="wide")

dark = st.session_state.get("dark_mode", True)
bg        = "#0d0f14" if dark else "#f4f6fb"
surface   = "#131720" if dark else "#ffffff"
card      = "#1a1f2e" if dark else "#ffffff"
border    = "#2a3040" if dark else "#dde3ef"
text_main = "#e8eaf0" if dark else "#1a2035"
text_sub  = "#8892a4" if dark else "#6b7a99"
accent    = "#4f8ef7" if dark else "#2563eb"
grid_col  = "#1e2535" if dark else "#e8edf5"
plot_bg   = "#131720" if dark else "#ffffff"
paper_bg  = "#0d0f14" if dark else "#f4f6fb"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
html, body, [data-testid="stApp"] {{ background-color: {bg} !important; color: {text_main} !important; font-family: 'Inter', sans-serif; }}
[data-testid="stSidebar"] {{ background-color: {surface} !important; border-right: 1px solid {border} !important; }}
[data-testid="stSidebar"] * {{ color: {text_main} !important; }}
.main .block-container {{ padding-top: 1.5rem; max-width: 1200px; }}
.page-title {{ font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: {text_main}; }}
.page-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.16em; color: {accent}; text-transform: uppercase; margin-bottom: 0.5rem; }}
.page-sub {{ font-size: 0.9rem; color: {text_sub}; margin-bottom: 1.5rem; }}
.greek-card {{ background: {card}; border: 1px solid {border}; border-radius: 8px; padding: 1rem 1.1rem; text-align: center; }}
.greek-symbol {{ font-size: 1.5rem; font-weight: 700; color: {accent}; margin-bottom: 0.1rem; }}
.greek-name {{ font-size: 0.7rem; color: {text_sub}; letter-spacing: 0.06em; margin-bottom: 0.3rem; }}
.greek-call {{ font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; color: {text_main}; }}
.greek-put {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: {text_sub}; }}
.section-header {{ font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.16em; text-transform: uppercase; color: {text_sub}; margin: 1.5rem 0 0.8rem 0; }}
[data-testid="metric-container"] {{ background: {card} !important; border: 1px solid {border} !important; border-radius: 8px !important; padding: 0.8rem 1rem !important; }}
[data-testid="metric-container"] label {{ color: {text_sub} !important; font-size: 0.75rem !important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {text_main} !important; font-family: 'JetBrains Mono', monospace !important; }}
.stTabs [data-baseweb="tab"] {{ color: {text_sub} !important; }}
.stTabs [aria-selected="true"] {{ color: {accent} !important; border-bottom-color: {accent} !important; }}
label {{ color: {text_sub} !important; font-size: 0.82rem !important; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ∂ VQD")
    st.caption("Valuación Cuantitativa de Derivados")
    st.divider()
    dark_toggle = st.toggle("Modo oscuro", value=dark, key="theme_toggle")
    if dark_toggle != dark:
        st.session_state.dark_mode = dark_toggle
        st.rerun()
    st.divider()
    mode = st.radio("Modo de visualización", ["Básico", "Avanzado"], index=0)

st.markdown('<div class="page-eyebrow">Análisis Avanzado · Página 5</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Complementos</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Griegas de primer y segundo orden, sonrisa de volatilidad, superficie IV, IV implícita desde precio de mercado, paridad put-call.</div>', unsafe_allow_html=True)
st.divider()

# ── Shared params ──
with st.expander("⚙️ Parámetros base", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        S     = st.number_input("S — precio subyacente", value=100.0, step=1.0, format="%.2f")
        K     = st.number_input("K — precio de ejercicio", value=100.0, step=1.0, format="%.2f")
    with c2:
        T     = st.number_input("T — tiempo (años)", value=1.0, min_value=0.01, step=0.05, format="%.3f")
        r     = st.number_input("r — tasa (%)", value=5.0, step=0.25, format="%.2f") / 100
    with c3:
        sigma = st.number_input("σ — volatilidad (%)", value=20.0, min_value=0.1, step=0.5, format="%.2f") / 100
        q     = st.number_input("q — dividendo (%)", value=0.0, step=0.1, format="%.2f") / 100

bsm = BSMEngine(S, K, T, r, sigma, q)
call = bsm.call_price()
put  = bsm.put_price()
g    = bsm.greeks()

def plotly_layout(title=""):
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
        font=dict(family="Inter", color=text_main, size=12),
        title=dict(text=title, font=dict(size=14, color=text_main)),
        xaxis=dict(gridcolor=grid_col),
        yaxis=dict(gridcolor=grid_col),
        margin=dict(l=40, r=20, t=40, b=40),
    )

tab1, tab2, tab3, tab4 = st.tabs(["Griegas completas", "Sonrisa de volatilidad", "IV implícita", "Paridad put-call"])

# ─────────────────────────────
# TAB 1: Greeks
# ─────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Griegas de primer orden</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    greeks_1st = [
        ("Δ", "Delta",  f"{g['call_delta']:.4f}", f"{g['put_delta']:.4f}"),
        ("Θ", "Theta",  f"{g['call_theta']:.5f}", f"{g['put_theta']:.5f}"),
        ("ν", "Vega",   f"{g['vega']:.5f}", "(sym)"),
        ("ρ", "Rho",    f"{g['call_rho']:.5f}", f"{g['put_rho']:.5f}"),
        ("Γ", "Gamma",  f"{g['gamma']:.6f}", "(sym)"),
    ]
    for col, (sym, name, cv, pv) in zip(cols, greeks_1st):
        with col:
            st.markdown(f"""
            <div class="greek-card">
                <div class="greek-symbol">{sym}</div>
                <div class="greek-name">{name}</div>
                <div class="greek-call">Call: {cv}</div>
                <div class="greek-put">Put: {pv}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Griegas de segundo orden</div>', unsafe_allow_html=True)
    cols2 = st.columns(3)
    greeks_2nd = [
        ("Vanna",  "∂Δ/∂σ",        f"{g['vanna']:.5f}",      "(sym)"),
        ("Vomma",  "∂²C/∂σ²",      f"{g['volga']:.5f}",      "(sym)"),
        ("Charm",  "∂Δ/∂t (día)",   f"{g['call_charm']:.7f}", "—"),
    ]
    for col, (name, desc, cv, pv) in zip(cols2, greeks_2nd):
        with col:
            st.markdown(f"""
            <div class="greek-card">
                <div class="greek-symbol" style="font-size:1.1rem">{name}</div>
                <div class="greek-name">{desc}</div>
                <div class="greek-call">Call: {cv}</div>
                <div class="greek-put">Put: {pv}</div>
            </div>
            """, unsafe_allow_html=True)

    if mode == "Avanzado":
        st.markdown('<div class="section-header">Perfil de griegas vs S</div>', unsafe_allow_html=True)
        greek_choice = st.selectbox("Graficar griega", ["Delta", "Gamma", "Theta", "Vega", "Rho", "Vanna", "Vomma"])
        xs = np.linspace(max(S * 0.5, 1), S * 1.5, 200)
        greek_map = {
            "Delta":  (lambda eng: eng.greeks()["call_delta"], lambda eng: eng.greeks()["put_delta"]),
            "Gamma":  (lambda eng: eng.greeks()["gamma"],      None),
            "Theta":  (lambda eng: eng.greeks()["call_theta"], lambda eng: eng.greeks()["put_theta"]),
            "Vega":   (lambda eng: eng.greeks()["vega"],       None),
            "Rho":    (lambda eng: eng.greeks()["call_rho"],   lambda eng: eng.greeks()["put_rho"]),
            "Vanna":  (lambda eng: eng.greeks()["vanna"],      None),
            "Vomma":  (lambda eng: eng.greeks()["volga"],      None),
        }
        fn_call, fn_put = greek_map[greek_choice]
        vals_call = [fn_call(BSMEngine(x, K, T, r, sigma, q)) for x in xs]
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=xs, y=vals_call, name=f"{greek_choice} (Call)", line=dict(color=accent, width=2)))
        if fn_put:
            vals_put = [fn_put(BSMEngine(x, K, T, r, sigma, q)) for x in xs]
            fig_g.add_trace(go.Scatter(x=xs, y=vals_put, name=f"{greek_choice} (Put)", line=dict(color="#a78bfa", width=2)))
        fig_g.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
        fig_g.add_vline(x=S, line_dash="dot", line_color="#34d399", annotation_text="S")
        fig_g.update_layout(**plotly_layout(f"{greek_choice} vs Precio subyacente"), height=340)
        fig_g.update_xaxes(title="S")
        st.plotly_chart(fig_g, use_container_width=True)

# ─────────────────────────────
# TAB 2: Vol Smile
# ─────────────────────────────
with tab2:
    st.markdown(f"<div style='font-size:0.82rem;color:{text_sub};margin-bottom:0.8rem'>Volatilidad implícita derivada de precios BSM. En un mundo BSM puro la superficie es plana; en mercados reales, inclinada (skew) o en forma de sonrisa.</div>", unsafe_allow_html=True)

    col_sm1, col_sm2 = st.columns([1, 2])
    with col_sm1:
        strike_range = st.slider("Rango de strikes (% de S)", 50, 90, 70, 5)
        vol_skew = st.slider("Skew de vol (pendiente manual, %)", -20, 20, 0, 1)
        vol_smile_param = st.slider("Smile (curvatura, %)", 0, 30, 10, 1)

    with col_sm2:
        strikes_plot = np.linspace(S * (strike_range / 100), S * (2 - strike_range / 100), 80)
        moneyness = np.log(strikes_plot / S)

        # Simulate a realistic smile: flat BSM + manual skew + smile curvature
        iv_flat = np.full(len(strikes_plot), sigma * 100)
        iv_skew = iv_flat + vol_skew * moneyness
        iv_smile = iv_skew + vol_smile_param * (moneyness**2)

        # Also compute BSM-inverted IV from BSM prices (should be flat)
        bsm_prices = [BSMEngine(S, k, T, r, sigma, q).call_price() for k in strikes_plot]
        bsm_inv_bsm = BSMEngine(S, K, T, r, sigma, q)
        ivs_inverted = [bsm_inv_bsm.implied_vol(p, "call") * 100 for p in bsm_prices]

        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(x=strikes_plot, y=iv_smile, name="IV con skew/smile",
                                        line=dict(color=accent, width=2.5)))
        fig_smile.add_trace(go.Scatter(x=strikes_plot, y=iv_flat, name="BSM flat",
                                        line=dict(color="#f87171", width=1.5, dash="dash")))
        fig_smile.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
        fig_smile.add_vline(x=S, line_dash="dot", line_color="#34d399", annotation_text="S")
        fig_smile.update_layout(**plotly_layout("Sonrisa / Skew de Volatilidad"), height=360)
        fig_smile.update_xaxes(title="Strike K")
        fig_smile.update_yaxes(title="Volatilidad Implícita (%)")
        st.plotly_chart(fig_smile, use_container_width=True)

    if mode == "Avanzado":
        # IV surface over maturity x strike
        st.markdown('<div class="section-header">Superficie IV (maturity × strike)</div>', unsafe_allow_html=True)
        maturities_surf = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
        strikes_surf    = np.linspace(S * 0.7, S * 1.3, 30)
        Z = np.zeros((len(maturities_surf), len(strikes_surf)))

        for i, t_m in enumerate(maturities_surf):
            for j, k_s in enumerate(strikes_surf):
                mn = np.log(k_s / S)
                base = sigma * 100
                skew_contrib  = vol_skew * mn / np.sqrt(t_m)
                smile_contrib = vol_smile_param * (mn**2) / t_m
                term_struct   = -2 * (1 - np.exp(-t_m))  # slight term structure
                Z[i, j] = base + skew_contrib + smile_contrib + term_struct

        fig_surf = go.Figure(data=[go.Surface(
            z=Z, x=strikes_surf, y=maturities_surf,
            colorscale="Blues" if dark else "RdBu",
        )])
        fig_surf.update_layout(
            paper_bgcolor=paper_bg,
            font=dict(color=text_main),
            scene=dict(
                xaxis=dict(title="Strike", backgroundcolor=plot_bg, gridcolor=grid_col),
                yaxis=dict(title="Madurez (años)", backgroundcolor=plot_bg, gridcolor=grid_col),
                zaxis=dict(title="IV (%)", backgroundcolor=plot_bg, gridcolor=grid_col),
            ),
            title=dict(text="Superficie de Volatilidad Implícita", font=dict(color=text_main, size=14)),
            height=480,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_surf, use_container_width=True)

# ─────────────────────────────
# TAB 3: Implied Vol Solver
# ─────────────────────────────
with tab3:
    st.markdown(f"<div style='font-size:0.82rem;color:{text_sub};margin-bottom:1rem'>Dado un precio de mercado observado, calcula la volatilidad implícita que lo reproduce bajo BSM. Método: Brent (bisección acelerada).</div>", unsafe_allow_html=True)

    col_iv1, col_iv2 = st.columns(2)
    with col_iv1:
        market_price = st.number_input("Precio de mercado observado", value=round(call, 2), min_value=0.001, step=0.01, format="%.4f")
        opt_type_iv  = st.radio("Tipo de opción", ["call", "put"], horizontal=True)

    bsm_iv_eng = BSMEngine(S, K, T, r, 0.3, q)
    iv_result  = bsm_iv_eng.implied_vol(market_price, opt_type_iv)

    with col_iv2:
        if not np.isnan(iv_result):
            st.markdown(f"""
            <div style='background:{card};border:1px solid {border};border-radius:10px;padding:1.3rem 1.5rem'>
                <div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.08em'>Volatilidad Implícita</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:2.2rem;font-weight:600;color:{accent}'>{iv_result*100:.4f}%</div>
                <div style='font-size:0.8rem;color:{text_sub};margin-top:0.3rem'>σ = {iv_result:.6f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No se encontró solución. Verifica que el precio sea arbitrage-free.")

    if mode == "Avanzado" and not np.isnan(iv_result):
        # Verify: BSM price with solved IV should equal market price
        check_price = BSMEngine(S, K, T, r, iv_result, q).call_price() if opt_type_iv == "call" else BSMEngine(S, K, T, r, iv_result, q).put_price()
        st.markdown(f"""
        <div style='background:{card};border:1px solid {border};border-radius:6px;padding:0.6rem 1rem;font-size:0.8rem;margin-top:0.5rem'>
            Verificación: BSM(σ_IV) = <span style='font-family:JetBrains Mono,monospace;color:{text_main}'>${check_price:.6f}</span>
            &nbsp;·&nbsp; Error = <span style='font-family:JetBrains Mono,monospace;color:#34d399'>{abs(check_price - market_price):.2e}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────
# TAB 4: Put-Call Parity
# ─────────────────────────────
with tab4:
    st.markdown(f"<div style='font-size:0.82rem;color:{text_sub};margin-bottom:1rem'>La paridad put-call es una relación de no-arbitraje fundamental: C − P = S·e<sup>−qT</sup> − K·e<sup>−rT</sup></div>", unsafe_allow_html=True)

    parity = parity_check(call, put, S, K, T, r, q)
    iv_c   = intrinsic_time_value(call, S, K, "call")
    iv_p   = intrinsic_time_value(put,  S, K, "put")

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("C − P (LHS)", f"${parity['lhs']:.6f}")
    col_p2.metric("Se⁻ᵍᵀ − Ke⁻ʳᵀ (RHS)", f"${parity['rhs']:.6f}")
    col_p3.metric("Residuo |LHS−RHS|", f"${parity['residual']:.2e}",
                   delta=("✓ Arbitrage-free" if parity["residual"] < 1e-4 else "✗ Violación detectada"),
                   delta_color="normal" if parity["residual"] < 1e-4 else "inverse")

    if mode == "Avanzado":
        st.markdown('<div class="section-header">Paridad a través de strikes</div>', unsafe_allow_html=True)
        strikes_par = np.linspace(max(S * 0.6, 1), S * 1.4, 80)
        lhs_arr, rhs_arr, residuals = [], [], []
        for k in strikes_par:
            be = BSMEngine(S, k, T, r, sigma, q)
            c_k = be.call_price()
            p_k = be.put_price()
            par = parity_check(c_k, p_k, S, k, T, r, q)
            lhs_arr.append(par["lhs"])
            rhs_arr.append(par["rhs"])
            residuals.append(par["residual"])

        fig_par = go.Figure()
        fig_par.add_trace(go.Scatter(x=strikes_par, y=lhs_arr, name="C − P", line=dict(color=accent, width=2)))
        fig_par.add_trace(go.Scatter(x=strikes_par, y=rhs_arr, name="Se⁻ᵍᵀ − Ke⁻ʳᵀ",
                                      line=dict(color="#34d399", width=2, dash="dash")))
        fig_par.update_layout(**plotly_layout("Paridad Put-Call vs Strike"), height=320)
        fig_par.update_xaxes(title="Strike K")
        fig_par.update_yaxes(title="Valor ($)")
        st.plotly_chart(fig_par, use_container_width=True)
