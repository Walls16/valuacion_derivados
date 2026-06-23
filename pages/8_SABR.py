"""
Page 8 — SABR Stochastic Alpha Beta Rho
Hagan et al. (2002) lognormal vol approximation with calibration to market smile.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import SABREngine, BSMEngine

st.set_page_config(page_title="SABR · VQD", page_icon="d", layout="wide")

dark      = st.session_state.get("dark_mode", True)
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
html, body, [data-testid="stApp"] {{ background-color:{bg}!important; color:{text_main}!important; font-family:'Inter',sans-serif; }}
[data-testid="stSidebar"] {{ background-color:{surface}!important; border-right:1px solid {border}!important; }}
[data-testid="stSidebar"] * {{ color:{text_main}!important; }}
.main .block-container {{ padding-top:1.5rem; max-width:1100px; }}
.page-title {{ font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:{text_main}; }}
.page-eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.16em; color:{accent}; text-transform:uppercase; margin-bottom:0.5rem; }}
.page-sub {{ font-size:0.9rem; color:{text_sub}; margin-bottom:1.5rem; }}
.formula-box {{ background:{card}; border:1px solid {border}; border-left:3px solid {accent}; border-radius:8px; padding:1.2rem 1.4rem; margin:0.8rem 0 1.2rem 0; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:{text_sub}; line-height:1.9; }}
.result-card {{ background:{card}; border:1px solid {border}; border-radius:10px; padding:1.3rem 1.5rem; }}
.result-value {{ font-family:'JetBrains Mono',monospace; font-size:1.6rem; font-weight:600; color:{text_main}; }}
.result-label {{ font-size:0.72rem; color:{text_sub}; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.2rem; }}
[data-testid="metric-container"] {{ background:{card}!important; border:1px solid {border}!important; border-radius:8px!important; padding:0.8rem 1rem!important; }}
[data-testid="metric-container"] label {{ color:{text_sub}!important; font-size:0.75rem!important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color:{text_main}!important; font-family:'JetBrains Mono',monospace!important; }}
.stTabs [data-baseweb="tab"] {{ color:{text_sub}!important; }}
.stTabs [aria-selected="true"] {{ color:{accent}!important; border-bottom-color:{accent}!important; }}
label {{ color:{text_sub}!important; font-size:0.82rem!important; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### d VQD")
    st.caption("Valuacion Cuantitativa de Derivados")
    st.divider()
    dark_toggle = st.toggle("Modo oscuro", value=dark, key="theme_toggle")
    if dark_toggle != dark:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

def plotly_layout(title=""):
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
        font=dict(family="Inter", color=text_main, size=12),
        title=dict(text=title, font=dict(size=14, color=text_main)),
        xaxis=dict(gridcolor=grid_col),
        yaxis=dict(gridcolor=grid_col),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

# ── Header ──
st.markdown('<div class="page-eyebrow">Vol Estocastica · Pagina 8</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Modelo SABR</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Hagan, Kumar, Lesniewski & Woodward (2002). Estandar de mercado en renta fija y FX. Produce smiles y skews sin necesidad de calibrar una vol por strike.</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="formula-box">
    <strong style="color:{text_main}">Dinamica SABR</strong><br><br>
    dF = &sigma;(t) &middot; F<sup>&beta;</sup> &middot; dW<sub>1</sub><br>
    d&sigma; = &alpha; &middot; &sigma; &middot; dW<sub>2</sub><br>
    corr(dW<sub>1</sub>, dW<sub>2</sub>) = &rho;<br><br>
    <strong style="color:{text_main}">Vol implicita Hagan et al. (aproximacion analitica)</strong><br><br>
    &sigma;<sub>LN</sub>(K) &asymp; [&alpha; / (FK)<sup>(1-&beta;)/2</sup> &middot; (z/x(z))] &middot; C(T)<br><br>
    z = (&nu;/&alpha;)(FK)<sup>(1-&beta;)/2</sup> ln(F/K) &nbsp;&nbsp;
    x(z) = ln[(1 - 2&rho;z + z&sup2;)<sup>1/2</sup> + z - &rho;) / (1 - &rho;)]<br><br>
    <span style="color:{text_sub}">&beta; = 0: modelo normal | &beta; = 1: modelo lognormal (BSM) | 0 &lt; &beta; &lt; 1: CEV intermedio</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{card};border:1px solid {border};border-radius:10px;padding:1.4rem 1.6rem;margin:0 0 1.2rem 0;line-height:1.85">
    <div style="font-size:0.7rem;font-family:'JetBrains Mono',monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.9rem">
        Por que SABR — el estandar de mercado en tasas y FX
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        SABR (Stochastic Alpha Beta Rho) fue publicado en 2002 y rapidamente se convirtio en el modelo dominante
        para opciones sobre tasas de interes (caps, floors, swaptions) y divisas.
        Su atractivo principal es doble: produce una <strong style="color:{text_main}">formula analitica cerrada para la vol implicita</strong>
        (no requiere integracion numerica como Heston), y sus 4 parametros tienen
        interpretacion economica directa y son calibrables a la sonrisa de mercado en segundos.
    </p>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        El parametro <strong style="color:{text_main}">&beta;</strong> controla la naturaleza del proceso subyacente:
        &beta;=0 produce un proceso aritmetico (Bachelier) donde la vol es absoluta,
        &beta;=1 produce un proceso lognormal donde la vol es relativa (como BSM),
        y valores intermedios dan modelos CEV (Constant Elasticity of Variance).
        En tasas de interes, &beta;=0.5 es el valor mas comun; en FX, &beta;=1.
    </p>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem 1.5rem">
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:'JetBrains Mono',monospace">&alpha;</span>
            <span style="color:{text_sub}"> — volatilidad inicial de F. Controla el nivel absoluto de la sonrisa.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:'JetBrains Mono',monospace">&beta;</span>
            <span style="color:{text_sub}"> — elasticidad CEV. Fijo por convencion de mercado (no calibrado).</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:'JetBrains Mono',monospace">&rho;</span>
            <span style="color:{text_sub}"> — correlacion F-sigma. Controla el skew (asimetria) de la sonrisa.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:'JetBrains Mono',monospace">&nu;</span>
            <span style="color:{text_sub}"> — vol de vol. Controla la curvatura (smile) de la sonrisa.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Inputs ──
col_in, col_out = st.columns([1, 1.6], gap="large")

with col_in:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Parametros</div>", unsafe_allow_html=True)

    PRESETS_SABR = {
        "Personalizado":             None,
        "Equity (SPX-like)":         dict(alpha=0.3, beta=0.5, rho=-0.3, nu=0.6),
        "FX (EUR/USD-like)":         dict(alpha=0.2, beta=1.0, rho= 0.0, nu=0.4),
        "Tasas (swaption-like)":     dict(alpha=0.05, beta=0.5, rho=-0.2, nu=0.3),
        "Smile simetrico":           dict(alpha=0.25, beta=0.5, rho= 0.0, nu=0.8),
        "Skew negativo pronunciado": dict(alpha=0.25, beta=0.5, rho=-0.7, nu=0.3),
    }
    preset = st.selectbox("Perfil de parametros", list(PRESETS_SABR.keys()))
    _p = PRESETS_SABR[preset] or {}

    S     = st.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K     = st.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T     = st.number_input("Tiempo al vencimiento T (anos)", value=1.0, min_value=0.01, max_value=10.0, step=0.05, format="%.3f")
    r     = st.number_input("Tasa libre de riesgo r (%)", value=5.0, step=0.25, format="%.2f") / 100
    q     = st.number_input("Dividendo continuo q (%)", value=0.0, step=0.1, format="%.2f") / 100
    F     = S * np.exp((r - q) * T)

    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin:0.8rem 0 0.4rem'>Parametros SABR</div>", unsafe_allow_html=True)
    alpha = st.slider("alpha — vol inicial", 0.01, 2.0, float(_p.get("alpha", 0.30)), 0.01)
    beta  = st.slider("beta — elasticidad CEV", 0.0, 1.0, float(_p.get("beta",  0.5)),  0.01)
    rho   = st.slider("rho — correlacion F-sigma", -0.99, 0.99, float(_p.get("rho", -0.3)), 0.01)
    nu    = st.slider("nu — vol de vol", 0.01, 3.0, float(_p.get("nu",   0.4)),  0.01)

    F_val = S * np.exp((r - q) * T)
    st.markdown(f"""
    <div style='background:{card};border:1px solid {border};border-radius:6px;padding:0.7rem 1rem;margin-top:0.5rem;font-size:0.8rem'>
        <span style='color:{text_sub}'>Forward F = S&middot;e<sup>(r-q)T</sup> = </span>
        <span style='font-family:"JetBrains Mono",monospace;color:{text_main}'>${F_val:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

sabr   = SABREngine(F=F_val, K=K, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu, r=r, q=q, S=S)
iv_atm = sabr.implied_vol()
call   = sabr.call_price()
put_p  = sabr.put_price()
bsm_call = BSMEngine(S, K, T, r, iv_atm, q).call_price()

with col_out:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Resultados</div>", unsafe_allow_html=True)

    rc, rp = st.columns(2)
    with rc:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Call europeo</div>
            <div class="result-value">${call:.4f}</div>
            <div style='font-size:0.78rem;color:{text_sub};margin-top:0.3rem'>via BSM(IV_SABR)</div>
        </div>""", unsafe_allow_html=True)
    with rp:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Put europeo</div>
            <div class="result-value">${put_p:.4f}</div>
            <div style='font-size:0.78rem;color:{text_sub};margin-top:0.3rem'>via BSM(IV_SABR)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("IV SABR (strike K)", f"{iv_atm*100:.3f}%")
    m2.metric("F/K (moneyness)", f"{F_val/K:.4f}")
    m3.metric("Diff Call vs BSM(alpha)", f"${call - BSMEngine(S,K,T,r,alpha,q).call_price():.4f}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Sonrisa SABR", "Efecto de parametros", "Calibracion manual", "Comparativa beta"])

with tab1:
    strikes_plot = np.linspace(max(S*0.65, 1), S*1.35, 100)
    ivs_sabr = sabr.vol_smile(strikes_plot) * 100
    ivs_flat = np.full(len(strikes_plot), alpha*100)

    show_moneyness = st.checkbox("Eje en moneyness ln(K/F)", value=False)
    x_ax = np.log(strikes_plot / F_val) if show_moneyness else strikes_plot
    x_K  = np.log(K / F_val)            if show_moneyness else K
    x_F  = 0.0                           if show_moneyness else F_val
    x_label = "Moneyness ln(K/F)" if show_moneyness else "Strike K"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ax, y=ivs_sabr, name="SABR IV",
                              line=dict(color=accent, width=2.5)))
    fig.add_trace(go.Scatter(x=x_ax, y=ivs_flat, name="alpha (nivel plano)",
                              line=dict(color="#f87171", width=1.5, dash="dash")))
    fig.add_vline(x=x_K, line_dash="dot", line_color=text_sub, annotation_text="K")
    fig.add_vline(x=x_F, line_dash="dot", line_color="#34d399", annotation_text="F (ATM)")
    fig.update_layout(**plotly_layout("Sonrisa de Volatilidad SABR"), height=400)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="IV Lognormal (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        # rho sweep
        rhos = np.linspace(-0.9, 0.9, 7)
        fig_r = go.Figure()
        for r_val in rhos:
            s_t = SABREngine(F_val, K, T, alpha, beta, r_val, nu)
            ivs = s_t.vol_smile(strikes_plot) * 100
            fig_r.add_trace(go.Scatter(x=strikes_plot, y=ivs,
                                        name=f"rho={r_val:.1f}",
                                        line=dict(width=1.5)))
        fig_r.add_vline(x=F_val, line_dash="dot", line_color=text_sub, annotation_text="F")
        _rl = plotly_layout("Efecto de rho — skew")
        _rl["height"] = 340
        _rl["showlegend"] = True
        fig_r.update_layout(**_rl)
        fig_r.update_xaxes(title="Strike K")
        fig_r.update_yaxes(title="IV (%)")
        st.plotly_chart(fig_r, use_container_width=True)

    with col_p2:
        # nu sweep
        nus = [0.1, 0.3, 0.5, 0.8, 1.2]
        fig_n = go.Figure()
        for n_val in nus:
            s_t = SABREngine(F_val, K, T, alpha, beta, rho, n_val)
            ivs = s_t.vol_smile(strikes_plot) * 100
            fig_n.add_trace(go.Scatter(x=strikes_plot, y=ivs,
                                        name=f"nu={n_val}",
                                        line=dict(width=1.5)))
        fig_n.add_vline(x=F_val, line_dash="dot", line_color=text_sub, annotation_text="F")
        _nl = plotly_layout("Efecto de nu — curvatura (smile)")
        _nl["height"] = 340
        fig_n.update_layout(**_nl)
        fig_n.update_xaxes(title="Strike K")
        fig_n.update_yaxes(title="IV (%)")
        st.plotly_chart(fig_n, use_container_width=True)

with tab3:
    st.markdown(f"<div style='font-size:0.85rem;color:{text_sub};margin-bottom:1rem'>Ingresa puntos de la sonrisa de mercado observada (strike, IV%) para calibrar alpha, rho y nu con beta fijo.</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Puntos de mercado (strike, IV%)</div>", unsafe_allow_html=True)

    n_points = st.slider("Numero de strikes de mercado", 3, 10, 5)
    mkt_strikes, mkt_ivs = [], []
    default_ks = np.linspace(S*0.85, S*1.15, n_points)
    default_ivs_mkt = [22.0, 20.5, 20.0, 20.5, 21.5][:n_points] + [20.0]*(n_points - min(n_points,5))

    cols_mkt = st.columns(min(n_points, 5))
    for i in range(n_points):
        col = cols_mkt[i % 5]
        with col:
            k_inp  = st.number_input(f"K_{i+1}", value=float(round(default_ks[i],1)), step=1.0, format="%.1f", key=f"mkt_k_{i}")
            iv_inp = st.number_input(f"IV_{i+1} (%)", value=float(default_ivs_mkt[i] if i < len(default_ivs_mkt) else 20.0),
                                      min_value=0.1, max_value=200.0, step=0.1, format="%.2f", key=f"mkt_iv_{i}")
            mkt_strikes.append(k_inp)
            mkt_ivs.append(iv_inp / 100)

    beta_cal = st.slider("beta fijo para calibracion", 0.0, 1.0, beta, 0.05)

    if st.button("Calibrar SABR"):
        with st.spinner("Calibrando..."):
            result = sabr.calibrate(mkt_strikes, mkt_ivs, beta=beta_cal)

        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("alpha calibrado", f"{result['alpha']:.4f}")
        col_c2.metric("rho calibrado", f"{result['rho']:.4f}")
        col_c3.metric("nu calibrado", f"{result['nu']:.4f}")
        col_c4.metric("RMSE", f"{result['rmse']*100:.4f}%")

        # Plot calibrated vs market
        sabr_cal = SABREngine(F_val, K, T, result["alpha"], beta_cal, result["rho"], result["nu"], r, q, S)
        ivs_cal  = sabr_cal.vol_smile(strikes_plot) * 100

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=strikes_plot, y=ivs_cal, name="SABR calibrado",
                                      line=dict(color="#34d399", width=2.5)))
        fig_cal.add_trace(go.Scatter(x=strikes_plot, y=ivs_sabr, name="SABR inicial",
                                      line=dict(color=text_sub, width=1.5, dash="dash")))
        fig_cal.add_trace(go.Scatter(x=mkt_strikes, y=[iv*100 for iv in mkt_ivs],
                                      name="Mercado", mode="markers",
                                      marker=dict(color="#f59e0b", size=10, symbol="diamond")))
        fig_cal.update_layout(**plotly_layout("SABR calibrado vs mercado"), height=360)
        fig_cal.update_xaxes(title="Strike K")
        fig_cal.update_yaxes(title="IV (%)")
        st.plotly_chart(fig_cal, use_container_width=True)

with tab4:
    betas_compare = [0.0, 0.25, 0.5, 0.75, 1.0]
    beta_colors   = [text_sub, "#f59e0b", accent, "#a78bfa", "#34d399"]
    fig_b = go.Figure()
    for b_val, b_col in zip(betas_compare, beta_colors):
        s_b  = SABREngine(F_val, K, T, alpha, b_val, rho, nu)
        ivs_b = s_b.vol_smile(strikes_plot) * 100
        label = f"beta={b_val} ({'Normal' if b_val==0 else ('Lognormal' if b_val==1 else 'CEV')})"
        fig_b.add_trace(go.Scatter(x=strikes_plot, y=ivs_b, name=label,
                                    line=dict(color=b_col, width=2)))
    fig_b.add_vline(x=F_val, line_dash="dot", line_color=text_sub, annotation_text="F (ATM)")
    _bl = plotly_layout("Sonrisa SABR para distintos valores de beta")
    _bl["height"] = 400
    fig_b.update_layout(**_bl)
    fig_b.update_xaxes(title="Strike K")
    fig_b.update_yaxes(title="IV Lognormal (%)")
    st.plotly_chart(fig_b, use_container_width=True)
    st.caption("beta=0: proceso normal (Bachelier). beta=1: proceso lognormal (BSM). Beta afecta la pendiente estructural del skew independientemente de rho.")
