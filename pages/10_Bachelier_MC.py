"""
Page 10 — Bachelier (Normal) Model & Monte Carlo con Intervalos de Confianza
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import BachelierEngine, BSMEngine, monte_carlo_bsm

st.set_page_config(page_title="Bachelier & MC · VQD", page_icon="d", layout="wide")

dark      = st.session_state.get("dark_mode", True)
bg        = "#0d0f14" if dark else "#f4f6fb"
surface   = "#131720" if dark else "#ffffff"
card      = "#1a1f2e" if dark else "#ffffff"
border    = "#2a3040" if dark else "#dde3ef"
text_main = "#e8eaf0" if dark else "#1a2035"
text_sub  = "#8892a4" if dark else "#6b7a99"
accent    = "#4f8ef7" if dark else "#2563eb"
accent2   = "#34d399"
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
.result-card {{ background:{card}; border:1px solid {border}; border-radius:10px; padding:1.1rem 1.3rem; text-align:center; }}
.result-value {{ font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:600; color:{text_main}; }}
.result-label {{ font-size:0.68rem; color:{text_sub}; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.2rem; }}
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
st.markdown('<div class="page-eyebrow">Modelo Normal · Pagina 10</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Bachelier & Monte Carlo</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">El modelo mas antiguo de valuacion de opciones (1900) y su relacion con el modelo normal moderno. Mas Monte Carlo con reduccion de varianza e intervalos de confianza.</div>', unsafe_allow_html=True)

# ── Theory ──
st.markdown(f"""
<div style="background:{card};border:1px solid {border};border-radius:10px;padding:1.4rem 1.6rem;margin:0 0 1.2rem 0;line-height:1.85">
    <div style="font-size:0.7rem;font-family:'JetBrains Mono',monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.9rem">
        Bachelier (1900) — el origen de la teoria de opciones
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        Louis Bachelier propuso en su tesis doctoral (1900) modelar el precio de un activo como
        un <strong style="color:{text_main}">movimiento browniano aritmetico</strong>: dS = sigma_n dW.
        A diferencia de BSM (que modela retornos lognormales), Bachelier modela el precio directamente como
        un proceso normal — lo que permite que el precio tome valores negativos.
        Esto lo hace inapropiado para acciones, pero es el modelo estandar para
        <strong style="color:{text_main}">tasas de interes y contratos en entornos de tasa negativa</strong>
        (LIBOR, SOFR swaps, inflacion), donde la lognormalidad es fisicamente incorrecta.
    </p>
    <p style="color:{text_sub};font-size:0.83rem;margin:0 0 0.4rem 0">
        La vol normal sigma_n se expresa en unidades absolutas (ej. $2/ano o 50bp/ano).
        La conversion aproximada ATM entre vol normal y lognormal es: sigma_LN ≈ sigma_n / F.
        En mercados de tasas, los traders cotizaban opciones en "basis points of normal vol"
        hasta la transicion al SOFR post-2021.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="formula-box">
    <strong style="color:{text_main}">Formulas de Bachelier</strong><br><br>
    dS = &sigma;<sub>n</sub> dW &nbsp;&nbsp; [proceso aritmetico — vol absoluta]<br><br>
    C = e<sup>-rT</sup>[(F-K)&middot;N(d) + &sigma;<sub>n</sub>&radic;T&middot;n(d)]<br>
    P = e<sup>-rT</sup>[(K-F)&middot;N(-d) + &sigma;<sub>n</sub>&radic;T&middot;n(d)]<br><br>
    d = (F-K) / (&sigma;<sub>n</sub>&radic;T) &nbsp;&nbsp; F = S&middot;e<sup>(r-q)T</sup><br><br>
    <span style="color:{text_sub}">n(d) = densidad normal estandar &middot; N(d) = CDF normal estandar</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Section 1: Bachelier ──
st.markdown(f"<div style='font-size:0.78rem;font-family:JetBrains Mono,monospace;letter-spacing:0.12em;text-transform:uppercase;color:{accent};margin-bottom:0.8rem'>Modelo de Bachelier</div>", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1.6], gap="large")

with col_in:
    S      = st.number_input("Spot S", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K      = st.number_input("Strike K", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T      = st.number_input("Tiempo T (anos)", value=1.0, min_value=0.01, step=0.05, format="%.3f")
    r      = st.number_input("Tasa r (%)", value=5.0, step=0.25, format="%.2f") / 100
    q      = st.number_input("Dividendo q (%)", value=0.0, step=0.1, format="%.2f") / 100
    sigma_n = st.number_input("sigma_n — vol normal (unidades absolutas, ej. $20 = 20% de S)", value=20.0,
                               min_value=0.01, max_value=S*5, step=0.5, format="%.2f")
    sigma_ln = st.number_input("sigma_LN — vol lognormal BSM (%) para comparar", value=20.0,
                                min_value=0.01, max_value=200.0, step=0.5, format="%.2f") / 100

    F_val = S * np.exp((r - q) * T)
    sigma_n_approx = sigma_ln * F_val
    st.markdown(f"""
    <div style='background:{card};border:1px solid {border};border-radius:6px;padding:0.7rem 1rem;margin-top:0.5rem;font-size:0.8rem'>
        <span style='color:{text_sub}'>Conversion ATM: sigma_n ≈ sigma_LN × F = </span>
        <span style='font-family:"JetBrains Mono",monospace;color:{text_main}'>{sigma_n_approx:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

bch = BachelierEngine(S, K, T, r, sigma_n, q)
bsm = BSMEngine(S, K, T, r, sigma_ln, q)
bch_call = bch.call_price(); bch_put = bch.put_price()
bsm_call = bsm.call_price(); bsm_put = bsm.put_price()
g_bch = bch.greeks()

with col_out:
    cols_r = st.columns(4)
    for col_r, (label, val, ref, color) in zip(cols_r, [
        ("Bachelier Call", bch_call, bsm_call, accent),
        ("Bachelier Put",  bch_put,  bsm_put,  accent),
        ("BSM Call",       bsm_call, bsm_call, text_sub),
        ("BSM Put",        bsm_put,  bsm_put,  text_sub),
    ]):
        diff = val - ref if label.startswith("Bachelier") else 0
        col_r.markdown(f"""
        <div class="result-card" style="border-top:3px solid {color}">
            <div class="result-label" style="color:{color}">{label}</div>
            <div class="result-value">${val:.4f}</div>
            <div style='font-size:0.72rem;color:{"#34d399" if diff>=0 else "#f87171"}'>
                {'vs BSM: ' + f'{diff:+.4f}' if diff!=0 else 'referencia'}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delta Call", f"{g_bch['call_delta']:.4f}")
    m2.metric("Gamma",      f"{g_bch['gamma']:.6f}")
    m3.metric("Vega",       f"{g_bch['vega']:.4f}")
    m4.metric("Theta/dia",  f"{g_bch['call_theta']:.5f}")

st.divider()

# ── Section 2: Monte Carlo ──
st.markdown(f"<div style='font-size:0.78rem;font-family:JetBrains Mono,monospace;letter-spacing:0.12em;text-transform:uppercase;color:{accent2};margin-bottom:0.8rem'>Monte Carlo con reduccion de varianza e intervalos de confianza</div>", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{card};border:1px solid {border};border-radius:10px;padding:1.4rem 1.6rem;margin:0 0 1.2rem 0;line-height:1.85">
    <div style="font-size:0.7rem;font-family:'JetBrains Mono',monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent2};margin-bottom:0.9rem">
        Reduccion de varianza — por que el MC ingenuo es ineficiente
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
        <div>
            <div style="font-size:0.72rem;color:{accent2};font-family:'JetBrains Mono',monospace;margin-bottom:0.3rem">Variables Antiteticas</div>
            <p style="font-size:0.83rem;color:{text_sub};margin:0">
                Por cada trayectoria z se genera tambien -z. Los pares (z, -z) estan negativamente
                correlacionados, lo que reduce la varianza del estimador casi a la mitad sin costo
                computacional adicional. Es la tecnica mas sencilla y siempre aplicable.
            </p>
        </div>
        <div>
            <div style="font-size:0.72rem;color:{accent2};font-family:'JetBrains Mono',monospace;margin-bottom:0.3rem">Variable de Control (BSM)</div>
            <p style="font-size:0.83rem;color:{text_sub};margin:0">
                Se usa el precio BSM (conocido analiticamente) como variable de control.
                Se estima beta_cv = Cov(payoff, BSM_path) / Var(BSM_path) y se ajusta:
                estimador_ajustado = payoff_MC - beta_cv * (payoff_BSM_path - E[BSM]).
                Puede reducir la varianza en 90-99% para opciones europeas bajo GBM.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_mc1, col_mc2 = st.columns([1, 1.8], gap="large")

with col_mc1:
    st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Configuracion MC</div>", unsafe_allow_html=True)
    mc_type     = st.radio("Tipo de opcion", ["call", "put"], horizontal=True)
    n_paths     = st.select_slider("Numero de trayectorias", [1_000, 5_000, 10_000, 50_000, 100_000, 500_000], value=50_000)
    use_anti    = st.checkbox("Variables antiteticas", value=True)
    use_cv      = st.checkbox("Variable de control (BSM)", value=True)
    mc_sigma    = st.slider("sigma BSM para MC (%)", 5.0, 80.0, float(sigma_ln*100), 0.5) / 100
    mc_seed     = st.number_input("Semilla aleatoria", value=42, min_value=0, step=1)

with col_mc2:
    with st.spinner(f"Simulando {n_paths:,} trayectorias..."):
        mc_result = monte_carlo_bsm(S, K, T, r, mc_sigma, q,
                                     option_type=mc_type,
                                     n_paths=n_paths,
                                     seed=int(mc_seed),
                                     antithetic=use_anti,
                                     control_variate=use_cv)

    bsm_ref = mc_result["bsm_price"]
    mc_price = mc_result["price"]
    mc_err   = mc_result["std_error"]
    mc_lo    = mc_result["ci_95_lo"]
    mc_hi    = mc_result["ci_95_hi"]

    # Color based on CI containing BSM
    ci_ok = mc_lo <= bsm_ref <= mc_hi
    ci_color = "#34d399" if ci_ok else "#f87171"

    st.markdown(f"""
    <div style='background:{card};border:1px solid {border};border-radius:10px;padding:1.3rem 1.5rem;margin-bottom:0.8rem'>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.8rem'>
            <div>
                <div style='font-size:0.65rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.08em'>Precio MC</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1.6rem;font-weight:600;color:{text_main}'>${mc_price:.4f}</div>
            </div>
            <div>
                <div style='font-size:0.65rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.08em'>Precio BSM analitico</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1.6rem;font-weight:600;color:{accent}'>${bsm_ref:.4f}</div>
            </div>
        </div>
        <div style='border-top:1px solid {border};margin:0.8rem 0'></div>
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;font-size:0.82rem'>
            <div><span style='color:{text_sub}'>Error std: </span><span style='font-family:"JetBrains Mono",monospace;color:{text_main}'>${mc_err:.5f}</span></div>
            <div><span style='color:{text_sub}'>|MC - BSM|: </span><span style='font-family:"JetBrains Mono",monospace;color:{text_main}'>${mc_result["error_vs_bsm"]:.5f}</span></div>
            <div><span style='color:{text_sub}'>N paths: </span><span style='font-family:"JetBrains Mono",monospace;color:{text_main}'>{n_paths:,}</span></div>
        </div>
        <div style='margin-top:0.6rem;font-size:0.82rem'>
            <span style='color:{text_sub}'>IC 95%: </span>
            <span style='font-family:"JetBrains Mono",monospace;color:{ci_color}'>
                [${mc_lo:.4f}, ${mc_hi:.4f}]
            </span>
            <span style='color:{ci_color};margin-left:0.5rem;font-size:0.75rem'>
                {'BSM dentro del IC' if ci_ok else 'BSM fuera del IC'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Bachelier vs BSM — perfiles", "Convergencia MC", "Comparativa tecnicas MC"])

with tab1:
    strikes_plot = np.linspace(max(S*0.6, 1), S*1.4, 200)
    bch_calls = [BachelierEngine(S, k, T, r, sigma_n, q).call_price() for k in strikes_plot]
    bsm_calls = [BSMEngine(S, k, T, r, sigma_ln, q).call_price() for k in strikes_plot]
    bch_puts  = [BachelierEngine(S, k, T, r, sigma_n, q).put_price() for k in strikes_plot]
    bsm_puts  = [BSMEngine(S, k, T, r, sigma_ln, q).put_price() for k in strikes_plot]

    col_pb1, col_pb2 = st.columns(2)
    with col_pb1:
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=strikes_plot, y=bch_calls, name="Bachelier Call", line=dict(color=accent, width=2.5)))
        fig_c.add_trace(go.Scatter(x=strikes_plot, y=bsm_calls, name="BSM Call", line=dict(color="#f87171", width=1.5, dash="dash")))
        fig_c.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
        fig_c.add_vline(x=S, line_dash="dot", line_color="#34d399", annotation_text="S")
        _cl = plotly_layout("Call: Bachelier vs BSM")
        _cl["height"] = 320
        fig_c.update_layout(**_cl)
        fig_c.update_xaxes(title="Strike K")
        fig_c.update_yaxes(title="Precio call")
        st.plotly_chart(fig_c, use_container_width=True)

    with col_pb2:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=strikes_plot, y=bch_puts, name="Bachelier Put", line=dict(color=accent, width=2.5)))
        fig_p.add_trace(go.Scatter(x=strikes_plot, y=bsm_puts, name="BSM Put", line=dict(color="#f87171", width=1.5, dash="dash")))
        fig_p.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
        fig_p.add_vline(x=S, line_dash="dot", line_color="#34d399", annotation_text="S")
        _pl = plotly_layout("Put: Bachelier vs BSM")
        _pl["height"] = 320
        fig_p.update_layout(**_pl)
        fig_p.update_xaxes(title="Strike K")
        fig_p.update_yaxes(title="Precio put")
        st.plotly_chart(fig_p, use_container_width=True)

    # Delta comparison
    bch_deltas = [BachelierEngine(S, k, T, r, sigma_n, q).greeks()["call_delta"] for k in strikes_plot]
    bsm_deltas = [BSMEngine(S, k, T, r, sigma_ln, q).greeks()["call_delta"] for k in strikes_plot]

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=strikes_plot, y=bch_deltas, name="Bachelier Delta", line=dict(color=accent, width=2)))
    fig_d.add_trace(go.Scatter(x=strikes_plot, y=bsm_deltas, name="BSM Delta", line=dict(color="#f87171", width=1.5, dash="dash")))
    fig_d.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
    _dl = plotly_layout("Delta call: Bachelier vs BSM")
    _dl["height"] = 280
    fig_d.update_layout(**_dl)
    fig_d.update_xaxes(title="Strike K")
    fig_d.update_yaxes(title="Delta")
    st.plotly_chart(fig_d, use_container_width=True)

with tab2:
    ns_conv = [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
    mc_prices_conv, mc_errs_conv = [], []

    with st.spinner("Calculando convergencia MC..."):
        for n in ns_conv:
            res = monte_carlo_bsm(S, K, T, r, mc_sigma, q,
                                   option_type=mc_type,
                                   n_paths=n, seed=int(mc_seed),
                                   antithetic=use_anti,
                                   control_variate=use_cv)
            mc_prices_conv.append(res["price"])
            mc_errs_conv.append(res["std_error"])

    bsm_analytical = mc_result["bsm_price"]

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=ns_conv, y=mc_prices_conv,
        name="Precio MC", mode="lines+markers",
        line=dict(color=accent2, width=2),
        marker=dict(size=6),
    ))
    # CI band
    hi_band = [p + 1.96*e for p, e in zip(mc_prices_conv, mc_errs_conv)]
    lo_band = [p - 1.96*e for p, e in zip(mc_prices_conv, mc_errs_conv)]
    fig_conv.add_trace(go.Scatter(
        x=ns_conv + ns_conv[::-1],
        y=hi_band + lo_band[::-1],
        fill="toself",
        fillcolor=f"rgba(52,211,153,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 95%",
    ))
    fig_conv.add_hline(y=bsm_analytical, line_dash="dot", line_color=accent,
                        annotation_text=f"BSM = ${bsm_analytical:.4f}")
    _cvl = plotly_layout("Convergencia MC — precio vs numero de trayectorias")
    _cvl["height"] = 380
    fig_conv.update_layout(**_cvl)
    fig_conv.update_xaxes(title="N trayectorias", type="log")
    fig_conv.update_yaxes(title=f"Precio {mc_type.upper()} ($)")
    st.plotly_chart(fig_conv, use_container_width=True)

    # Error convergence
    fig_err = go.Figure()
    theo_err = [bsm_analytical * mc_sigma * np.sqrt(T) / np.sqrt(n) for n in ns_conv]
    fig_err.add_trace(go.Scatter(x=ns_conv, y=mc_errs_conv, name="Error std MC",
                                  line=dict(color=accent2, width=2), mode="lines+markers"))
    fig_err.add_trace(go.Scatter(x=ns_conv, y=theo_err, name="1/sqrt(N) teorico",
                                  line=dict(color=text_sub, width=1.5, dash="dash")))
    _el = plotly_layout("Error std vs N trayectorias")
    _el["height"] = 260
    fig_err.update_layout(**_el)
    fig_err.update_xaxes(title="N", type="log")
    fig_err.update_yaxes(title="Error std ($)", type="log")
    st.plotly_chart(fig_err, use_container_width=True)
    st.caption("El error converge como O(1/sqrt(N)). Duplicar la precision requiere 4x mas trayectorias.")

with tab3:
    st.markdown(f"<div style='font-size:0.82rem;color:{text_sub};margin-bottom:0.8rem'>Comparacion de precision entre las 4 combinaciones de tecnicas de reduccion de varianza con N fijo.</div>", unsafe_allow_html=True)

    n_fixed = 10_000
    configs = [
        ("Ingenuo",                        False, False),
        ("Antiteticas",                    True,  False),
        ("Control variante (BSM)",         False, True),
        ("Antiteticas + Control variante", True,  True),
    ]

    with st.spinner("Comparando tecnicas MC..."):
        results_mc = []
        for name_mc, anti, cv in configs:
            res = monte_carlo_bsm(S, K, T, r, mc_sigma, q,
                                   option_type=mc_type, n_paths=n_fixed,
                                   seed=int(mc_seed), antithetic=anti, control_variate=cv)
            results_mc.append((name_mc, res))

    bsm_ref2 = results_mc[0][1]["bsm_price"]
    cols_mc_tech = st.columns(4)
    for col_t, (name_mc, res) in zip(cols_mc_tech, results_mc):
        eff = results_mc[0][1]["std_error"] / res["std_error"] if res["std_error"] > 0 else float("inf")
        col_t.markdown(f"""
        <div style='background:{card};border:1px solid {border};border-radius:8px;padding:0.9rem 1rem'>
            <div style='font-size:0.65rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem'>{name_mc}</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:1.1rem;color:{text_main}'>${res["price"]:.4f}</div>
            <div style='font-size:0.78rem;color:{text_sub};margin-top:0.3rem'>
                SE: <span style='color:{text_main};font-family:"JetBrains Mono",monospace'>${res["std_error"]:.5f}</span>
            </div>
            <div style='font-size:0.78rem;color:{accent};margin-top:0.2rem'>
                Eficiencia: {eff:.1f}x
            </div>
        </div>""", unsafe_allow_html=True)

    # Bar chart of std errors
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Bar(
        x=[r[0] for r in results_mc],
        y=[r[1]["std_error"] for r in results_mc],
        marker_color=[text_sub, accent, "#a78bfa", "#34d399"],
        text=[f"${r[1]['std_error']:.5f}" for r in results_mc],
        textposition="outside",
    ))
    _efl = plotly_layout(f"Error std por tecnica (N={n_fixed:,} trayectorias)")
    _efl["height"] = 340
    _efl["showlegend"] = False
    fig_eff.update_layout(**_efl)
    fig_eff.update_xaxes(title="")
    fig_eff.update_yaxes(title="Error std ($)")
    st.plotly_chart(fig_eff, use_container_width=True)
    st.caption(f"Eficiencia = SE_ingenuo / SE_tecnica. La variable de control (BSM) es la mas efectiva para opciones europeas bajo GBM — la solucion analitica elimina casi todo el error de estimacion.")
