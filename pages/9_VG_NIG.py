"""
Page 9 — Variance Gamma & NIG
Levy process models: infinite activity, pure-jump representations of asset returns.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import VarianceGammaEngine, NIGEngine, BSMEngine

st.set_page_config(page_title="VG & NIG · VQD", page_icon="d", layout="wide")

dark      = st.session_state.get("dark_mode", True)
bg        = "#0d0f14" if dark else "#f4f6fb"
surface   = "#131720" if dark else "#ffffff"
card      = "#1a1f2e" if dark else "#ffffff"
border    = "#2a3040" if dark else "#dde3ef"
text_main = "#e8eaf0" if dark else "#1a2035"
text_sub  = "#8892a4" if dark else "#6b7a99"
accent    = "#4f8ef7" if dark else "#2563eb"
accent2   = "#a78bfa"
grid_col  = "#1e2535" if dark else "#e8edf5"
plot_bg   = "#131720" if dark else "#ffffff"
paper_bg  = "#0d0f14" if dark else "#f4f6fb"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
html, body, [data-testid="stApp"] {{ background-color:{bg}!important; color:{text_main}!important; font-family:'Inter',sans-serif; }}
[data-testid="stSidebar"] {{ background-color:{surface}!important; border-right:1px solid {border}!important; }}
[data-testid="stSidebar"] * {{ color:{text_main}!important; }}
.main .block-container {{ padding-top:1.5rem; max-width:1150px; }}
.page-title {{ font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:{text_main}; }}
.page-eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.16em; color:{accent}; text-transform:uppercase; margin-bottom:0.5rem; }}
.page-sub {{ font-size:0.9rem; color:{text_sub}; margin-bottom:1.5rem; }}
.formula-box {{ background:{card}; border:1px solid {border}; border-left:3px solid {accent}; border-radius:8px; padding:1.2rem 1.4rem; margin:0.8rem 0; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:{text_sub}; line-height:1.9; }}
.formula-box-nig {{ background:{card}; border:1px solid {border}; border-left:3px solid {accent2}; border-radius:8px; padding:1.2rem 1.4rem; margin:0.8rem 0; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:{text_sub}; line-height:1.9; }}
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
st.markdown('<div class="page-eyebrow">Procesos de Levy · Pagina 9</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Variance Gamma & NIG</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Modelos de actividad infinita basados en procesos de Levy. Capturan colas pesadas, asimetria y kurtosis sin saltos discretos de Poisson.</div>', unsafe_allow_html=True)

# Theory block
st.markdown(f"""
<div style="background:{card};border:1px solid {border};border-radius:10px;padding:1.4rem 1.6rem;margin:0 0 1.2rem 0;line-height:1.85">
    <div style="font-size:0.7rem;font-family:'JetBrains Mono',monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.9rem">
        Procesos de Levy — mas alla de los saltos de Poisson
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        El modelo de Merton usa saltos discretos con frecuencia finita (proceso de Poisson compuesto).
        Los procesos de Levy de <strong style="color:{text_main}">actividad infinita</strong> como VG y NIG van mas lejos:
        modelan el precio como la superposicion de <em>infinitos</em> micro-saltos, con frecuencia que crece sin limite
        a medida que el tamano del salto tiende a cero. El resultado es un proceso que es
        continuo en distribucion pero discontinuo en trayectoria — mas realista que GBM pero sin
        los saltos macroscopicos discretos de Merton.
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem">
        <div style="border-left:3px solid {accent};padding-left:0.9rem">
            <div style="font-size:0.72rem;color:{accent};font-family:'JetBrains Mono',monospace;text-transform:uppercase;margin-bottom:0.4rem">Variance Gamma (VG)</div>
            <p style="font-size:0.83rem;color:{text_sub};margin:0">
                Madan, Carr & Chang (1998). Un Browniano con drift <em>evaluado en tiempo aleatorio Gamma</em>.
                El tiempo gamma introduce heteroscedasticidad: en periodos de alta actividad el tiempo
                avanza rapido, generando mas varianza. sigma controla la difusion,
                theta el sesgo (drift del BM), nu la kurtosis (varianza del tiempo gamma).
            </p>
        </div>
        <div style="border-left:3px solid {accent2};padding-left:0.9rem">
            <div style="font-size:0.72rem;color:{accent2};font-family:'JetBrains Mono',monospace;text-transform:uppercase;margin-bottom:0.4rem">Normal Inverse Gaussian (NIG)</div>
            <p style="font-size:0.83rem;color:{text_sub};margin:0">
                Barndorff-Nielsen (1997). Un Browniano evaluado en tiempo Inverso Gaussiano.
                La distribucion NIG es cerrada bajo convolucion y tiene funciones de densidad
                semi-analiticas. alpha controla las colas (mayor = mas ligeras), beta
                la asimetria, delta la escala. Muy usada en modelado de riesgo y credito.
            </p>
        </div>
    </div>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <p style="color:{text_sub};font-size:0.83rem;margin:0">
        Ambos modelos se pricean via <strong style="color:{text_main}">FFT de Carr-Madan (1999)</strong>:
        la funcion caracteristica del log-precio es conocida analiticamente, y el precio de la opcion
        se obtiene via transformada de Fourier rapida. Esto permite calcular toda la cadena de strikes
        en un solo pase de FFT — computacionalmente muy eficiente para calibracion.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Formula boxes ──
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown(f"""
    <div class="formula-box">
        <strong style="color:{text_main}">Variance Gamma</strong><br><br>
        X(t) = &theta;&middot;G(t) + &sigma;&middot;W(G(t))<br>
        G(t) ~ Gamma(t/&nu;, &nu;) [tiempo aleatorio]<br><br>
        Funcion caracteristica:<br>
        &phi;(u) = (1 - iu&theta;&nu; + &sigma;&sup2;&nu;u&sup2;/2)<sup>-T/&nu;</sup><br><br>
        <span style="color:{text_sub}">&sigma; = vol difusiva &middot; &theta; = sesgo &middot; &nu; = kurtosis</span>
    </div>
    """, unsafe_allow_html=True)
with col_f2:
    st.markdown(f"""
    <div class="formula-box-nig">
        <strong style="color:{text_main}">Normal Inverse Gaussian</strong><br><br>
        X(t) = &beta;&middot;Z(t) + W(Z(t))<br>
        Z(t) ~ InvGaussian(&delta;t, &radic;(&alpha;&sup2;-&beta;&sup2;))<br><br>
        Funcion caracteristica:<br>
        &phi;(u) = exp[-&delta;t(&radic;(&alpha;&sup2;-(β+iu)&sup2;) - &radic;(&alpha;&sup2;-&beta;&sup2;))]<br><br>
        <span style="color:{text_sub}">&alpha; = colas &middot; &beta; = sesgo &middot; &delta; = escala</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Shared params + model params ──
st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.6rem'>Parametros comunes</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    S = st.number_input("Spot S", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K = st.number_input("Strike K", value=100.0, min_value=0.01, step=1.0, format="%.2f")
with c2:
    T = st.number_input("Tiempo T (anos)", value=1.0, min_value=0.01, step=0.05, format="%.3f")
    r = st.number_input("Tasa r (%)", value=5.0, step=0.25, format="%.2f") / 100
with c3:
    q     = st.number_input("Dividendo q (%)", value=0.0, step=0.1, format="%.2f") / 100
    sigma_bsm = st.number_input("sigma BSM ref (%)", value=20.0, step=0.5, format="%.2f") / 100

col_vg, col_nig = st.columns(2)

with col_vg:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin:0.8rem 0 0.4rem'>Parametros VG</div>", unsafe_allow_html=True)

    VG_PRESETS = {
        "Personalizado":       None,
        "Equity (calibrado)":  dict(sigma=0.12, theta=-0.14, nu=0.20),
        "FX (simetrico)":      dict(sigma=0.10, theta= 0.00, nu=0.15),
        "Colas pesadas":       dict(sigma=0.10, theta=-0.10, nu=0.50),
        "Alta asimetria":      dict(sigma=0.15, theta=-0.25, nu=0.10),
    }
    preset_vg  = st.selectbox("Perfil VG", list(VG_PRESETS.keys()), key="pvg")
    _pvg = VG_PRESETS[preset_vg] or {}
    vg_sigma = st.slider("sigma VG (vol difusiva)", 0.01, 1.0, float(_pvg.get("sigma", 0.12)), 0.01, key="vgs")
    vg_theta = st.slider("theta VG (sesgo)", -1.0, 1.0, float(_pvg.get("theta", -0.14)), 0.01, key="vgt")
    vg_nu    = st.slider("nu VG (kurtosis)", 0.01, 2.0, float(_pvg.get("nu",   0.20)), 0.01, key="vgn")

with col_nig:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{accent2};margin:0.8rem 0 0.4rem'>Parametros NIG</div>", unsafe_allow_html=True)

    NIG_PRESETS = {
        "Personalizado":        None,
        "Equity (calibrado)":   dict(alpha=15.0, beta=-5.0, delta=1.5),
        "Colas muy pesadas":    dict(alpha=5.0,  beta=-2.0, delta=1.0),
        "Casi simetrico":       dict(alpha=20.0, beta= 0.0, delta=2.0),
        "Alta asimetria":       dict(alpha=10.0, beta=-8.0, delta=1.2),
    }
    preset_nig = st.selectbox("Perfil NIG", list(NIG_PRESETS.keys()), key="pnig")
    _pnig = NIG_PRESETS[preset_nig] or {}
    nig_alpha = st.slider("alpha NIG (colas)", 1.0, 50.0, float(_pnig.get("alpha", 15.0)), 0.5, key="nia")
    nig_beta  = st.slider("beta NIG (sesgo)",  float(-nig_alpha+0.1), float(nig_alpha-0.1),
                           float(np.clip(_pnig.get("beta", -5.0), -nig_alpha+0.01, nig_alpha-0.01)), 0.1, key="nib")
    nig_delta = st.slider("delta NIG (escala)", 0.1, 5.0, float(_pnig.get("delta", 1.5)), 0.1, key="nid")

# ── Compute ──
with st.spinner("Calculando VG y NIG via FFT..."):
    vg  = VarianceGammaEngine(S, K, T, r, vg_sigma, vg_theta, vg_nu, q)
    nig = NIGEngine(S, K, T, r, nig_alpha, nig_beta, nig_delta, q)
    bsm = BSMEngine(S, K, T, r, sigma_bsm, q)

    vg_call  = vg.call_price();  vg_put  = vg.put_price()
    nig_call = nig.call_price(); nig_put = nig.put_price()
    bsm_call = bsm.call_price(); bsm_put = bsm.put_price()

st.divider()
st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.6rem'>Resultados</div>", unsafe_allow_html=True)

rc = st.columns(5)
for col_r, (label, c_val, p_val, ref_c, color) in zip(rc, [
    ("BSM ref", bsm_call, bsm_put, bsm_call, text_sub),
    ("VG Call", vg_call, None, bsm_call, accent),
    ("VG Put", None, vg_put, bsm_put, accent),
    ("NIG Call", nig_call, None, bsm_call, accent2),
    ("NIG Put", None, nig_put, bsm_put, accent2),
]):
    price = c_val if c_val is not None else p_val
    diff  = price - ref_c
    col_r.markdown(f"""
    <div class="result-card" style="border-top:3px solid {color}">
        <div class="result-label" style="color:{color}">{label}</div>
        <div class="result-value">${price:.4f}</div>
        <div style='font-size:0.72rem;color:{"#34d399" if diff>=0 else "#f87171"}'>
            {'BSM' if label=='BSM ref' else f'vs BSM: {diff:+.4f}'}
        </div>
    </div>""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Sonrisas VG vs NIG", "Distribucion de retornos", "Sensibilidad parametrica"])

with tab1:
    strikes_sm = np.linspace(max(S*0.70, 1), S*1.30, 60)
    with st.spinner("Calculando sonrisas..."):
        ivs_vg  = vg.vol_smile(strikes_sm, "call")
        ivs_nig = nig.vol_smile(strikes_sm, "call")
    ivs_bsm_flat = np.full(len(strikes_sm), sigma_bsm * 100)

    fig_sm = go.Figure()
    fig_sm.add_trace(go.Scatter(x=strikes_sm, y=ivs_vg, name="VG IV",
                                 line=dict(color=accent, width=2.5)))
    fig_sm.add_trace(go.Scatter(x=strikes_sm, y=ivs_nig, name="NIG IV",
                                 line=dict(color=accent2, width=2.5)))
    fig_sm.add_trace(go.Scatter(x=strikes_sm, y=ivs_bsm_flat, name="BSM flat",
                                 line=dict(color="#f87171", width=1.5, dash="dash")))
    fig_sm.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
    fig_sm.add_vline(x=S, line_dash="dot", line_color="#34d399", annotation_text="S")
    fig_sm.update_layout(**plotly_layout("Sonrisas de Volatilidad Implicita — VG vs NIG vs BSM"), height=400)
    fig_sm.update_xaxes(title="Strike K")
    fig_sm.update_yaxes(title="IV (%)")
    st.plotly_chart(fig_sm, use_container_width=True)

with tab2:
    st.markdown(f"<div style='font-size:0.82rem;color:{text_sub};margin-bottom:0.8rem'>Distribucion simulada de log-retornos bajo cada modelo vs Normal (GBM).</div>", unsafe_allow_html=True)

    n_sim = 50_000
    np.random.seed(42)

    # VG simulation: Gamma time-change
    gamma_times = np.random.gamma(T/vg_nu, vg_nu, n_sim)
    vg_returns  = vg_theta*gamma_times + vg_sigma*np.sqrt(gamma_times)*np.random.randn(n_sim)

    # NIG simulation: via inverse Gaussian time-change
    ig_mean  = nig_delta * T / np.sqrt(nig_alpha**2 - nig_beta**2)
    ig_shape = (nig_delta * T)**2
    # Inverse Gaussian sampling
    v_ig    = np.random.randn(n_sim)**2
    x_ig    = ig_mean + ig_mean**2*v_ig/(2*ig_shape) - ig_mean/(2*ig_shape)*np.sqrt(4*ig_mean*ig_shape*v_ig + ig_mean**2*v_ig**2)
    u_ig    = np.random.rand(n_sim)
    ig_times = np.where(u_ig <= ig_mean/(ig_mean + x_ig), x_ig, ig_mean**2/x_ig)
    nig_returns = nig_beta*ig_times + np.sqrt(ig_times)*np.random.randn(n_sim)

    # GBM
    gbm_returns = np.random.normal(0, sigma_bsm*np.sqrt(T), n_sim)

    # Compute moments
    def moments(x):
        return {"mean": np.mean(x), "std": np.std(x),
                "skew": float(np.mean((x-np.mean(x))**3)/np.std(x)**3),
                "kurt": float(np.mean((x-np.mean(x))**4)/np.std(x)**4 - 3)}

    m_vg  = moments(vg_returns)
    m_nig = moments(nig_returns)
    m_gbm = moments(gbm_returns)

    col_mom = st.columns(3)
    for col_m, (name, m, color) in zip(col_mom, [
        ("GBM (BSM)", m_gbm, text_sub),
        ("VG", m_vg, accent),
        ("NIG", m_nig, accent2),
    ]):
        col_m.markdown(f"""
        <div style='background:{card};border:1px solid {border};border-top:3px solid {color};border-radius:8px;padding:0.8rem 1rem'>
            <div style='font-size:0.68rem;color:{color};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>{name}</div>
            <div style='font-size:0.8rem;color:{text_sub}'>Skew: <span style='color:{text_main};font-family:"JetBrains Mono",monospace'>{m["skew"]:+.3f}</span></div>
            <div style='font-size:0.8rem;color:{text_sub}'>Kurt exc: <span style='color:{text_main};font-family:"JetBrains Mono",monospace'>{m["kurt"]:+.3f}</span></div>
            <div style='font-size:0.8rem;color:{text_sub}'>Std: <span style='color:{text_main};font-family:"JetBrains Mono",monospace'>{m["std"]:.4f}</span></div>
        </div>""", unsafe_allow_html=True)

    fig_dist = go.Figure()
    for returns, name, color in [
        (gbm_returns, "GBM (Normal)", text_sub),
        (vg_returns,  "VG",           accent),
        (nig_returns, "NIG",          accent2),
    ]:
        fig_dist.add_trace(go.Histogram(
            x=returns, nbinsx=120, name=name,
            marker_color=color, opacity=0.55,
            histnorm="probability density",
        ))
    _dl = plotly_layout("Distribucion de log-retornos: VG vs NIG vs GBM")
    _dl["barmode"] = "overlay"
    _dl["height"]  = 380
    fig_dist.update_layout(**_dl)
    fig_dist.update_xaxes(title="Log-retorno", range=[-0.8, 0.8])
    fig_dist.update_yaxes(title="Densidad")
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Colas mas pesadas y mayor asimetria que la Normal = mayor precio para opciones OTM.")

with tab3:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;margin-bottom:0.4rem'>VG: precio call vs nu (kurtosis)</div>", unsafe_allow_html=True)
        nus_range = np.linspace(0.01, 1.5, 40)
        vg_calls_nu = []
        for nu_v in nus_range:
            try:
                vg_t = VarianceGammaEngine(S, K, T, r, vg_sigma, vg_theta, nu_v, q)
                vg_calls_nu.append(vg_t.call_price())
            except:
                vg_calls_nu.append(np.nan)
        fig_vgnu = go.Figure()
        fig_vgnu.add_trace(go.Scatter(x=nus_range, y=vg_calls_nu, line=dict(color=accent, width=2)))
        fig_vgnu.add_hline(y=bsm_call, line_dash="dot", line_color="#f87171", annotation_text="BSM")
        fig_vgnu.add_vline(x=vg_nu, line_dash="dot", line_color=text_sub, annotation_text="nu actual")
        _vgnul = plotly_layout("VG Call vs nu")
        _vgnul["height"] = 300
        fig_vgnu.update_layout(**_vgnul)
        fig_vgnu.update_xaxes(title="nu")
        fig_vgnu.update_yaxes(title="Call ($)")
        st.plotly_chart(fig_vgnu, use_container_width=True)

    with col_s2:
        st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;margin-bottom:0.4rem'>NIG: precio call vs beta (sesgo)</div>", unsafe_allow_html=True)
        betas_range = np.linspace(-nig_alpha*0.9, nig_alpha*0.9, 40)
        nig_calls_b = []
        for b_v in betas_range:
            try:
                nig_t = NIGEngine(S, K, T, r, nig_alpha, b_v, nig_delta, q)
                nig_calls_b.append(nig_t.call_price())
            except:
                nig_calls_b.append(np.nan)
        fig_nigb = go.Figure()
        fig_nigb.add_trace(go.Scatter(x=betas_range, y=nig_calls_b, line=dict(color=accent2, width=2)))
        fig_nigb.add_hline(y=bsm_call, line_dash="dot", line_color="#f87171", annotation_text="BSM")
        fig_nigb.add_vline(x=nig_beta, line_dash="dot", line_color=text_sub, annotation_text="beta actual")
        _nigbl = plotly_layout("NIG Call vs beta")
        _nigbl["height"] = 300
        fig_nigb.update_layout(**_nigbl)
        fig_nigb.update_xaxes(title="beta")
        fig_nigb.update_yaxes(title="Call ($)")
        st.plotly_chart(fig_nigb, use_container_width=True)
