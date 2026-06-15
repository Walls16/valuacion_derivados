"""
app.py — Valuación Cuantitativa de Derivados
Landing page and navigation hub.
"""

import streamlit as st

st.set_page_config(
    page_title="Derivados · VQD",
    page_icon="∂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme persistence ──
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── CSS: dual theme system ──
def inject_css(dark: bool):
    if dark:
        bg        = "#0d0f14"
        surface   = "#131720"
        card      = "#1a1f2e"
        border    = "#2a3040"
        text_main = "#e8eaf0"
        text_sub  = "#8892a4"
        accent    = "#4f8ef7"
        accent2   = "#a78bfa"
        positive  = "#34d399"
        negative  = "#f87171"
    else:
        bg        = "#f4f6fb"
        surface   = "#ffffff"
        card      = "#ffffff"
        border    = "#dde3ef"
        text_main = "#1a2035"
        text_sub  = "#6b7a99"
        accent    = "#2563eb"
        accent2   = "#7c3aed"
        positive  = "#059669"
        negative  = "#dc2626"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

    html, body, [data-testid="stApp"] {{
        background-color: {bg} !important;
        color: {text_main} !important;
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {surface} !important;
        border-right: 1px solid {border} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {text_main} !important; }}

    .main .block-container {{ padding-top: 2rem; max-width: 1100px; }}

    /* ── Hero ── */
    .hero {{
        padding: 3.5rem 0 2.5rem 0;
        text-align: center;
    }}
    .hero-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: {accent};
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.4rem, 5vw, 3.6rem);
        font-weight: 700;
        color: {text_main};
        line-height: 1.15;
        margin: 0 0 0.6rem 0;
    }}
    .hero-title span {{
        color: {accent};
    }}
    .hero-sub {{
        font-size: 1.05rem;
        color: {text_sub};
        max-width: 560px;
        margin: 0 auto 2rem auto;
        line-height: 1.7;
    }}

    /* ── Model cards ── */
    .model-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }}
    .model-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.4rem 1.3rem;
        transition: border-color 0.2s, transform 0.15s;
        cursor: default;
    }}
    .model-card:hover {{
        border-color: {accent};
        transform: translateY(-2px);
    }}
    .model-card-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: {accent};
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .model-card-name {{
        font-size: 1rem;
        font-weight: 600;
        color: {text_main};
        margin-bottom: 0.3rem;
    }}
    .model-card-desc {{
        font-size: 0.82rem;
        color: {text_sub};
        line-height: 1.55;
    }}

    /* ── Stat pills ── */
    .stat-row {{
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1.5rem 0 2rem 0;
    }}
    .stat-pill {{
        background: {card};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 0.55rem 1.1rem;
        text-align: center;
    }}
    .stat-pill-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 500;
        color: {accent};
    }}
    .stat-pill-label {{
        font-size: 0.7rem;
        color: {text_sub};
        letter-spacing: 0.05em;
    }}

    /* ── Divider ── */
    .vqd-divider {{
        border: none;
        border-top: 1px solid {border};
        margin: 2rem 0;
    }}

    /* ── Section header ── */
    .section-header {{
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: {text_sub};
        margin-bottom: 1rem;
    }}

    /* ── Streamlit widget overrides ── */
    .stButton > button {{
        background: {accent} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.45rem 1.2rem !important;
    }}
    .stButton > button:hover {{
        opacity: 0.88 !important;
    }}
    label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stTextInput label {{
        color: {text_sub} !important;
        font-size: 0.82rem !important;
    }}
    .stSelectbox > div > div {{
        background-color: {card} !important;
        border-color: {border} !important;
        color: {text_main} !important;
    }}
    input[type="number"], input[type="text"] {{
        background-color: {card} !important;
        border-color: {border} !important;
        color: {text_main} !important;
    }}
    [data-testid="metric-container"] {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
    }}
    [data-testid="metric-container"] label {{
        color: {text_sub} !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {text_main} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {text_sub} !important;
        font-size: 0.85rem !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {accent} !important;
        border-bottom-color: {accent} !important;
    }}
    .stExpander {{
        border-color: {border} !important;
        background: {card} !important;
    }}
    .stExpander summary {{
        color: {text_main} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ∂ VQD")
    st.caption("Valuación Cuantitativa de Derivados")
    st.divider()

    dark_toggle = st.toggle("Modo oscuro", value=st.session_state.dark_mode, key="theme_toggle")
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.divider()
    st.markdown("**Navegación**")
    pages = {
        "Inicio":                  None,
        "BSM":                     "pages/1_BSM.py",
        "Árboles CRR":             "pages/2_CRR.py",
        "Heston":                   "pages/3_Heston.py",
        "Merton — Saltos":          "pages/4_Merton.py",
        "Complementos":             "pages/5_Complementos.py",
        "Comparativa":              "pages/6_Comparativa.py",
        "Subyacentes en Vivo":     "pages/7_LiveData.py",
    }
    for name in pages:
        st.markdown(f"<div style='padding:0.2rem 0; font-size:0.87rem'>{name}</div>", unsafe_allow_html=True)

    st.divider()
    st.caption("v1.0 · Owen · 2026")

# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Valuación Cuantitativa · Derivados Financieros</div>
    <h1 class="hero-title">
        Pricing de Derivados<br><span>del modelo al mercado</span>
    </h1>
    <p class="hero-sub">
        Plataforma analítica para la valuación de opciones europeas y americanas mediante 
        modelos clásicos y estocásticos avanzados, con datos de mercado en tiempo real.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stat pills ──
st.markdown("""
<div class="stat-row">
    <div class="stat-pill">
        <div class="stat-pill-num">4</div>
        <div class="stat-pill-label">Modelos de valuación</div>
    </div>
    <div class="stat-pill">
        <div class="stat-pill-num">11</div>
        <div class="stat-pill-label">Griegas calculadas</div>
    </div>
    <div class="stat-pill">
        <div class="stat-pill-num">∞</div>
        <div class="stat-pill-label">Subyacentes via yfinance</div>
    </div>
    <div class="stat-pill">
        <div class="stat-pill-num">2</div>
        <div class="stat-pill-label">Modos de visualización</div>
    </div>
</div>
<hr class="vqd-divider">
""", unsafe_allow_html=True)

# ── Model cards ──
st.markdown('<div class="section-header">Modelos disponibles</div>', unsafe_allow_html=True)
st.markdown("""
<div class="model-grid">
    <div class="model-card">
        <div class="model-card-tag">Analítico · Página 1</div>
        <div class="model-card-name">Black-Scholes-Merton</div>
        <div class="model-card-desc">Fórmula cerrada para opciones europeas. Base canónica para todos los modelos derivados.</div>
    </div>
    <div class="model-card">
        <div class="model-card-tag">Lattice · Página 2</div>
        <div class="model-card-name">Cox-Ross-Rubinstein</div>
        <div class="model-card-desc">Árbol binomial para opciones europeas y americanas. Visualización paso a paso del árbol.</div>
    </div>
    <div class="model-card">
        <div class="model-card-tag">Vol Estocástica · Página 3</div>
        <div class="model-card-name">Heston (1993)</div>
        <div class="model-card-desc">Varianza estocástica con reversión a la media. Genera la sonrisa de volatilidad de forma endógena.</div>
    </div>
    <div class="model-card">
        <div class="model-card-tag">Jump-Diffusion · Página 4</div>
        <div class="model-card-name">Merton (1976)</div>
        <div class="model-card-desc">Proceso difusivo con saltos de Poisson. Captura eventos de cola y skew de volatilidad.</div>
    </div>
    <div class="model-card">
        <div class="model-card-tag">Análisis · Página 5</div>
        <div class="model-card-name">Complementos</div>
        <div class="model-card-desc">Griegas de 1.° y 2.° orden, sonrisa de volatilidad, superficie IV, paridad put-call.</div>
    </div>
    <div class="model-card">
        <div class="model-card-tag">Benchmark · Página 6</div>
        <div class="model-card-name">Comparativa</div>
        <div class="model-card-desc">Divergencia de precios entre modelos. Análisis de sensibilidad paramétrica cruzada.</div>
    </div>
    <div class="model-card">
        <div class="model-card-tag">Live · Página 7</div>
        <div class="model-card-name">Subyacentes en Vivo</div>
        <div class="model-card-desc">Valuación sobre datos reales vía yfinance. IV implícita, cadena de opciones, ATM pricing.</div>
    </div>
</div>
<hr class="vqd-divider">
""", unsafe_allow_html=True)

# ── Quick start ──
st.markdown('<div class="section-header">Inicio rápido</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.info("**Nuevo en derivados?** → Empieza con la página **BSM** para familiarizarte con los parámetros base: S, K, T, r, σ.")

with col2:
    st.info("**Quieres valuar un ticker real?** → Ve directo a **Subyacentes en Vivo** (Pág. 7) e ingresa cualquier símbolo de yfinance.")

st.markdown("")
st.markdown(
    "<div style='text-align:center; font-size:0.75rem; color:#6b7a99; padding-top:1rem'>"
    "Valuación Cuantitativa de Derivados · UDLAP Actuaría · 2026"
    "</div>",
    unsafe_allow_html=True
)
