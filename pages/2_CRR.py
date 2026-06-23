"""
Page 2 — Cox-Ross-Rubinstein Binomial Tree
European and American options via lattice method.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import CRREngine, BSMEngine

st.set_page_config(page_title="CRR · VQD", page_icon="∂", layout="wide")

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
.main .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}
.page-title {{ font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: {text_main}; margin-bottom: 0.2rem; }}
.page-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.16em; color: {accent}; text-transform: uppercase; margin-bottom: 0.5rem; }}
.page-sub {{ font-size: 0.9rem; color: {text_sub}; margin-bottom: 1.5rem; }}
.formula-box {{ background: {card}; border: 1px solid {border}; border-left: 3px solid {accent}; border-radius: 8px; padding: 1.2rem 1.4rem; margin: 1rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: {text_sub}; line-height: 1.9; }}
.result-card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 1.3rem 1.5rem; }}
.result-label {{ font-size: 0.72rem; color: {text_sub}; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; }}
.result-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600; color: {text_main}; }}
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

st.markdown('<div class="page-eyebrow">Árbol Lattice · Página 2</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Cox-Ross-Rubinstein</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Valuación discreta mediante árbol binomial recombinante. Soporta opciones americanas con ejercicio anticipado.</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="formula-box">
    <strong style="color:{text_main}">Parámetros del árbol CRR</strong><br><br>
    u = e<sup>σ√Δt</sup> &nbsp;·&nbsp; d = 1/u &nbsp;·&nbsp; Δt = T/N<br><br>
    p = (e<sup>(r−q)Δt</sup> &minus; d) / (u &minus; d) &nbsp;&nbsp;&nbsp; [probabilidad risk-neutral de subida]<br><br>
    V<sub>i,j</sub> = e<sup>&minus;rΔt</sup> · [p·V<sub>i+1,j</sub> + (1&minus;p)·V<sub>i+1,j+1</sub>] &nbsp;&nbsp;&nbsp; [inducción hacia atrás]<br><br>
    <span style="color:{text_sub}">Americana: V<sub>i,j</sub> = max(V<sub>europeo</sub>, payoff intrínseco en nodo (i,j))</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{card};border:1px solid {border};border-radius:10px;padding:1.4rem 1.6rem;margin:0.8rem 0 1.2rem 0;line-height:1.85">
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.9rem">
        Origen e intuición del árbol binomial
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        El árbol binomial de Cox, Ross y Rubinstein (1979) es una <strong style="color:{text_main}">discretización del proceso
        continuo de BSM</strong>. La idea es dividir el tiempo al vencimiento T en N pasos iguales de duración Δt = T/N.
        En cada paso, el precio del subyacente solo puede moverse a dos estados: sube por un factor u o baja por un factor d.
        Al hacer N → ∞ con los factores correctos, el árbol converge exactamente a la fórmula de BSM.
    </p>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        El factor u = e<sup>σ√Δt</sup> no es arbitrario: está calibrado para que la varianza del log-retorno en cada paso
        sea σ²Δt, replicando exactamente la volatilidad del GBM subyacente.
        La probabilidad p no es la probabilidad real de que el precio suba —
        es la <strong style="color:{text_main}">probabilidad risk-neutral</strong>, derivada de imponer la condición de
        no-arbitraje: el precio esperado descontado debe ser igual al precio actual.
    </p>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        El árbol se <strong style="color:{text_main}">resuelve hacia atrás</strong> (backward induction): se calculan
        los payoffs en los nodos terminales (t=T) y se descuentan paso a paso hasta el nodo raíz (t=0),
        que es el precio de la opción hoy. La estructura recombinante (u·d = 1) garantiza que el árbol
        tenga N+1 nodos terminales en lugar de 2<sup>N</sup>, haciendo el cómputo polinomial.
    </p>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.6rem">
        Opciones americanas — ejercicio anticipado
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        La gran ventaja del árbol sobre BSM es que permite valuar opciones con <strong style="color:{text_main}">ejercicio anticipado</strong>.
        En cada nodo interno, el tenedor compara el valor de continuar (valor de continuación) contra ejercer inmediatamente (valor intrínseco).
        El árbol resuelve este problema de control óptimo de forma natural: en cada nodo simplemente se toma el máximo.
        BSM no puede hacer esto — no existe fórmula cerrada para una put americana (excepto en casos límite).
    </p>
    <p style="color:{text_sub};font-size:0.83rem;margin:0 0 0.4rem 0">
        Nota: para una <em>call americana sobre un activo sin dividendos</em>, nunca es óptimo ejercer anticipadamente —
        el valor de continuar siempre domina. Prueba: C<sub>americana</sub> = C<sub>europea</sub> en ese caso.
        Para una <em>put americana</em>, sí puede ser óptimo ejercer anticipadamente (especialmente deep ITM),
        lo que se refleja en que P<sub>americana</sub> &gt; P<sub>europea</sub>.
    </p>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.6rem">
        Convergencia y precisión
    </div>
    <p style="color:{text_sub};font-size:0.83rem;margin:0">
        Con N pequeño el árbol es impreciso — la opción solo puede tener un conjunto discreto de payoffs terminales.
        Con N grande (200-1000) la convergencia a BSM es prácticamente exacta para opciones europeas.
        Curiosamente, el error oscila alrededor del valor BSM alternando según si N es par o impar
        (efecto de paridad del árbol). Puedes verlo en la gráfica de convergencia: la serie zigzaguea antes de estabilizarse.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

col_in, col_out = st.columns([1, 1.6], gap="large")

with col_in:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Parámetros</div>", unsafe_allow_html=True)
    S     = st.number_input("Precio subyacente (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K     = st.number_input("Precio de ejercicio (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T     = st.number_input("Tiempo al vencimiento (T, años)", value=1.0, min_value=0.01, max_value=10.0, step=0.05, format="%.3f")
    r     = st.number_input("Tasa libre de riesgo (r, %)", value=5.0, min_value=0.0, max_value=30.0, step=0.25, format="%.2f") / 100
    sigma = st.number_input("Volatilidad (σ, %)", value=20.0, min_value=0.1, max_value=200.0, step=0.5, format="%.2f") / 100
    q     = st.number_input("Dividendo continuo (q, %)", value=0.0, min_value=0.0, max_value=20.0, step=0.1, format="%.2f") / 100
    N     = st.slider("Pasos del árbol (N)", min_value=10, max_value=1000, value=200, step=10)
    american = st.checkbox("Opción Americana (ejercicio anticipado)", value=False)

    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp((r - q) * dt) - d) / (u - d)

    st.markdown(f"""
    <div style='background:{card};border:1px solid {border};border-radius:6px;padding:0.8rem 1rem;margin-top:0.6rem;font-family:JetBrains Mono,monospace;font-size:0.78rem;color:{text_sub}'>
    Δt = {dt:.5f}&nbsp;&nbsp; u = {u:.5f}<br>
    d  = {d:.5f}&nbsp;&nbsp; p = {p:.5f}
    </div>
    """, unsafe_allow_html=True)

crr = CRREngine(S, K, T, r, sigma, q, N, american)
call_crr = crr.call_price()
put_crr  = crr.put_price()

bsm_call = BSMEngine(S, K, T, r, sigma, q).call_price()
bsm_put  = BSMEngine(S, K, T, r, sigma, q).put_price()

with col_out:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Resultados</div>", unsafe_allow_html=True)

    rc, rp = st.columns(2)
    with rc:
        diff_c = call_crr - bsm_call
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Call {'Americana' if american else 'Europea'}</div>
            <div class="result-value">${call_crr:.4f}</div>
            <div style='font-size:0.78rem;color:{"#34d399" if diff_c>=0 else "#f87171"};margin-top:0.3rem'>
                vs BSM: {"+" if diff_c>=0 else ""}{diff_c:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with rp:
        diff_p = put_crr - bsm_put
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Put {'Americana' if american else 'Europea'}</div>
            <div class="result-value">${put_crr:.4f}</div>
            <div style='font-size:0.78rem;color:{"#34d399" if diff_p>=0 else "#f87171"};margin-top:0.3rem'>
                vs BSM: {"+" if diff_p>=0 else ""}{diff_p:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Δ Call (CRR)", f"{crr.delta('call'):.4f}")
    m2.metric("BSM Call", f"${bsm_call:.4f}")
    m3.metric("Error relativo", f"{abs(call_crr-bsm_call)/bsm_call*100:.4f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["Árbol (visualización)", "Convergencia vs N", "Comparativa con BSM"])

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

with tab1:
    col_t1a, col_t1b, col_t1c = st.columns(3)
    with col_t1a:
        display_steps = st.slider("Pasos a visualizar", min_value=2, max_value=10, value=5)
    with col_t1b:
        opt_type_tree = st.radio("Tipo de opción", ["call", "put"], horizontal=True)
    with col_t1c:
        tree_show = st.radio("Mostrar en nodos", ["Valor opción", "Precio S", "Ambos"], horizontal=False)

    crr_vis = CRREngine(S, K, T, r, sigma, q, display_steps, american)
    S_tree, V_tree = crr_vis.full_tree(opt_type_tree, max_display=display_steps)

    # Build Plotly tree graph
    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x, edge_y = [], []

    for step in range(len(S_tree)):
        for j in range(step + 1):
            x = step
            y = step - 2 * j
            s_val = S_tree[step][j]
            v_val = V_tree[step][j] if V_tree[step] is not None else 0
            node_x.append(x)
            node_y.append(y)
            if tree_show == "Valor opción":
                node_text.append(f"V={v_val:.3f}")
            elif tree_show == "Precio S":
                node_text.append(f"S={s_val:.2f}")
            else:
                node_text.append(f"S={s_val:.2f}<br>V={v_val:.3f}")
            node_color.append(v_val)

            if step < len(S_tree) - 1:
                x_next_u = step + 1
                y_next_u = (step + 1) - 2 * j
                x_next_d = step + 1
                y_next_d = (step + 1) - 2 * (j + 1)
                edge_x += [x, x_next_u, None]
                edge_y += [y, y_next_u, None]
                edge_x += [x, x_next_d, None]
                edge_y += [y, y_next_d, None]

    fig_tree = go.Figure()
    fig_tree.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                   line=dict(color=border, width=1), hoverinfo="none", showlegend=False))
    fig_tree.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=32, color=node_color, colorscale="Blues" if not dark else "Blues",
                    showscale=True, colorbar=dict(title="Valor")),
        text=[t.split("<br>")[-1] for t in node_text],
        textposition="middle center",
        textfont=dict(size=9, color=text_main),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))
    _tree_layout = plotly_layout(f"Árbol CRR — {display_steps} pasos · {opt_type_tree.upper()}")
    _tree_layout["height"] = 420
    _tree_layout["showlegend"] = False
    _tree_layout["yaxis"] = dict(showticklabels=False, showgrid=False)
    _tree_layout["xaxis"] = dict(showgrid=False, title="Paso")
    fig_tree.update_layout(**_tree_layout)
    st.plotly_chart(fig_tree, use_container_width=True)

with tab2:
    Ns = list(range(5, 501, 10))
    conv_calls = [CRREngine(S, K, T, r, sigma, q, n, american).call_price() for n in Ns]
    conv_puts  = [CRREngine(S, K, T, r, sigma, q, n, american).put_price()  for n in Ns]

    show_error = st.checkbox("Mostrar error absoluto vs BSM", value=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=Ns, y=conv_calls, name="CRR Call", line=dict(color=accent, width=2)))
    fig2.add_trace(go.Scatter(x=Ns, y=conv_puts,  name="CRR Put",  line=dict(color="#a78bfa", width=2)))
    fig2.add_hline(y=bsm_call, line_dash="dot", line_color="#34d399", annotation_text="BSM Call", annotation_font_color="#34d399")
    fig2.add_hline(y=bsm_put,  line_dash="dot", line_color="#f59e0b", annotation_text="BSM Put",  annotation_font_color="#f59e0b")
    fig2.update_layout(**plotly_layout("Convergencia del árbol CRR → BSM"), height=340)
    fig2.update_xaxes(title="Número de pasos N")
    fig2.update_yaxes(title="Precio")
    st.plotly_chart(fig2, use_container_width=True)

    if show_error:
        err_calls = [abs(c - bsm_call) for c in conv_calls]
        err_puts  = [abs(p - bsm_put)  for p in conv_puts]
        fig2e = go.Figure()
        fig2e.add_trace(go.Scatter(x=Ns, y=err_calls, name="|CRR Call - BSM|", line=dict(color=accent, width=1.5)))
        fig2e.add_trace(go.Scatter(x=Ns, y=err_puts,  name="|CRR Put - BSM|",  line=dict(color="#a78bfa", width=1.5)))
        fig2e.update_layout(**plotly_layout("Error absoluto de convergencia"), height=260)
        fig2e.update_xaxes(title="N")
        fig2e.update_yaxes(title="Error ($)", type="log")
        st.plotly_chart(fig2e, use_container_width=True)
        st.caption(f"Oscilación par/impar visible en N pequeño. A N={N}: error call = ${abs(call_crr - bsm_call):.4f}")

with tab3:
    # CRR vs BSM across strikes
    strikes = np.linspace(max(S * 0.6, 1), S * 1.4, 60)
    crr_c = [CRREngine(S, k, T, r, sigma, q, N, american).call_price() for k in strikes]
    bsm_c = [BSMEngine(S, k, T, r, sigma, q).call_price() for k in strikes]
    diff_c_arr = [c - b for c, b in zip(crr_c, bsm_c)]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=strikes, y=crr_c, name="CRR Call", line=dict(color=accent, width=2)))
    fig3.add_trace(go.Scatter(x=strikes, y=bsm_c, name="BSM Call", line=dict(color="#34d399", width=2, dash="dash")))
    fig3.add_trace(go.Bar(x=strikes, y=diff_c_arr, name="Diferencia",
                           marker_color=[("#34d399" if d >= 0 else "#f87171") for d in diff_c_arr],
                           opacity=0.5, yaxis="y2"))
    fig3.update_layout(
        **plotly_layout("CRR vs BSM — Call por strike"),
        height=380,
        yaxis2=dict(overlaying="y", side="right", title="Diferencia CRR−BSM", showgrid=False),
    )
    fig3.update_xaxes(title="Strike K")
    fig3.update_yaxes(title="Precio call")
    st.plotly_chart(fig3, use_container_width=True)
