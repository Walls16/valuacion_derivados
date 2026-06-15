# ∂ Valuación Cuantitativa de Derivados

> Plataforma analítica para la valuación de opciones europeas y americanas mediante modelos clásicos y estocásticos avanzados, con datos de mercado en tiempo real.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Descripción

**VQD** es una aplicación multi-página construida en Streamlit que implementa cuatro modelos de valuación de derivados financieros desde cero, sin depender de librerías de pricing de terceros. Cada modelo expone sus parámetros de forma interactiva, genera visualizaciones analíticas y puede compararse directamente contra los demás bajo los mismos supuestos de mercado.

La página de datos en vivo permite valuar opciones reales sobre cualquier subyacente disponible en Yahoo Finance, extrayendo precio spot, cadena de opciones y volatilidad implícita de mercado para comparar contra los modelos implementados.

---

## Modelos implementados

| Página | Modelo | Tipo | Opciones |
|--------|--------|------|----------|
| 1 | **Black-Scholes-Merton** | Analítico cerrado | Europeas |
| 2 | **Cox-Ross-Rubinstein** | Árbol binomial (lattice) | Europeas + Americanas |
| 3 | **Heston (1993)** | Vol estocástica, FCaract. | Europeas |
| 4 | **Merton Jump-Diffusion (1976)** | Difusión + saltos Poisson | Europeas |
| 5 | Complementos | Griegas, Vol Smile, IV | — |
| 6 | Comparativa | Benchmark cruzado | — |
| 7 | Subyacentes en Vivo | yfinance real-time | Cualquier ticker |

---

## Estructura del proyecto

```
valuacion_cuantitativa_derivados/
│
├── app.py                  # Landing page y navegación
├── engine.py               # Motor de cálculo — todos los modelos
├── requirements.txt
├── README.md
│
└── pages/
    ├── 1_BSM.py            # Black-Scholes-Merton
    ├── 2_CRR.py            # Árboles CRR (europeo + americano)
    ├── 3_Heston.py         # Modelo de Heston
    ├── 4_Merton.py         # Merton jump-diffusion
    ├── 5_Complementos.py   # Griegas, sonrisa IV, paridad put-call
    ├── 6_Comparativa.py    # Benchmark cruzado de modelos
    └── 7_LiveData.py       # Datos en vivo via yfinance
```

### `engine.py` — arquitectura interna

El motor centraliza toda la lógica de pricing en clases independientes con interfaz estandarizada:

```python
class BSMEngine:      # Fórmula cerrada Black-Scholes-Merton
class CRREngine:      # Árbol binomial Cox-Ross-Rubinstein
class HestonEngine:   # Función característica + integración numérica (Albrecher et al.)
class MertonEngine:   # Serie de Poisson sobre BSM (n_terms configurable)

# Helpers
compare_all_models()  # Ejecuta los 4 motores con parámetros comunes
parity_check()        # Verifica paridad put-call
intrinsic_time_value()
```

Todas las clases reciben `(S, K, T, r, sigma, q)` como parámetros base más sus propios extras, lo que garantiza comparabilidad directa en la página 6.

---

## Características

### Modo Básico / Avanzado
Cada página expone un toggle **Básico / Avanzado** en la barra lateral:
- **Básico**: inputs, precio, gráfica principal.
- **Avanzado**: fórmulas matemáticas en pantalla, parámetros adicionales, visualizaciones de mayor profundidad (superficies 3D, heatmaps, convergencia paso a paso).

### Tema claro / oscuro
Toggle persistente via `st.session_state` propagado a todas las páginas, con paleta definida por variables CSS. El tema oscuro usa fondo `#0d0f14` con acento `#4f8ef7`.

### Griegas (página 5)
Implementadas analíticamente en `engine.py`:

| Orden | Griega | Fórmula |
|-------|--------|---------|
| 1.° | Delta (Δ) | ∂C/∂S |
| 1.° | Theta (Θ) | ∂C/∂t |
| 1.° | Vega (ν) | ∂C/∂σ |
| 1.° | Rho (ρ) | ∂C/∂r |
| 2.° | Gamma (Γ) | ∂²C/∂S² |
| 2.° | Vanna | ∂²C/∂S∂σ |
| 2.° | Volga/Vomma | ∂²C/∂σ² |
| 2.° | Charm | ∂Δ/∂t |

### Heston — detalles técnicos
- Función característica via formulación de Albrecher et al. (numéricamente estable para parámetros extremos)
- Integración numérica via `scipy.integrate.quad`
- Verificación de condición de Feller: `2κθ > ξ²`
- Sonrisa de volatilidad por inversión BSM sobre precios Heston

### CRR — detalles técnicos
- Inducción hacia atrás vectorizada con `numpy`
- Ejercicio anticipado americano integrado en la inducción
- Visualización del árbol hasta 10 pasos via `plotly` (nodos coloreados por valor de la opción)
- Gráfica de convergencia CRR → BSM conforme N → ∞

### Merton Jump-Diffusion — detalles técnicos
- Serie de Poisson truncada a `n_terms` configurables (default: 50)
- Parámetros ajustados por número de saltos: `rₙ`, `σₙ`
- Simulación de Monte Carlo de trayectorias con saltos en la página 4
- Sonrisa de volatilidad via inversión BSM

---

## Instalación y ejecución local

```bash
# Clonar el repositorio
git clone https://github.com/Walls16/valuacion-cuantitativa-derivados.git
cd valuacion-cuantitativa-derivados

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

**Requisitos:** Python 3.10+

---

## Deploy en Streamlit Community Cloud

1. Fork este repositorio o súbelo a tu cuenta de GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub
3. **New app** → selecciona el repositorio → `app.py` como archivo principal
4. Deploy — el entorno instala `requirements.txt` automáticamente

> La página de datos en vivo requiere conectividad saliente hacia `query1.finance.yahoo.com`. Streamlit Community Cloud lo permite sin configuración adicional.

---

## Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| `streamlit` | Framework de UI multi-página |
| `numpy` | Álgebra lineal, arrays, inducción hacia atrás |
| `scipy` | Integración numérica (Heston), solver de IV (Brent) |
| `pandas` | Manejo de cadena de opciones |
| `plotly` | Visualizaciones interactivas (2D y 3D) |
| `yfinance` | Datos de mercado en tiempo real |

---

## Fórmulas de referencia

**BSM Call:**
$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

**Heston — función característica:**
$$\phi_j(\omega) = \exp\left(C_j(\omega, T) + D_j(\omega, T) v_0 + i\omega \ln S\right)$$

**Merton — precio como serie:**
$$C_{Merton} = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!} \cdot C_{BSM}(r_n, \sigma_n)$$

**CRR — inducción hacia atrás:**
$$V_{i,j} = e^{-r\Delta t}\left[p \cdot V_{i+1,j} + (1-p) \cdot V_{i+1,j+1}\right]$$

---

## Autor

**Owen** — Estudiante de Actuaría, Universidad de las Américas Puebla (UDLAP)  


---

## Licencia

MIT License — libre para uso académico y personal.
