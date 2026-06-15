"""
engine.py — Quantitative Derivatives Pricing Engine
=====================================================
Modular pricing engines for BSM, CRR, Heston, and Merton Jump-Diffusion models.
All engines share a standardized parameter interface for clean cross-model comparison.
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Shared parameter dataclass-like dict spec:
#  S, K, T, r, q, sigma  (+ model-specific extras)
# ─────────────────────────────────────────────

class BSMEngine:
    """Black-Scholes-Merton analytical pricing."""

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def _d1(self):
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        if self.T <= 0:
            return max(self.S - self.K, 0.0)
        d1, d2 = self._d1(), self._d2()
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
                - self.K * np.exp(-self.r * self.T) * norm.cdf(d2))

    def put_price(self):
        if self.T <= 0:
            return max(self.K - self.S, 0.0)
        d1, d2 = self._d1(), self._d2()
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))

    def greeks(self):
        """Returns dict of all first and second-order Greeks."""
        d1, d2 = self._d1(), self._d2()
        sqrtT = np.sqrt(self.T) if self.T > 0 else 1e-10
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        discount = np.exp(-self.r * self.T)
        div_disc = np.exp(-self.q * self.T)

        # ── Delta ──
        call_delta = div_disc * cdf_d1
        put_delta  = div_disc * (cdf_d1 - 1)

        # ── Gamma ──
        gamma = div_disc * pdf_d1 / (self.S * self.sigma * sqrtT)

        # ── Theta (per calendar day) ──
        call_theta = (
            - (self.S * div_disc * pdf_d1 * self.sigma) / (2 * sqrtT)
            - self.r * self.K * discount * cdf_d2
            + self.q * self.S * div_disc * cdf_d1
        ) / 365

        put_theta = (
            - (self.S * div_disc * pdf_d1 * self.sigma) / (2 * sqrtT)
            + self.r * self.K * discount * norm.cdf(-d2)
            - self.q * self.S * div_disc * norm.cdf(-d1)
        ) / 365

        # ── Vega (per 1% move in vol) ──
        vega = self.S * div_disc * pdf_d1 * sqrtT / 100

        # ── Rho (per 1% move in rate) ──
        call_rho = self.K * self.T * discount * cdf_d2 / 100
        put_rho  = -self.K * self.T * discount * norm.cdf(-d2) / 100

        # ── Vanna ──
        vanna = -div_disc * pdf_d1 * d2 / self.sigma

        # ── Volga / Vomma ──
        volga = self.S * div_disc * pdf_d1 * sqrtT * d1 * d2 / self.sigma

        # ── Charm (delta decay per day) ──
        call_charm = (div_disc * (
            pdf_d1 * (self.r - self.q) / (self.sigma * sqrtT)
            - d2 / (2 * self.T) * pdf_d1
        )) / 365

        return {
            "call_delta": call_delta,
            "put_delta":  put_delta,
            "gamma":      gamma,
            "call_theta": call_theta,
            "put_theta":  put_theta,
            "vega":       vega,
            "call_rho":   call_rho,
            "put_rho":    put_rho,
            "vanna":      vanna,
            "volga":      volga,
            "call_charm": call_charm,
        }

    def implied_vol(self, market_price, option_type="call", tol=1e-6, max_iter=200):
        """Bisection/Brent IV solver."""
        def objective(sigma):
            eng = BSMEngine(self.S, self.K, self.T, self.r, sigma, self.q)
            return (eng.call_price() if option_type == "call" else eng.put_price()) - market_price
        try:
            return brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        except Exception:
            return np.nan

    def vol_smile_surface(self, strikes, maturities):
        """
        Returns a 2D array of implied vols for a given vol surface.
        Used in Page 5 smile visualization.
        (Assumes flat vol input; real usage feeds market prices.)
        """
        surface = np.full((len(maturities), len(strikes)), self.sigma)
        return surface


# ─────────────────────────────────────────────
#  CRR Binomial Tree
# ─────────────────────────────────────────────

class CRREngine:
    """Cox-Ross-Rubinstein binomial tree."""

    def __init__(self, S, K, T, r, sigma, q=0.0, N=200, american=False):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N = N
        self.american = american

    def _tree_params(self):
        dt = self.T / self.N
        u  = np.exp(self.sigma * np.sqrt(dt))
        d  = 1.0 / u
        p  = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)
        return dt, u, d, p, disc

    def price(self, option_type="call"):
        dt, u, d, p, disc = self._tree_params()
        N = self.N

        # Terminal stock prices
        ST = self.S * (u ** (np.arange(N, -1, -1))) * (d ** (np.arange(0, N + 1, 1)))

        # Terminal payoffs
        if option_type == "call":
            V = np.maximum(ST - self.K, 0)
        else:
            V = np.maximum(self.K - ST, 0)

        # Backward induction
        for _ in range(N):
            V = disc * (p * V[:-1] + (1 - p) * V[1:])
            if self.american:
                ST = ST[:-1] / u  # one step back
                if option_type == "call":
                    V = np.maximum(V, np.maximum(ST - self.K, 0))
                else:
                    V = np.maximum(V, np.maximum(self.K - ST, 0))

        return float(V[0])

    def call_price(self):
        return self.price("call")

    def put_price(self):
        return self.price("put")

    def full_tree(self, option_type="call", max_display=7):
        """Returns stock and option trees truncated to max_display steps for visualization."""
        dt, u, d, p, disc = self._tree_params()
        N = min(self.N, max_display)

        S_tree = [[0.0] * (i + 1) for i in range(N + 1)]
        for i in range(N + 1):
            for j in range(i + 1):
                S_tree[i][j] = self.S * (u ** (i - j)) * (d ** j)

        # Terminal payoffs
        if option_type == "call":
            V_tree = [max(s - self.K, 0) for s in S_tree[N]]
        else:
            V_tree = [max(self.K - s, 0) for s in S_tree[N]]

        V_full = [None] * N + [V_tree]
        for i in range(N - 1, -1, -1):
            row = []
            for j in range(i + 1):
                val = disc * (p * V_full[i + 1][j] + (1 - p) * V_full[i + 1][j + 1])
                if self.american:
                    s = S_tree[i][j]
                    intrinsic = max(s - self.K, 0) if option_type == "call" else max(self.K - s, 0)
                    val = max(val, intrinsic)
                row.append(val)
            V_full[i] = row

        return S_tree, V_full

    def delta(self, option_type="call"):
        dt, u, d, p, disc = self._tree_params()
        Su = self.S * u
        Sd = self.S * d
        eng_u = CRREngine(Su, self.K, self.T - dt, self.r, self.sigma, self.q, self.N - 1, self.american)
        eng_d = CRREngine(Sd, self.K, self.T - dt, self.r, self.sigma, self.q, self.N - 1, self.american)
        fu = eng_u.price(option_type)
        fd = eng_d.price(option_type)
        return (fu - fd) / (Su - Sd)


# ─────────────────────────────────────────────
#  Heston Stochastic Volatility
# ─────────────────────────────────────────────

class HestonEngine:
    """
    Heston (1993) stochastic volatility model via semi-analytical characteristic function.
    dS = (r-q)S dt + sqrt(v) S dW1
    dv = kappa*(theta - v) dt + xi*sqrt(v) dW2,  corr(dW1,dW2) = rho
    """

    def __init__(self, S, K, T, r, q, v0, kappa, theta, xi, rho):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.v0 = v0        # initial variance
        self.kappa = kappa  # mean reversion speed
        self.theta = theta  # long-run variance
        self.xi = xi        # vol of vol
        self.rho = rho      # correlation

    def _char_func(self, phi, j):
        """
        Heston characteristic function — Albrecher et al. (2007) stable formulation.
        Avoids the discontinuity in the complex log branch cut.
        """
        S, K, T = self.S, self.K, self.T
        r, q = self.r, self.q
        v0, kappa, theta, xi, rho = self.v0, self.kappa, self.theta, self.xi, self.rho

        i = complex(0, 1)

        if j == 1:
            u = 0.5
            b = kappa - rho * xi
        else:
            u = -0.5
            b = kappa

        a  = kappa * theta
        x  = np.log(S)
        ln_K = np.log(K)

        d = np.sqrt((rho * xi * i * phi - b)**2 - xi**2 * (2 * u * i * phi - phi**2))

        # Use the formulation that avoids the principal value discontinuity
        g2 = (b - rho * xi * i * phi - d) / (b - rho * xi * i * phi + d)

        exp_dT = np.exp(-d * T)

        C = (r - q) * i * phi * T + (a / xi**2) * (
            (b - rho * xi * i * phi - d) * T
            - 2.0 * np.log((1.0 - g2 * exp_dT) / (1.0 - g2))
        )
        D = ((b - rho * xi * i * phi - d) / xi**2) * (
            (1.0 - exp_dT) / (1.0 - g2 * exp_dT)
        )

        return np.exp(C + D * v0 + i * phi * (x - ln_K))

    def _integrand(self, phi, j):
        return np.real(self._char_func(phi, j) / (complex(0, 1) * phi))

    def _Pj(self, j, upper=200, limit=500):
        integral, _ = quad(self._integrand, 1e-6, upper, args=(j,), limit=limit)
        return 0.5 + integral / np.pi

    def call_price(self):
        P1 = self._Pj(1)
        P2 = self._Pj(2)
        return (self.S * np.exp(-self.q * self.T) * P1
                - self.K * np.exp(-self.r * self.T) * P2)

    def put_price(self):
        call = self.call_price()
        return call - self.S * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)

    def vol_smile(self, strikes):
        """Compute implied vol smile across strikes using BSM inversion."""
        ivs = []
        for K in strikes:
            eng_h = HestonEngine(self.S, K, self.T, self.r, self.q,
                                 self.v0, self.kappa, self.theta, self.xi, self.rho)
            price = eng_h.call_price()
            bsm = BSMEngine(self.S, K, self.T, self.r, 0.3, self.q)
            iv = bsm.implied_vol(price, "call")
            ivs.append(iv)
        return np.array(ivs)


# ─────────────────────────────────────────────
#  Merton Jump-Diffusion
# ─────────────────────────────────────────────

class MertonEngine:
    """
    Merton (1976) jump-diffusion model.
    Price is a Poisson-weighted sum of BSM prices with adjusted parameters.
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, lam=1.0, mu_j=0.0, sigma_j=0.2, n_terms=50):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.lam = lam        # jump intensity (jumps per year)
        self.mu_j = mu_j      # mean log-jump size
        self.sigma_j = sigma_j  # jump vol
        self.n_terms = n_terms

    def _kappa(self):
        """Expected proportional jump size."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

    def price(self, option_type="call"):
        kappa = self._kappa()
        lam_prime = self.lam * (1 + kappa)
        total = 0.0

        for n in range(self.n_terms):
            # Poisson weight under λ' = λ(1+κ)
            poisson_w = np.exp(-lam_prime * self.T) * (lam_prime * self.T)**n / math.factorial(n)
            if poisson_w < 1e-15:
                break

            # Adjusted vol for n jumps
            sigma_n = np.sqrt(self.sigma**2 + n * self.sigma_j**2 / self.T)

            # Adjusted rate: compensate for jump drift so risk-neutral condition holds
            r_n = self.r - self.lam * kappa + n * (self.mu_j + 0.5 * self.sigma_j**2) / self.T

            bsm = BSMEngine(self.S, self.K, self.T, r_n, sigma_n, self.q)
            price_n = bsm.call_price() if option_type == "call" else bsm.put_price()
            total += poisson_w * price_n

        return total

    def call_price(self):
        return self.price("call")

    def put_price(self):
        return self.price("put")

    def vol_smile(self, strikes):
        """Jump-diffusion implied vol smile."""
        ivs = []
        for K in strikes:
            eng_m = MertonEngine(self.S, K, self.T, self.r, self.sigma,
                                 self.q, self.lam, self.mu_j, self.sigma_j, self.n_terms)
            price = eng_m.call_price()
            bsm = BSMEngine(self.S, K, self.T, self.r, 0.3, self.q)
            iv = bsm.implied_vol(price, "call")
            ivs.append(iv)
        return np.array(ivs)


# ─────────────────────────────────────────────
#  Comparison helper
# ─────────────────────────────────────────────

def compare_all_models(S, K, T, r, sigma, q=0.0,
                       heston_params=None, merton_params=None,
                       crr_N=200, crr_american=False):
    """
    Run all four engines with compatible parameters.
    Returns a dict with call/put prices per model.
    heston_params: dict(v0, kappa, theta, xi, rho)
    merton_params: dict(lam, mu_j, sigma_j)
    """
    results = {}

    # BSM
    bsm = BSMEngine(S, K, T, r, sigma, q)
    results["BSM"] = {"call": bsm.call_price(), "put": bsm.put_price()}

    # CRR
    crr = CRREngine(S, K, T, r, sigma, q, crr_N, crr_american)
    results["CRR"] = {"call": crr.call_price(), "put": crr.put_price()}

    # Heston
    if heston_params is None:
        heston_params = {"v0": sigma**2, "kappa": 2.0, "theta": sigma**2, "xi": 0.3, "rho": -0.7}
    h = HestonEngine(S, K, T, r, q,
                     heston_params["v0"], heston_params["kappa"],
                     heston_params["theta"], heston_params["xi"], heston_params["rho"])
    results["Heston"] = {"call": h.call_price(), "put": h.put_price()}

    # Merton
    if merton_params is None:
        merton_params = {"lam": 1.0, "mu_j": -0.1, "sigma_j": 0.15}
    m = MertonEngine(S, K, T, r, sigma, q,
                     merton_params["lam"], merton_params["mu_j"], merton_params["sigma_j"])
    results["Merton"] = {"call": m.call_price(), "put": m.put_price()}

    return results


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def parity_check(call, put, S, K, T, r, q=0.0):
    """Put-Call parity residual."""
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return {"lhs": lhs, "rhs": rhs, "residual": abs(lhs - rhs)}


def intrinsic_time_value(price, S, K, option_type="call"):
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    return {"intrinsic": intrinsic, "time_value": max(price - intrinsic, 0)}
