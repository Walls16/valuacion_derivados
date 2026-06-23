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


# 
#  Shared parameter dataclass-like dict spec:
#  S, K, T, r, q, sigma  (+ model-specific extras)
# 

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

        #  Delta 
        call_delta = div_disc * cdf_d1
        put_delta  = div_disc * (cdf_d1 - 1)

        #  Gamma 
        gamma = div_disc * pdf_d1 / (self.S * self.sigma * sqrtT)

        #  Theta (per calendar day) 
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

        #  Vega (per 1% move in vol) 
        vega = self.S * div_disc * pdf_d1 * sqrtT / 100

        #  Rho (per 1% move in rate) 
        call_rho = self.K * self.T * discount * cdf_d2 / 100
        put_rho  = -self.K * self.T * discount * norm.cdf(-d2) / 100

        #  Vanna 
        vanna = -div_disc * pdf_d1 * d2 / self.sigma

        #  Volga / Vomma 
        volga = self.S * div_disc * pdf_d1 * sqrtT * d1 * d2 / self.sigma

        #  Charm (delta decay per day) 
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


# 
#  CRR Binomial Tree
# 

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


# 
#  Heston Stochastic Volatility
# 

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


# 
#  Merton Jump-Diffusion
# 

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


# 
#  Comparison helper
# 

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


# 
#  Utilities
# 

def parity_check(call, put, S, K, T, r, q=0.0):
    """Put-Call parity residual."""
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return {"lhs": lhs, "rhs": rhs, "residual": abs(lhs - rhs)}


def intrinsic_time_value(price, S, K, option_type="call"):
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    return {"intrinsic": intrinsic, "time_value": max(price - intrinsic, 0)}


# ─────────────────────────────────────────────
#  SABR Stochastic Volatility Model (Hagan et al. 2002)
# ─────────────────────────────────────────────

class SABREngine:
    """
    SABR model (Hagan, Kumar, Lesniewski, Woodward 2002).
    Provides an analytical approximation for implied volatility,
    then prices via BSM with that IV.

    dF = sigma * F^beta * dW1
    dsigma = alpha * sigma * dW2
    corr(dW1, dW2) = rho

    Parameters
    ----------
    F      : forward price = S * exp((r-q)*T)
    K      : strike
    T      : time to expiry
    alpha  : initial vol (SABR sigma_0)
    beta   : elasticity parameter in [0,1]
             0 = normal (Bachelier-like), 1 = lognormal (BSM-like)
    rho    : correlation dF-dsigma
    nu     : vol of vol (SABR alpha)
    r, q   : for discounting / forward calculation
    S      : spot (optional, used to compute F if not given directly)
    """

    def __init__(self, F, K, T, alpha, beta, rho, nu, r=0.0, q=0.0, S=None):
        self.F     = F
        self.K     = K
        self.T     = T
        self.alpha = alpha
        self.beta  = beta
        self.rho   = rho
        self.nu    = nu
        self.r     = r
        self.q     = q
        self.S     = S if S is not None else F

    def implied_vol(self):
        """
        Hagan et al. (2002) lognormal implied vol approximation.
        Numerically stable for F ≈ K (ATM) via L'Hopital expansion.
        """
        F, K, T      = self.F, self.K, self.T
        alpha, beta  = self.alpha, self.beta
        rho, nu      = self.rho, self.nu

        eps = 1e-8
        if T <= 0:
            return alpha

        FK_mid = (F * K) ** ((1 - beta) / 2)
        ln_FK  = np.log(F / K) if abs(F - K) > eps else 0.0

        # Leading term
        A = alpha / (FK_mid * (1 + (1 - beta)**2 / 24 * ln_FK**2
                                  + (1 - beta)**4 / 1920 * ln_FK**4))

        # z and x(z) correction
        if abs(F - K) > eps:
            z   = nu / alpha * FK_mid * ln_FK
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
            B   = z / x_z if abs(x_z) > eps else 1.0
        else:
            B = 1.0

        # Time-correction term
        C = 1 + ((1-beta)**2/24 * alpha**2 / FK_mid**2
                  + rho*beta*nu*alpha / (4 * FK_mid)
                  + (2 - 3*rho**2) / 24 * nu**2) * T

        return A * B * C

    def call_price(self):
        iv  = self.implied_vol()
        bsm = BSMEngine(self.S, self.K, self.T, self.r, iv, self.q)
        return bsm.call_price()

    def put_price(self):
        iv  = self.implied_vol()
        bsm = BSMEngine(self.S, self.K, self.T, self.r, iv, self.q)
        return bsm.put_price()

    def vol_smile(self, strikes):
        """IV smile across strikes."""
        ivs = []
        for K in strikes:
            eng = SABREngine(self.F, K, self.T, self.alpha, self.beta,
                             self.rho, self.nu, self.r, self.q, self.S)
            ivs.append(eng.implied_vol())
        return np.array(ivs)

    def calibrate(self, strikes, market_ivs, beta=0.5,
                  alpha0=0.2, rho0=-0.3, nu0=0.4, tol=1e-8):
        """
        Calibrate (alpha, rho, nu) to observed IV smile given fixed beta.
        Uses scipy.optimize.minimize with L-BFGS-B.
        Returns dict with calibrated params and RMSE.
        """
        from scipy.optimize import minimize

        def objective(params):
            a, r, n = params
            if a <= 0 or n <= 0 or abs(r) >= 1:
                return 1e6
            total = 0.0
            for K, miv in zip(strikes, market_ivs):
                if np.isnan(miv):
                    continue
                eng = SABREngine(self.F, K, self.T, a, beta, r, n)
                model_iv = eng.implied_vol()
                total += (model_iv - miv)**2
            return total

        res = minimize(objective, [alpha0, rho0, nu0],
                       method="L-BFGS-B",
                       bounds=[(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)],
                       options={"ftol": tol, "maxiter": 500})

        a_cal, r_cal, n_cal = res.x
        rmse = np.sqrt(res.fun / max(len(strikes), 1))
        return {"alpha": a_cal, "beta": beta, "rho": r_cal, "nu": n_cal,
                "rmse": rmse, "success": res.success}


# ─────────────────────────────────────────────
#  Variance Gamma Model (Madan, Carr, Chang 1998)
# ─────────────────────────────────────────────

class VarianceGammaEngine:
    """
    Variance Gamma model (Madan, Carr & Chang 1998).
    X(t) = theta*G(t) + sigma*W(G(t))
    where G(t) ~ Gamma(t/nu, nu) is a Gamma time-change.

    Parameters
    ----------
    S, K, T, r, q : standard
    sigma  : diffusion vol of the VG process
    theta  : drift of Brownian motion (skewness parameter, <0 for left skew)
    nu     : variance of Gamma time (kurtosis parameter, >0)
    """

    def __init__(self, S, K, T, r, sigma, theta, nu, q=0.0):
        self.S     = S
        self.K     = K
        self.T     = T
        self.r     = r
        self.sigma = sigma
        self.theta = theta
        self.nu    = nu
        self.q     = q

    def _omega(self):
        """Risk-neutral drift correction."""
        return (1/self.nu) * np.log(1 - self.theta*self.nu - 0.5*self.sigma**2*self.nu)

    def _char_func(self, u):
        """Characteristic function of log(S_T/S) under Q."""
        T, r, q    = self.T, self.r, self.q
        sigma, theta, nu = self.sigma, self.theta, self.nu
        omega      = self._omega()
        i          = complex(0, 1)
        drift_term = i*u*(np.log(self.S) + (r - q + omega)*T)
        vg_term    = -(T/nu)*np.log(1 - i*u*theta*nu + 0.5*sigma**2*nu*u**2)
        return np.exp(drift_term + vg_term)

    def _price_via_fft(self, option_type="call", N=4096, eta=0.25):
        """
        Option price via Carr-Madan (1999) FFT.
        alpha: dampening factor (call=1.5, put=1.5 with sign flip)
        """
        alpha  = 1.5
        lam    = 2*np.pi / (N * eta)
        b      = np.pi / eta
        k_grid = -b + lam * np.arange(N)
        v_grid = eta * np.arange(N)

        # Carr-Madan modified characteristic function
        def psi(v):
            cf_val = self._char_func(v - (alpha + 1)*1j)
            denom  = alpha**2 + alpha - v**2 + 1j*(2*alpha + 1)*v
            return np.exp(-self.r*self.T) * cf_val / denom

        psi_vals = np.array([psi(v) for v in v_grid])
        simpson  = (eta/3) * np.array([
            3 + (-1)**j - (1 if j == 0 else 0)
            for j in range(N)
        ])
        fft_input = np.exp(1j * b * v_grid) * psi_vals * simpson
        prices_raw = np.real(np.fft.fft(fft_input)) * np.exp(-alpha * k_grid) / np.pi
        log_K      = np.log(self.K)
        # Interpolate at desired log-strike
        from scipy.interpolate import interp1d
        interp = interp1d(k_grid, prices_raw, kind="cubic", fill_value="extrapolate")
        call   = float(interp(log_K))
        call   = max(call, max(self.S*np.exp(-self.q*self.T) - self.K*np.exp(-self.r*self.T), 0))

        if option_type == "call":
            return call
        else:  # put via parity
            return call - self.S*np.exp(-self.q*self.T) + self.K*np.exp(-self.r*self.T)

    def call_price(self):
        return self._price_via_fft("call")

    def put_price(self):
        return self._price_via_fft("put")

    def implied_vol(self, option_type="call"):
        """BSM-inverted IV for VG price."""
        price = self.call_price() if option_type == "call" else self.put_price()
        bsm   = BSMEngine(self.S, self.K, self.T, self.r, 0.3, self.q)
        return bsm.implied_vol(price, option_type)

    def vol_smile(self, strikes, option_type="call"):
        ivs = []
        for K in strikes:
            try:
                eng = VarianceGammaEngine(self.S, K, self.T, self.r,
                                          self.sigma, self.theta, self.nu, self.q)
                iv  = eng.implied_vol(option_type)
                ivs.append(iv * 100 if not np.isnan(iv) else np.nan)
            except Exception:
                ivs.append(np.nan)
        return np.array(ivs)


# ─────────────────────────────────────────────
#  Normal Inverse Gaussian (NIG) Model (Barndorff-Nielsen 1997)
# ─────────────────────────────────────────────

class NIGEngine:
    """
    Normal Inverse Gaussian model via characteristic function + Carr-Madan FFT.

    Parameters
    ----------
    S, K, T, r, q : standard
    alpha  : tail heaviness (>0, larger = lighter tails)
    beta   : skewness (-alpha < beta < alpha, <0 = left skew)
    delta  : scale parameter (>0)
    """

    def __init__(self, S, K, T, r, alpha, beta, delta, q=0.0):
        self.S     = S
        self.K     = K
        self.T     = T
        self.r     = r
        self.alpha = alpha
        self.beta  = beta
        self.delta = delta
        self.q     = q
        assert alpha > 0 and abs(beta) < alpha and delta > 0, "Invalid NIG params"

    def _mu(self):
        """Risk-neutral drift correction."""
        a, b, d = self.alpha, self.beta, self.delta
        return d*(np.sqrt(a**2 - b**2) - np.sqrt(a**2 - (b+1)**2))

    def _char_func(self, u):
        a, b, d = self.alpha, self.beta, self.delta
        i       = complex(0, 1)
        mu      = self._mu()
        drift   = i*u*(np.log(self.S) + (self.r - self.q + mu)*self.T)
        nig     = -d*self.T*(np.sqrt(a**2 - (b + i*u)**2) - np.sqrt(a**2 - b**2))
        return np.exp(drift + nig)

    def _price_via_fft(self, option_type="call", N=4096, eta=0.25):
        alpha_d = 1.5
        lam     = 2*np.pi / (N * eta)
        b       = np.pi / eta
        k_grid  = -b + lam * np.arange(N)
        v_grid  = eta * np.arange(N)

        def psi(v):
            cf_val = self._char_func(v - (alpha_d+1)*1j)
            denom  = alpha_d**2 + alpha_d - v**2 + 1j*(2*alpha_d+1)*v
            return np.exp(-self.r*self.T) * cf_val / denom

        psi_vals  = np.array([psi(v) for v in v_grid])
        simpson   = (eta/3)*np.array([3+(-1)**j-(1 if j==0 else 0) for j in range(N)])
        fft_input = np.exp(1j*b*v_grid) * psi_vals * simpson
        prices_raw= np.real(np.fft.fft(fft_input))*np.exp(-alpha_d*k_grid)/np.pi

        from scipy.interpolate import interp1d
        interp = interp1d(k_grid, prices_raw, kind="cubic", fill_value="extrapolate")
        call   = float(interp(np.log(self.K)))
        call   = max(call, max(self.S*np.exp(-self.q*self.T)-self.K*np.exp(-self.r*self.T), 0))

        if option_type == "call":
            return call
        return call - self.S*np.exp(-self.q*self.T) + self.K*np.exp(-self.r*self.T)

    def call_price(self):
        return self._price_via_fft("call")

    def put_price(self):
        return self._price_via_fft("put")

    def implied_vol(self, option_type="call"):
        price = self.call_price() if option_type == "call" else self.put_price()
        bsm   = BSMEngine(self.S, self.K, self.T, self.r, 0.3, self.q)
        return bsm.implied_vol(price, option_type)

    def vol_smile(self, strikes, option_type="call"):
        ivs = []
        for K in strikes:
            try:
                eng = NIGEngine(self.S, K, self.T, self.r,
                                self.alpha, self.beta, self.delta, self.q)
                iv  = eng.implied_vol(option_type)
                ivs.append(iv*100 if not np.isnan(iv) else np.nan)
            except Exception:
                ivs.append(np.nan)
        return np.array(ivs)


# ─────────────────────────────────────────────
#  Bachelier (Normal) Model
# ─────────────────────────────────────────────

class BachelierEngine:
    """
    Bachelier (1900) normal model — assumes arithmetic Brownian motion.
    dS = sigma_n * dW  (absolute, not percentage)

    Used in: interest rate options, inflation derivatives, negative-rate environments.
    sigma_n is the *normal* vol (in price units, not percentage).

    Parameters
    ----------
    S, K, T, r, q : standard
    sigma_n        : normal volatility (absolute, e.g. $2 per year)
    """

    def __init__(self, S, K, T, r, sigma_n, q=0.0):
        self.S       = S
        self.K       = K
        self.T       = T
        self.r       = r
        self.sigma_n = sigma_n
        self.q       = q

    def _d(self):
        if self.T <= 0 or self.sigma_n <= 0:
            return 0.0
        return (self.S*np.exp(-self.q*self.T) - self.K*np.exp(-self.r*self.T)) / \
               (self.sigma_n * np.sqrt(self.T))

    def call_price(self):
        """Bachelier call: C = e^{-rT}[(F-K)N(d) + sigma_n*sqrt(T)*n(d)]"""
        if self.T <= 0:
            return max(self.S - self.K, 0.0)
        F   = self.S * np.exp((self.r - self.q) * self.T)
        vol = self.sigma_n * np.sqrt(self.T)
        d   = (F - self.K) / vol
        return np.exp(-self.r*self.T) * ((F - self.K)*norm.cdf(d) + vol*norm.pdf(d))

    def put_price(self):
        """Bachelier put: P = e^{-rT}[(K-F)N(-d) + sigma_n*sqrt(T)*n(d)]"""
        if self.T <= 0:
            return max(self.K - self.S, 0.0)
        F   = self.S * np.exp((self.r - self.q) * self.T)
        vol = self.sigma_n * np.sqrt(self.T)
        d   = (F - self.K) / vol
        return np.exp(-self.r*self.T) * ((self.K - F)*norm.cdf(-d) + vol*norm.pdf(d))

    def greeks(self):
        """Analytical Bachelier Greeks."""
        F   = self.S * np.exp((self.r - self.q)*self.T)
        vol = self.sigma_n * np.sqrt(self.T)
        d   = (F - self.K) / vol if vol > 0 else 0.0
        disc = np.exp(-self.r * self.T)

        call_delta = disc * norm.cdf(d)
        put_delta  = disc * (norm.cdf(d) - 1)
        gamma      = disc * norm.pdf(d) / (self.sigma_n * np.sqrt(self.T)) if self.T > 0 else 0.0
        vega       = disc * np.sqrt(self.T) * norm.pdf(d)            # per unit sigma_n
        call_theta = (-disc * self.sigma_n * norm.pdf(d) / (2*np.sqrt(self.T))
                      - self.r * self.call_price()) / 365 if self.T > 0 else 0.0

        return {
            "call_delta": call_delta,
            "put_delta":  put_delta,
            "gamma":      gamma,
            "vega":       vega,
            "call_theta": call_theta,
        }

    def implied_normal_vol(self, market_price, option_type="call", tol=1e-8):
        """Invert Bachelier formula to get normal vol from market price."""
        def obj(sn):
            eng = BachelierEngine(self.S, self.K, self.T, self.r, sn, self.q)
            return (eng.call_price() if option_type=="call" else eng.put_price()) - market_price
        try:
            return brentq(obj, 1e-6, self.S*10, xtol=tol)
        except Exception:
            return np.nan

    def lognormal_vol(self):
        """
        Approximate equivalent BSM lognormal vol via Hagan formula:
        sigma_LN ≈ sigma_N / F  for ATM.
        More accurate via Bachelier-BSM parity at the given strike.
        """
        bsm = BSMEngine(self.S, self.K, self.T, self.r, 0.3, self.q)
        price = self.call_price()
        return bsm.implied_vol(price, "call")

    def vol_smile(self, strikes):
        """
        Bachelier IV smile — in a pure normal model this is FLAT.
        Useful to see vs lognormal smile.
        """
        return np.array([self.sigma_n for _ in strikes])


# ─────────────────────────────────────────────
#  Heston Calibration
# ─────────────────────────────────────────────

def calibrate_heston(S, T, r, q, strikes, market_ivs,
                     kappa0=2.0, theta0=0.04, xi0=0.3, rho0=-0.5, v00=0.04,
                     method="L-BFGS-B", tol=1e-8, max_iter=300):
    """
    Calibrate Heston to a market IV smile via minimizing RMSE.

    Parameters
    ----------
    S, T, r, q    : spot, maturity, rate, dividend
    strikes       : array of market strikes
    market_ivs    : array of market implied vols (as decimals, e.g. 0.20)
    Returns       : dict with calibrated params + RMSE + success flag
    """
    from scipy.optimize import minimize

    def objective(params):
        kappa, theta, xi, rho, v0 = params
        # Parameter constraints
        if (kappa <= 0 or theta <= 0 or xi <= 0
                or abs(rho) >= 1 or v0 <= 0
                or 2*kappa*theta <= xi**2 * 0.1):  # soft Feller
            return 1e8
        total = 0.0
        for K, miv in zip(strikes, market_ivs):
            if np.isnan(miv) or miv <= 0:
                continue
            try:
                h     = HestonEngine(S, K, T, r, q, v0, kappa, theta, xi, rho)
                price = h.call_price()
                bsm   = BSMEngine(S, K, T, r, 0.3, q)
                model_iv = bsm.implied_vol(price, "call")
                if np.isnan(model_iv):
                    total += 1.0
                else:
                    total += (model_iv - miv)**2
            except Exception:
                total += 1.0
        return total

    x0     = [kappa0, theta0, xi0, rho0, v00]
    bounds = [(0.01, 15.0), (0.001, 1.0), (0.01, 2.0), (-0.999, 0.999), (0.0001, 1.0)]

    res = minimize(objective, x0, method=method, bounds=bounds,
                   options={"maxiter": max_iter, "ftol": tol})

    kappa_c, theta_c, xi_c, rho_c, v0_c = res.x
    n_valid = sum(1 for iv in market_ivs if not np.isnan(iv) and iv > 0)
    rmse    = np.sqrt(res.fun / max(n_valid, 1))

    return {
        "kappa":   kappa_c,
        "theta":   theta_c,
        "xi":      xi_c,
        "rho":     rho_c,
        "v0":      v0_c,
        "rmse":    rmse,
        "success": res.success,
        "message": res.message,
    }


# ─────────────────────────────────────────────
#  Historical Volatility
# ─────────────────────────────────────────────

def realized_vol(prices, window=21, annualize=252):
    """
    Compute rolling realized (historical) volatility from price series.

    Parameters
    ----------
    prices   : array-like of closing prices
    window   : rolling window in days (default 21 = ~1 month)
    annualize: trading days per year for annualization

    Returns
    -------
    rv : np.array of annualized realized vols (same length as prices, NaN for first window)
    """
    prices  = np.array(prices, dtype=float)
    log_ret = np.log(prices[1:] / prices[:-1])
    rv      = np.full(len(prices), np.nan)
    for i in range(window, len(log_ret)+1):
        rv[i] = np.std(log_ret[i-window:i], ddof=1) * np.sqrt(annualize)
    return rv


def iv_rv_spread(iv_series, rv_series):
    """
    Compute IV - RV spread (variance risk premium proxy).
    Positive spread = market paying premium for vol insurance.
    """
    iv  = np.array(iv_series, dtype=float)
    rv  = np.array(rv_series, dtype=float)
    spread = iv - rv
    return {
        "spread":      spread,
        "mean_spread": np.nanmean(spread),
        "std_spread":  np.nanstd(spread),
        "pct_positive": np.nanmean(spread > 0) * 100,
    }


# ─────────────────────────────────────────────
#  Monte Carlo with Confidence Intervals
# ─────────────────────────────────────────────

def monte_carlo_bsm(S, K, T, r, sigma, q=0.0,
                    option_type="call", n_paths=50_000, seed=42,
                    antithetic=True, control_variate=True):
    """
    Monte Carlo pricing of European options under GBM with variance reduction.

    Variance reduction techniques
    ─────────────────────────────
    antithetic    : pairs each path z with -z, halves variance
    control_variate: uses BSM analytical price as control to reduce MC error

    Returns
    -------
    dict with: price, std_error, ci_95_lo, ci_95_hi, n_paths, bsm_price
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        z    = rng.standard_normal(half)
        z    = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    # Terminal stock prices under GBM
    ST = S * np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)

    # Payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    disc_payoffs = np.exp(-r*T) * payoffs

    if control_variate:
        # Use BSM as control: E[payoff_BSM] is known analytically
        bsm_price = BSMEngine(S, K, T, r, sigma, q).call_price() if option_type=="call" \
                    else BSMEngine(S, K, T, r, sigma, q).put_price()
        # Regress payoffs on BSM path prices to find optimal beta_cv
        bsm_ST_price = np.exp(-r*T) * np.maximum(ST - K, 0.0) if option_type=="call" \
                       else np.exp(-r*T) * np.maximum(K - ST, 0.0)
        cov_matrix = np.cov(disc_payoffs, bsm_ST_price)
        beta_cv    = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1,1] > 0 else 0.0
        adjusted   = disc_payoffs - beta_cv*(bsm_ST_price - bsm_price)
        price      = np.mean(adjusted)
        std_err    = np.std(adjusted, ddof=1) / np.sqrt(n_paths)
    else:
        bsm_price = BSMEngine(S, K, T, r, sigma, q).call_price() if option_type=="call" \
                    else BSMEngine(S, K, T, r, sigma, q).put_price()
        price     = np.mean(disc_payoffs)
        std_err   = np.std(disc_payoffs, ddof=1) / np.sqrt(n_paths)

    z95 = 1.96
    return {
        "price":      float(price),
        "std_error":  float(std_err),
        "ci_95_lo":   float(price - z95*std_err),
        "ci_95_hi":   float(price + z95*std_err),
        "n_paths":    n_paths,
        "bsm_price":  float(bsm_price),
        "error_vs_bsm": float(abs(price - bsm_price)),
    }
