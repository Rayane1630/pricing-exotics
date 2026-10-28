"""
VOLATILITÉ LOCALE DE DUPIRE
===========================

Objectif
--------
À partir de la surface de vol implicite du marché, construire la surface
de volatilité locale sigma_loc(S, t) : l'unique diffusion
        dS / S = (r - q) dt + sigma_loc(S, t) dW
qui redonne EXACTEMENT tous les prix de vanilles du marché.

Formule de Dupire (version de Gatheral, en variance totale implicite)
--------------------------------------------------------------------
    w(y, T) = vol_imp(y, T)^2 x T       variance totale implicite
    y       = ln(K / F_T)               log-moneyness forward

                              dw/dT
    sigma_loc^2 = -------------------------------------------------------------
                  1 - (y/w) dw/dy + 1/4 (-1/4 - 1/w + y^2/w^2) (dw/dy)^2 + 1/2 d2w/dy2

Lecture :
    - numérateur   = pente de la surface dans le temps (structure par terme) ;
    - dénominateur = effet du smile (pente et convexité en strike).
    - smile plat -> dénominateur = 1 -> sigma_loc = vol implicite.

Vérification
------------
On simule le sous-jacent avec la vol locale (Monte Carlo), on reprice des
calls et on recalcule leur vol implicite : on doit retrouver la surface
de départ. C'est le test de cohérence du modèle.

Lancement
---------
    python vol_locale_dupire.py          -> lit surface_vol_implicite.xlsx (voir commun.py)
    python vol_locale_dupire.py --demo   -> surface fictive
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

from commun import bs_prix, charger, vol_implicite

NB_CHEMINS = 100_000       # nombre de simulations Monte Carlo
PAS_PAR_AN = 252           # pas de temps journalier
GRAINE = 7


# ---------------------------------------------------------------------------
# 1. CALCUL DE LA SURFACE DE VOL LOCALE
# ---------------------------------------------------------------------------

def surface_vol_locale(parametres, maturites, moneyness, vols):
    """
    Renvoie une fonction sigma_loc(x, t), avec x = ln(S / S0).

    On travaille sur une grille rectangulaire en x = ln(K / S0).
    Comme y = x - (r - q) T, les dérivées se convertissent ainsi :
        dw/dy         = dw/dx
        dw/dT (y fixé) = dw/dT (x fixé) + (r - q) dw/dx
    """
    r, q = parametres["r"], parametres["q"]
    x_noeuds = np.log(moneyness)
    w_noeuds = vols ** 2 * maturites[:, None]          # variance totale

    # On ajoute le point T = 0, où la variance totale est nulle par définition :
    # cela donne une vol locale aussi avant la première maturité cotée.
    T_noeuds = np.concatenate([[0.0], maturites])
    w_noeuds = np.vstack([np.zeros(len(x_noeuds)), w_noeuds])

    # Interpolation lisse de w (spline cubique en x, quadratique en T)
    kT = min(2, len(T_noeuds) - 1)
    spline_w = RectBivariateSpline(T_noeuds, x_noeuds, w_noeuds, kx=kT, ky=3, s=0)

    # Grille fine sur laquelle on calcule la vol locale
    T_fin = np.linspace(0.02, maturites[-1], 80)
    x_fin = np.linspace(x_noeuds[0], x_noeuds[-1], 121)
    TT, XX = np.meshgrid(T_fin, x_fin, indexing="ij")

    w = spline_w.ev(TT, XX)
    dw_dx = spline_w.ev(TT, XX, dy=1)
    d2w_dx2 = spline_w.ev(TT, XX, dy=2)
    dw_dT = spline_w.ev(TT, XX, dx=1) + (r - q) * dw_dx
    y = XX - (r - q) * TT

    numerateur = dw_dT
    denominateur = (1 - y / w * dw_dx
                    + 0.25 * (-0.25 - 1 / w + y ** 2 / w ** 2) * dw_dx ** 2
                    + 0.5 * d2w_dx2)

    var_loc = numerateur / denominateur

    # Garde-fou : une variance négative signale un arbitrage dans les données
    nb_negatifs = int(np.sum(var_loc <= 0))
    if nb_negatifs:
        print(f"  Attention : {nb_negatifs} points à variance locale <= 0 "
              "(arbitrage dans la surface), remplacés par un plancher.")
    vol_loc = np.sqrt(np.clip(var_loc, 1e-4, None))

    spline_loc = RectBivariateSpline(T_fin, x_fin, vol_loc, kx=1, ky=1)

    def sigma_loc(x, t):
        """Vol locale au niveau x = ln(S/S0) et à la date t (bornée à la grille)."""
        t_b = np.clip(t, T_fin[0], T_fin[-1])
        x_b = np.clip(x, x_fin[0], x_fin[-1])
        return spline_loc.ev(np.full_like(x_b, t_b), x_b)

    return sigma_loc, T_fin, x_fin, vol_loc


# ---------------------------------------------------------------------------
# 2. MONTE CARLO EN VOL LOCALE
# ---------------------------------------------------------------------------

def monte_carlo_vol_locale(parametres, sigma_loc, dates_a_garder):
    """
    Simule ln(S/S0) pas à pas :
        dx = (r - q - sigma_loc^2 / 2) dt + sigma_loc dW
    Renvoie les valeurs de x aux dates demandées uniquement (économie de mémoire).
    """
    rng = np.random.default_rng(GRAINE)
    r, q = parametres["r"], parametres["q"]
    dt = 1.0 / PAS_PAR_AN
    pas_a_garder = {int(round(T / dt)): T for T in dates_a_garder}
    nb_pas = max(pas_a_garder)

    x = np.zeros(NB_CHEMINS)
    resultats = {}
    for i in range(1, nb_pas + 1):
        t = (i - 0.5) * dt
        sig = sigma_loc(x, t)
        z = rng.standard_normal(NB_CHEMINS)
        x = x + (r - q - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * z
        if i in pas_a_garder:
            resultats[pas_a_garder[i]] = x.copy()
    return resultats


def verification(parametres, maturites, moneyness, vols, sigma_loc):
    """Reprice des calls en Monte Carlo et compare les vols implicites au marché."""
    S0, r, q = parametres["spot"], parametres["r"], parametres["q"]
    T_tests = [T for T in maturites if T >= 0.5][:3]
    x_finaux = monte_carlo_vol_locale(parametres, sigma_loc, T_tests)

    print("\nVérification : vol implicite marché vs Monte Carlo vol locale (en %)")
    print(f"{'Maturité':>9} {'K/S':>6} {'Marché':>8} {'MC':>8} {'Écart':>7}")
    for T in T_tests:
        S_T = S0 * np.exp(x_finaux[T])
        # Correction martingale : on force la moyenne simulée à égaler le forward
        # (réduit fortement le bruit Monte Carlo sur les prix)
        S_T *= S0 * np.exp((r - q) * T) / S_T.mean()
        i_T = list(maturites).index(T)
        for j, m in enumerate(moneyness):
            if not 0.8 <= m <= 1.2:
                continue
            K = m * S0
            prix_mc = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
            vol_mc = vol_implicite(prix_mc, S0, K, T, r, q)
            print(f"{T:>9.2f} {m:>6.2f} {vols[i_T, j] * 100:>8.2f} "
                  f"{vol_mc * 100:>8.2f} {(vol_mc - vols[i_T, j]) * 100:>+7.2f}")


# ---------------------------------------------------------------------------
# 3. GRAPHIQUE
# ---------------------------------------------------------------------------

def tracer(parametres, maturites, moneyness, vols, T_fin, x_fin, vol_loc):
    """Smiles implicites vs smiles locaux, pour quelques maturités."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    couleurs = plt.cm.viridis(np.linspace(0, 0.85, len(maturites)))

    for i, T in enumerate(maturites):
        ax.plot(moneyness, vols[i] * 100, "o-", color=couleurs[i], ms=4,
                label=f"Implicite {T:g} an")
        k = np.argmin(np.abs(T_fin - T))
        ax.plot(np.exp(x_fin), vol_loc[k] * 100, "--", color=couleurs[i], lw=1.2)

    ax.set_title(f"Vol implicite (traits pleins) et vol locale de Dupire (pointillés) "
                 f"– {parametres['sous_jacent']}")
    ax.set_xlabel("Moneyness K / Spot")
    ax.set_ylabel("Volatilité (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    sortie = Path(__file__).with_name("vol_locale_dupire.png")
    fig.savefig(sortie, dpi=150)
    print(f"\nGraphique enregistré : {sortie}")
    plt.show()


# ---------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    parametres, maturites, moneyness, vols = charger("--demo" in sys.argv)

    sigma_loc, T_fin, x_fin, vol_loc = surface_vol_locale(parametres, maturites, moneyness, vols)

    # Lecture rapide : la vol locale ATM est plus pentue que l'implicite
    # (règle empirique : skew local ≈ 2 x skew implicite)
    k1 = np.argmin(np.abs(T_fin - 1.0))
    print(f"\nVol locale à 1 an : K/S=0,90 -> {np.interp(np.log(0.9), x_fin, vol_loc[k1]) * 100:.2f} %"
          f" | ATM -> {np.interp(0.0, x_fin, vol_loc[k1]) * 100:.2f} %"
          f" | K/S=1,10 -> {np.interp(np.log(1.1), x_fin, vol_loc[k1]) * 100:.2f} %")

    verification(parametres, maturites, moneyness, vols, sigma_loc)
    tracer(parametres, maturites, moneyness, vols, T_fin, x_fin, vol_loc)


if __name__ == "__main__":
    main()
