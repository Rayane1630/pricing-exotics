"""
VOLATILITÉ STOCHASTIQUE : MODÈLE DE HESTON
==========================================

Objectif
--------
Calibrer le modèle de Heston sur la surface de vol implicite du marché,
puis vérifier le prix obtenu par Monte Carlo.

Le modèle
---------
La variance v_t n'est plus fixe : elle suit son propre processus aléatoire.

    dS_t / S_t = (r - q) dt + sqrt(v_t) dW1
    dv_t       = kappa (theta - v_t) dt + xi sqrt(v_t) dW2
    corr(dW1, dW2) = rho

    v0    : variance de départ                (vol spot = sqrt(v0))
    theta : variance de long terme            (vol long terme = sqrt(theta))
    kappa : vitesse de retour vers theta      (mean reversion)
    xi    : vol de la variance ("vol of vol") -> courbure du smile
    rho   : corrélation spot / vol            -> pente du smile (skew)
            sur les actions rho < 0 : quand le marché baisse, la vol monte

Condition de Feller : 2 kappa theta > xi^2 garantit que v_t ne touche pas 0.

Prix des options
----------------
Heston a une formule semi-fermée via la fonction caractéristique de ln(S_T)
(Heston 1993, forme "little trap" d'Albrecher et al. 2007, plus stable) :
    Call = S e^{-qT} P1 - K e^{-rT} P2
P1 et P2 sont des probabilités obtenues par une intégrale numérique.

Vol locale vs vol stochastique (question classique d'entretien)
---------------------------------------------------------------
    - La vol locale redonne EXACTEMENT toute la surface d'aujourd'hui,
      mais prévoit un smile futur qui s'aplatit (forward skew trop faible).
    - Heston ne colle pas parfaitement à la surface (5 paramètres seulement),
      mais donne une dynamique du smile plus réaliste.
    - Les desks utilisent souvent un mélange des deux : la vol "locale-stochastique" (LSV).

Lancement
---------
    python vol_stochastique_heston.py          -> lit surface_vol_implicite.xlsx (voir commun.py)
    python vol_stochastique_heston.py --demo   -> surface fictive
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from commun import bs_prix, bs_vega, charger, vol_implicite

# Intégration numérique : grille de u (fréquences)
U_MAX, NB_U = 200.0, 4000
U = np.linspace(1e-6, U_MAX, NB_U)[:, None]

NOMS = ["kappa", "theta", "xi", "rho", "v0"]
DEPART = np.array([1.5, 0.04, 0.5, -0.5, 0.04])
BORNES = ([0.05, 0.001, 0.05, -0.99, 0.001], [10.0, 1.0, 3.0, 0.99, 1.0])

NB_CHEMINS = 100_000
PAS_PAR_AN = 252
GRAINE = 3


# ---------------------------------------------------------------------------
# 1. PRIX SEMI-FERMÉ DE HESTON
# ---------------------------------------------------------------------------

def fonction_caracteristique(u, S, T, r, q, kappa, theta, xi, rho, v0):
    """phi(u) = E[exp(i u ln S_T)] sous Heston (forme "little trap")."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b ** 2 + xi ** 2 * (iu + u ** 2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (r - q) * iu * T + kappa * theta / xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0 + iu * np.log(S))


def heston_call(S, K, T, r, q, params):
    """
    Prix de calls européens (K peut être un vecteur).
        P2 = proba (mesure risque-neutre) que le call finisse dans la monnaie
        P1 = même proba sous la mesure "action" (numéraire S)
    """
    K = np.atleast_1d(K)[None, :]
    lnK = np.log(K)
    phi = fonction_caracteristique(U, S, T, r, q, *params)
    phi_decale = fonction_caracteristique(U - 1j, S, T, r, q, *params)
    forward = S * np.exp((r - q) * T)

    integrande_2 = np.real(np.exp(-1j * U * lnK) * phi / (1j * U))
    integrande_1 = np.real(np.exp(-1j * U * lnK) * phi_decale / (1j * U * forward))
    P2 = 0.5 + np.trapezoid(integrande_2, U[:, 0], axis=0) / np.pi
    P1 = 0.5 + np.trapezoid(integrande_1, U[:, 0], axis=0) / np.pi

    return S * np.exp(-q * T) * P1 - K[0] * np.exp(-r * T) * P2


# ---------------------------------------------------------------------------
# 2. CALIBRATION
# ---------------------------------------------------------------------------

def calibrer(parametres, maturites, moneyness, vols):
    """
    Cherche les 5 paramètres qui minimisent l'écart au marché.
    Erreur = (prix Heston - prix marché) / vega  ≈  écart en points de vol,
    ce qui donne le même poids aux options chères et bon marché.
    """
    S, r, q = parametres["spot"], parametres["r"], parametres["q"]
    K = moneyness * S
    prix_marche = [bs_prix(S, K, T, r, q, vols[i]) for i, T in enumerate(maturites)]
    vegas = [bs_vega(S, K, T, r, q, vols[i]) for i, T in enumerate(maturites)]

    def erreurs(p):
        return np.concatenate([
            (heston_call(S, K, T, r, q, p) - prix_marche[i]) / vegas[i]
            for i, T in enumerate(maturites)
        ])

    resultat = least_squares(erreurs, DEPART, bounds=BORNES, method="trf", x_scale="jac")
    rmse = np.sqrt(np.mean(resultat.fun ** 2)) * 100
    return resultat.x, rmse


def vols_heston(parametres, maturites, moneyness, params):
    """Surface de vol implicite produite par Heston (pour comparer au marché)."""
    S, r, q = parametres["spot"], parametres["r"], parametres["q"]
    K = moneyness * S
    surface = np.zeros((len(maturites), len(moneyness)))
    for i, T in enumerate(maturites):
        prix = heston_call(S, K, T, r, q, params)
        surface[i] = [vol_implicite(p, S, k, T, r, q) for p, k in zip(prix, K)]
    return surface


# ---------------------------------------------------------------------------
# 3. VÉRIFICATION PAR MONTE CARLO
# ---------------------------------------------------------------------------

def monte_carlo_heston(parametres, params, T, K):
    """
    Schéma d'Euler "full truncation" : on remplace v par max(v, 0) dans
    les termes de dérive et de diffusion pour éviter une variance négative.
    """
    rng = np.random.default_rng(GRAINE)
    S, r, q = parametres["spot"], parametres["r"], parametres["q"]
    kappa, theta, xi, rho, v0 = params
    nb_pas = int(round(T * PAS_PAR_AN))
    dt = T / nb_pas

    x = np.zeros(NB_CHEMINS)
    v = np.full(NB_CHEMINS, v0)
    for _ in range(nb_pas):
        z1 = rng.standard_normal(NB_CHEMINS)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(NB_CHEMINS)
        v_pos = np.maximum(v, 0)
        x += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1
        v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z2

    gains = np.exp(-r * T) * np.maximum(S * np.exp(x) - K, 0)
    return gains.mean(), gains.std() / np.sqrt(NB_CHEMINS)


# ---------------------------------------------------------------------------
# 4. GRAPHIQUE
# ---------------------------------------------------------------------------

def tracer(parametres, maturites, moneyness, vols, surface_heston):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    couleurs = plt.cm.viridis(np.linspace(0, 0.85, len(maturites)))
    for i, T in enumerate(maturites):
        ax.plot(moneyness, vols[i] * 100, "o", color=couleurs[i], ms=5, label=f"Marché {T:g} an")
        ax.plot(moneyness, surface_heston[i] * 100, "-", color=couleurs[i], lw=1.3)
    ax.set_title(f"Calibration Heston : marché (points) vs modèle (lignes) – {parametres['sous_jacent']}")
    ax.set_xlabel("Moneyness K / Spot")
    ax.set_ylabel("Volatilité implicite (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    sortie = Path(__file__).with_name("vol_stochastique_heston.png")
    fig.savefig(sortie, dpi=150)
    print(f"\nGraphique enregistré : {sortie}")
    plt.show()


# ---------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    parametres, maturites, moneyness, vols = charger("--demo" in sys.argv)

    print("\nCalibration en cours...")
    params, rmse = calibrer(parametres, maturites, moneyness, vols)

    print("\nParamètres calibrés :")
    for nom, valeur in zip(NOMS, params):
        print(f"  {nom:>6} = {valeur:.4f}")
    kappa, theta, xi, rho, v0 = params
    print(f"  vol spot = {np.sqrt(v0) * 100:.2f} % | vol long terme = {np.sqrt(theta) * 100:.2f} %")
    feller = 2 * kappa * theta > xi ** 2
    print(f"  Feller (2 kappa theta > xi^2) : {'respectée' if feller else 'NON respectée'}")
    print(f"  Erreur moyenne (RMSE) : {rmse:.2f} points de vol")

    # Contrôle du prix semi-fermé par Monte Carlo : call ATM à 1 an
    S, r, q = parametres["spot"], parametres["r"], parametres["q"]
    T_test = 1.0
    prix_formule = heston_call(S, S, T_test, r, q, params)[0]
    prix_mc, erreur_mc = monte_carlo_heston(parametres, params, T_test, S)
    print(f"\nCall ATM 1 an : formule = {prix_formule:.2f} | "
          f"Monte Carlo = {prix_mc:.2f} ± {1.96 * erreur_mc:.2f} (IC 95 %)")

    surface_heston = vols_heston(parametres, maturites, moneyness, params)
    tracer(parametres, maturites, moneyness, vols, surface_heston)


if __name__ == "__main__":
    main()
