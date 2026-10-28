"""
PRICER D'AUTOCALL (MONTE CARLO, BLACK-SCHOLES)
==============================================

Le produit (autocall "Athena" à effet mémoire sur un indice)
------------------------------------------------------------
À chaque date d'observation annuelle t_i :
    - si S(t_i) >= barrière autocall x S0 :
          remboursement anticipé = nominal x (1 + coupon x t_i)   -> le produit s'arrête
    - sinon : on continue.

À maturité T, si le produit n'a pas été rappelé :
    - S(T) >= barrière autocall x S0   : nominal x (1 + coupon x T)
    - S(T) >= barrière de protection   : nominal (capital protégé)
    - S(T) <  barrière de protection   : nominal x S(T) / S0 (perte en capital)

Décomposition financière : l'investisseur est
    - long d'une obligation zéro-coupon ;
    - long de digitales (les coupons conditionnels) ;
    - COURT d'un put down-and-in (la protection conditionnelle) : c'est
      cette vente de put qui finance les coupons élevés.

Modèle
------
Black-Scholes : dS/S = (r - q) dt + sigma dW. Comme les observations sont
discrètes (annuelles), on simule S exactement aux dates d'observation,
sans pas de temps intermédiaires.

Greeks
------
Par "bump and revalue" avec les MÊMES nombres aléatoires (common random
numbers) : sans ça, le bruit Monte Carlo noierait l'effet du bump.

Validation
----------
Un cas particulier a une formule fermée : sans autocall et sans coupon,
le produit = zéro-coupon - (1/S0) x [Put(B) + (S0 - B) x Digitale put(B)].
Le Monte Carlo doit retrouver ce prix.

Lancement
---------
    python pricer_autocall.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# PARAMÈTRES DU PRODUIT ET DU MARCHÉ (à modifier)
# ---------------------------------------------------------------------------

PRODUIT = {
    "nominal": 100.0,
    "maturite": 5,                 # années
    "barriere_autocall": 1.00,     # 100 % du spot initial
    "coupon_annuel": 0.075,        # 7,5 % par an, versé avec effet mémoire
    "barriere_protection": 0.60,   # capital protégé jusqu'à -40 %
}

MARCHE = {
    "spot": 5500.0,
    "vol": 0.20,
    "r": 0.025,                    # taux sans risque
    "q": 0.030,                    # taux de dividende
}

NB_CHEMINS = 200_000
GRAINE = 2026


# ---------------------------------------------------------------------------
# 1. SIMULATION ET PAYOFF
# ---------------------------------------------------------------------------

def tirages(nb_obs):
    """Nombres aléatoires gaussiens, tirés une seule fois et réutilisés partout."""
    return np.random.default_rng(GRAINE).standard_normal((NB_CHEMINS, nb_obs))


def prix_autocall(produit, spot, vol, r, q, Z, details=False):
    """
    Prix Monte Carlo de l'autocall.
    Z : matrice (chemins x dates d'observation) de tirages gaussiens.
    """
    T = produit["maturite"]
    dates = np.arange(1, T + 1, dtype=float)
    dt = np.diff(np.concatenate([[0.0], dates]))

    # Performance S(t_i) / S0 sur chaque chemin, simulée exactement
    increments = (r - q - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z
    perf = (spot / MARCHE["spot"]) * np.exp(np.cumsum(increments, axis=1))
    # Remarque : la performance est mesurée par rapport au spot de STRIKE
    # (MARCHE["spot"]), ce qui permet de bumper le spot pour les Greeks.

    nominal, coupon = produit["nominal"], produit["coupon_annuel"]
    valeur = np.zeros(NB_CHEMINS)
    vivant = np.ones(NB_CHEMINS, dtype=bool)
    proba_rappel = []

    for i, t in enumerate(dates[:-1]):
        rappel = vivant & (perf[:, i] >= produit["barriere_autocall"])
        valeur[rappel] = nominal * (1 + coupon * t) * np.exp(-r * t)
        vivant &= ~rappel
        proba_rappel.append(rappel.mean())

    # Maturité
    perf_T = perf[:, -1]
    gain_T = np.where(perf_T >= produit["barriere_autocall"], nominal * (1 + coupon * T),
             np.where(perf_T >= produit["barriere_protection"], nominal, nominal * perf_T))
    valeur[vivant] = gain_T[vivant] * np.exp(-r * T)
    proba_rappel.append((vivant & (perf_T >= produit["barriere_autocall"])).mean())

    prix = valeur.mean()
    if not details:
        return prix

    perte = (vivant & (perf_T < produit["barriere_protection"])).mean()
    duree_moyenne = sum(p * t for p, t in zip(proba_rappel, dates)) + (1 - sum(proba_rappel)) * T
    return {
        "prix": prix,
        "erreur_std": valeur.std() / np.sqrt(NB_CHEMINS),
        "proba_rappel": proba_rappel,
        "proba_perte": perte,
        "duree_moyenne": duree_moyenne,
    }


# ---------------------------------------------------------------------------
# 2. GREEKS PAR BUMP AND REVALUE
# ---------------------------------------------------------------------------

def greeks(produit, marche, Z):
    S, v, r, q = marche["spot"], marche["vol"], marche["r"], marche["q"]
    p0 = prix_autocall(produit, S, v, r, q, Z)
    p_haut = prix_autocall(produit, S * 1.01, v, r, q, Z)
    p_bas = prix_autocall(produit, S * 0.99, v, r, q, Z)
    return {
        "Delta (pour +1 % de spot)": (p_haut - p_bas) / 2,
        "Gamma (pour 1 % de spot)": p_haut - 2 * p0 + p_bas,
        "Vega (pour +1 pt de vol)": prix_autocall(produit, S, v + 0.01, r, q, Z) - p0,
        "Sensi dividende (pour +10 bp)": prix_autocall(produit, S, v, r, q + 0.001, Z) - p0,
        "Rho (pour +10 bp de taux)": prix_autocall(produit, S, v, r + 0.001, q, Z) - p0,
    }


# ---------------------------------------------------------------------------
# 3. VALIDATION PAR FORMULE FERMÉE (BLACK-SCHOLES)
# ---------------------------------------------------------------------------

def validation(marche):
    """
    Sans autocall (barrière infinie) et sans coupon, à maturité T :
        gain = N                          si S_T >= B
             = N x S_T / S0               sinon
    soit  N - (N / S0) x (S0 - S_T) x 1{S_T < B}
    et    (S0 - S_T) x 1{S_T < B} = Put(K=B) + (S0 - B) x Digitale_put(B)
    """
    S, v, r, q = marche["spot"], marche["vol"], marche["r"], marche["q"]
    produit_test = dict(PRODUIT, barriere_autocall=np.inf, coupon_annuel=0.0)
    T, N = produit_test["maturite"], produit_test["nominal"]
    B = produit_test["barriere_protection"] * S

    d1 = (np.log(S / B) + (r - q + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    put = B * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    digitale_put = np.exp(-r * T) * norm.cdf(-d2)
    prix_ferme = N * np.exp(-r * T) - N / S * (put + (S - B) * digitale_put)

    res = prix_autocall(produit_test, S, v, r, q, tirages(T), details=True)
    print("\nValidation (sans autocall ni coupon) :")
    print(f"  Formule fermée BS = {prix_ferme:.3f} | Monte Carlo = {res['prix']:.3f} "
          f"± {1.96 * res['erreur_std']:.3f}")


# ---------------------------------------------------------------------------
# 4. GRAPHIQUE : PRIX ET DELTA SELON LE NIVEAU DU SPOT
# ---------------------------------------------------------------------------

def tracer_profil(produit, marche, Z):
    S0 = marche["spot"]
    niveaux = np.linspace(0.5, 1.5, 41)
    prix = [prix_autocall(produit, S0 * n, marche["vol"], marche["r"], marche["q"], Z) for n in niveaux]
    delta = np.gradient(prix, niveaux * 100)        # variation pour 1 % de spot

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(niveaux * 100, prix, color="#1F3864", lw=2, label="Prix (% du nominal)")
    ax1.set_xlabel("Niveau du spot (% du spot initial)")
    ax1.set_ylabel("Prix (% du nominal)", color="#1F3864")
    ax1.axvline(produit["barriere_protection"] * 100, color="grey", ls=":", lw=1)
    ax1.axvline(produit["barriere_autocall"] * 100, color="grey", ls=":", lw=1)

    ax2 = ax1.twinx()
    ax2.plot(niveaux * 100, delta, color="#C00000", lw=1.5, ls="--", label="Delta (pour 1 %)")
    ax2.set_ylabel("Delta (pour 1 % de spot)", color="#C00000")

    ax1.set_title("Autocall : prix et delta selon le niveau du sous-jacent")
    ax1.grid(alpha=0.3)
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.35))
    fig.tight_layout()

    sortie = Path(__file__).with_name("profil_autocall.png")
    fig.savefig(sortie, dpi=150)
    print(f"\nGraphique enregistré : {sortie}")
    plt.show()


# ---------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    Z = tirages(PRODUIT["maturite"])
    m = MARCHE

    res = prix_autocall(PRODUIT, m["spot"], m["vol"], m["r"], m["q"], Z, details=True)
    print(f"Prix de l'autocall : {res['prix']:.2f} % du nominal "
          f"(± {1.96 * res['erreur_std']:.2f} à 95 %)")
    # Écart au pair : marge du distributeur/émetteur, hors spread de financement de l'émetteur
    print(f"Écart au pair (si vendu à 100) : {100 - res['prix']:.2f} %")
    print(f"Durée de vie moyenne : {res['duree_moyenne']:.2f} ans")
    print(f"Probabilité de perte en capital : {res['proba_perte'] * 100:.1f} %")
    print("Probabilité de remboursement à chaque date :")
    for t, p in enumerate(res["proba_rappel"], start=1):
        print(f"  année {t} : {p * 100:5.1f} %")

    print("\nGreeks (en % du nominal) :")
    for nom, valeur in greeks(PRODUIT, m, Z).items():
        print(f"  {nom:<32} {valeur:+.3f}")

    validation(m)
    tracer_profil(PRODUIT, m, Z)


if __name__ == "__main__":
    main()
