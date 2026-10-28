"""
OUTILS COMMUNS AUX MODÈLES DE VOLATILITÉ
========================================

Contient :
    - Black-Scholes (prix, vega, vol implicite) ;
    - la lecture de la surface de vol implicite depuis Excel ;
    - une surface fictive (SSVI) pour le mode --demo.


FICHIER EXCEL À CRÉER POUR UTILISER LES MODÈLES
-----------------------------------------------
Nom du fichier : surface_vol_implicite.xlsx
Emplacement    : le dossier indiqué dans DOSSIER_DONNEES ci-dessous

Onglet "Parametres"  (valeurs en colonne B)
    B2 : sous-jacent                 (ex. SX5E Index)
    B3 : date de valorisation        (ex. 28/10/2026)
    B4 : spot                        (ex. 5500)
    B5 : taux sans risque annuel     (ex. 0,025 pour 2,5 %)
    B6 : taux de dividende annuel    (ex. 0,03 pour 3 %)

Onglet "Vol_Implicite"  (une grille)
    Ligne 1, à partir de B1 : moneyness K / Spot  (ex. 0,80  0,90  1,00  1,10  1,20)
    Colonne A, à partir de A2 : maturités en années (ex. 0,25  0,5  1  2)
    Cellules de la grille : vol implicite (0,225 ou 22,5 : les deux sont acceptés)

Conseils :
    - au moins 4 maturités et 7 strikes, sans case vide ;
    - Bloomberg : OVDV sur l'indice, exporter la grille en moneyness.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# PARAMÈTRES À MODIFIER
# ---------------------------------------------------------------------------

# Dossier où se trouve le fichier Excel : remplacer les XXXX par ton chemin
DOSSIER_DONNEES = r"XXXXXXXXXXXXXXXXXXXXXXXX"
NOM_FICHIER = "surface_vol_implicite.xlsx"


# ---------------------------------------------------------------------------
# BLACK-SCHOLES
# ---------------------------------------------------------------------------

def bs_prix(S, K, T, r, q, vol, call=True):
    """Prix Black-Scholes d'une option européenne, avec dividende continu q."""
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, vol):
    """Vega : variation du prix pour +1 (soit +100 %) de vol."""
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def vol_implicite(prix, S, K, T, r, q, call=True):
    """Vol qui redonne le prix observé (recherche de racine entre 0,01 % et 500 %)."""
    try:
        return brentq(lambda v: bs_prix(S, K, T, r, q, v, call) - prix, 1e-4, 5.0)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# LECTURE DE LA SURFACE
# ---------------------------------------------------------------------------

def lire_surface(chemin_fichier):
    """
    Renvoie :
        parametres : dict (sous_jacent, date_valo, spot, r, q)
        maturites  : vecteur des maturités en années           (taille nT)
        moneyness  : vecteur des K / Spot                      (taille nK)
        vols       : matrice des vols implicites, en décimal   (nT x nK)
    """
    chemin = Path(chemin_fichier)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}\n"
            "Vérifie DOSSIER_DONNEES dans commun.py, ou lance avec --demo."
        )

    p = pd.read_excel(chemin, sheet_name="Parametres", header=None)
    parametres = {
        "sous_jacent": str(p.iloc[1, 1]),
        "date_valo": pd.to_datetime(p.iloc[2, 1]).date(),
        "spot": float(p.iloc[3, 1]),
        "r": float(p.iloc[4, 1]),
        "q": float(p.iloc[5, 1]),
    }

    grille = pd.read_excel(chemin, sheet_name="Vol_Implicite", header=None)
    moneyness = grille.iloc[0, 1:].astype(float).to_numpy()
    maturites = grille.iloc[1:, 0].astype(float).to_numpy()
    vols = grille.iloc[1:, 1:].astype(float).to_numpy()

    # Tolérance sur les unités : 22,5 -> 0,225 et 90 -> 0,90
    if np.nanmax(vols) > 3:
        vols = vols / 100
    if np.nanmax(moneyness) > 10:
        moneyness = moneyness / 100

    if np.isnan(vols).any():
        raise ValueError("La grille de vol contient des cases vides.")

    ordre = np.argsort(maturites)
    return parametres, maturites[ordre], moneyness, vols[ordre]


# ---------------------------------------------------------------------------
# SURFACE FICTIVE (mode --demo)
# ---------------------------------------------------------------------------

def surface_demo():
    """
    Surface SSVI (Gatheral & Jacquier) : paramétrage sans arbitrage
    qui reproduit un skew actions réaliste (vol plus haute sur les puts).

        w(y, T) = theta_T / 2 x [1 + rho phi y + sqrt((phi y + rho)^2 + 1 - rho^2)]
        y = ln(K / F_T), theta_T = variance totale ATM, phi = eta / sqrt(theta_T)
    """
    parametres = {"sous_jacent": "INDICE FICTIF", "date_valo": date(2026, 10, 28),
                  "spot": 5500.0, "r": 0.025, "q": 0.03}
    maturites = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    moneyness = np.round(np.arange(0.70, 1.301, 0.05), 2)

    vol_atm, rho, eta = 0.18, -0.6, 1.0
    vols = np.zeros((len(maturites), len(moneyness)))
    for i, T in enumerate(maturites):
        theta = vol_atm ** 2 * T
        phi = eta / np.sqrt(theta)
        y = np.log(moneyness) - (parametres["r"] - parametres["q"]) * T
        w = theta / 2 * (1 + rho * phi * y + np.sqrt((phi * y + rho) ** 2 + 1 - rho ** 2))
        vols[i] = np.sqrt(w / T)

    return parametres, maturites, moneyness, vols


def charger(demo):
    """Charge la surface : fictive si demo, sinon depuis le fichier Excel."""
    if demo:
        print("Mode démo : surface fictive (SSVI).")
        return surface_demo()
    chemin = Path(DOSSIER_DONNEES) / NOM_FICHIER
    print(f"Lecture de {chemin}")
    return lire_surface(chemin)
