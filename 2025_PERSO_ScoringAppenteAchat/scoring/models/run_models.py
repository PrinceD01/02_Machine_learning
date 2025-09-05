# Importation des bibliothèques
import sys
import os
import warnings

import numpy as np
import pandas as pd
import xgboost 


# Import du connecteur
sys.path.append(os.path.abspath("C:/Users/prince.mezuirotimi/OneDrive - Gedeon - SIPAOF/Documents/INTRASIPA/PROJETS/_setup"))
import environnement as env

# Import des fichiers de preparation des données à la modelisation
sys.path.append(os.path.abspath("C:/Users/prince.mezuirotimi/OneDrive - Gedeon - SIPAOF/Documents/INTRASIPA/PROJETS/ADDITI - SCORING APPETENCE PRODUIT/03_models"))
from logistic_regression import apply_logistic_regression
from random_forest import apply_random_forest
from XGBoost import apply_xgboost

def run_models(data_train: pd.DataFrame,
              data_val: pd.DataFrame,
              data_test: pd.DataFrame,
              target: str = 'FL_ACHAT_PRODUIT',
              model_selected: list|str = 'all',
              threshold_decision: float = 0.5,
              beta: float = 1.0,
              lift_prct: int = 10)  -> dict:
    """
    Fonction pour exécuter les modèles de régression logistique et Random Forest.
    
    Paramètres :
        data_train : pd.DataFrame
            Jeu de données d'entraînement.
        data_val : pd.DataFrame
            Jeu de données de validation.
        data_test : pd.DataFrame
            Jeu de données de test.
        target : str
            Nom de la colonne cible.
    
    Retour :
        None
    """
    models = {}
    
    if model_selected == 'all' or 'logistic_regression' in model_selected:   
        # Exécution du modèle de régression logistique
        model_name, mdl, scores_grid = apply_logistic_regression(data_train=data_train, data_val=data_val, data_test=data_test, target=target, threshold_decision=threshold_decision, beta=beta, lift_prct=lift_prct)
        models[model_name] = [model_name, mdl, scores_grid]

    if model_selected == 'all' or 'random_forest' in model_selected:
        # Exécution du modèle Random Forest
        model_name, mdl, scores_grid = apply_random_forest(data_train=data_train, data_val=data_val, data_test=data_test, target=target, threshold_decision=threshold_decision, beta=beta, lift_prct=lift_prct)
        models[model_name] = [model_name, mdl, scores_grid]

    if model_selected == 'all' or 'xgboost' in model_selected:
        # Exécution du modèle XGBoost
        model_name, mdl, scores_grid = apply_xgboost(data_train=data_train, data_val=data_val, data_test=data_test, target=target, threshold_decision=threshold_decision, beta=beta, lift_prct=lift_prct)
        models[model_name] = [model_name, mdl, scores_grid]
    
    # Retour des modèles sélectionnés
    return models

