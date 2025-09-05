import sys
import os

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# Import du fichier d'évaluation des modèles
from evaluate_models import evaluate_model
from scores_grid import get_scores_grid

def apply_logistic_regression(data_train: pd.DataFrame,
                                data_val: pd.DataFrame,
                                data_test: pd.DataFrame,
                                dict_encode_mapping: dict = None,
                                target: str = 'FL_ACHAT_PRODUIT',
                                threshold_decision: float = 0.5,
                                beta: float = 1.0,
                                lift_prct: int = 10
                            ) -> tuple:
    """
    Fonction pour appliquer une Logistic Regression avec GridSearch sur les données d'entraînement, de validation et de test.
    """
    
    print('---')
    print('\t Lancement de la modélisation Logistic Regression')
    
    model_name = 'Logistic Regression'
    
    # Définir X et y
    X_train, y_train = data_train.drop(columns=[target]), data_train[target]
    X_val, y_val = data_val.drop(columns=[target]), data_val[target]
    X_test, y_test = data_test.drop(columns=[target]), data_test[target]
    
    # Grid des hyperparamètres à tester
    param_grid = {
        'C': [0.01, 0.1, 1, 10], # Coefficient de régularisation - il faut le tester pour éviter l'overfitting
        'penalty': ['l1', 'l2'], # Type de pénalité
        'solver': ['lbfgs', 'liblinear', 'saga'], # Algorithmes de résolution
        'class_weight': [None, 'balanced'], # Gestion du déséquilibre
        'max_iter': [50, 100, 250, 500, 1000]         # Nombre d'itérations pour la convergence
    }
    
    # Initialisation du modèle
    lr = LogisticRegression(random_state=241)

    # Initialisation du GridSearch
    grid_search = GridSearchCV(estimator=lr,
                               param_grid=param_grid,
                               scoring='roc_auc',  # On optimise l'AUC ROC dans un cadre de scoring
                               cv=3,
                               n_jobs=-1)

    # Entraînement
    grid_search.fit(X_train, y_train)

    
    # Meilleur modèle
    best_model = grid_search.best_estimator_
    score_grid_df = get_scores_grid(best_model, X_train, dict_encode_mapping) # Détermination de la grille de scores

    print("\t\t > Meilleurs paramètres trouvés :")
    print("\t\t\t", grid_search.best_params_)

    # Prédictions sur test
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Évaluation du modèle
    evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        score_grid=score_grid_df,
        model_name=model_name,
        threshold_decision=threshold_decision,
        beta=beta,
        lift_prct=lift_prct
    )
    
    
    print("\t Modélisation Logistic Regression terminée.")
    print('---')
    
    # Sauvegarde du modèle
    return model_name, LogisticRegression(**grid_search.best_params_, random_state=241), score_grid_df
