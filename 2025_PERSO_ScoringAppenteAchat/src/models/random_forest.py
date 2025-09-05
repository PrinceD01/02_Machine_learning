import sys
import os

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Import du fichier d'évaluation des modèles
from evaluate_models import evaluate_model
from scores_grid import get_scores_grid


def apply_random_forest(data_train: pd.DataFrame,
                                        data_val: pd.DataFrame,
                                        data_test: pd.DataFrame,
                                        dict_encode_mapping: dict = None,
                                        target: str = 'FL_ACHAT_PRODUIT',
                                        threshold_decision: float = 0.5,
                                        beta: float = 1.0,
                                        lift_prct: int = 10) -> tuple:
    """
    Fonction pour appliquer un Random Forest avec GridSearch sur les données d'entraînement, 
    de validation et de test. Optimisation sur le scoring roc_auc.
    """

    print('---')
    print('\t Lancement de la modélisation Random Forest')

    model_name = 'Random Forest'
    
    # Séparation des features et de la cible
    X_train, y_train = data_train.drop(columns=[target]), data_train[target]
    X_val, y_val = data_val.drop(columns=[target]), data_val[target]
    X_test, y_test = data_test.drop(columns=[target]), data_test[target]

    # Définir la grille des hyperparamètres
    # {'class_weight': 'balanced', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 300}
    param_grid = {
        'n_estimators': [300], #[100, 300, 500],
        'max_depth': [5, 10], #[5, 10, 20, None],
        'min_samples_split': [2], #[2, 5, 10],
        'min_samples_leaf': [5], #[1, 3, 5],
        'max_features': ['sqrt'], #['sqrt', 'log2'],
        'class_weight': ['balanced'] #[None, 'balanced']  
    }
    
    # Initialisation du modèle
    rf = RandomForestClassifier(random_state=241)
    
    # Initialisation du GridSearch
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               scoring='roc_auc',
                               cv=3,  # 3-fold cross-validation
                               verbose=0,
                               n_jobs=-1)

    # Entraînement
    grid_search.fit(X_train, y_train)


    # Meilleur modèle
    best_model = grid_search.best_estimator_
    score_grid_df = get_scores_grid(best_model, X_train, dict_encode_mapping=dict_encode_mapping)

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

    
    print("\t Modélisation Random Forest terminée.")
    print('---')

    # Sauvegarde du modèle
    return model_name, RandomForestClassifier(**grid_search.best_params_, random_state=241), score_grid_df
