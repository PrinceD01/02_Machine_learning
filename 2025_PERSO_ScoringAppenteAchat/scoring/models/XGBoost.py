import sys
import os

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Import du fichier d'évaluation des modèles
sys.path.append(os.path.abspath(path="C:/Users/prince.mezuirotimi/OneDrive - Gedeon - SIPAOF/Documents/INTRASIPA/PROJETS/ADDITI - SCORING APPETENCE PRODUIT/03_models"))
from evaluate_models import evaluate_model
from scores_grid import get_scores_grid


def apply_xgboost(data_train: pd.DataFrame,
                  data_val: pd.DataFrame,
                  data_test: pd.DataFrame,
                  target: str = 'FL_ACHAT_PRODUIT',
                  threshold_decision: float = 0.5,
                  beta: float = 1.0,
                  lift_prct: int = 10):
    """
    Fonction pour appliquer un modèle XGBoost avec GridSearch sur les données d'entraînement, 
    de validation et de test. Optimisation sur le scoring roc_auc.
    """

    print('---')
    print('\t Lancement de la modélisation XGBoost')

    model_name = 'XGBoost'

    # Séparation des features et de la cible
    X_train, y_train = data_train.drop(columns=[target]), data_train[target]
    X_val, y_val = data_val.drop(columns=[target]), data_val[target]
    X_test, y_test = data_test.drop(columns=[target]), data_test[target]

    # Définition de la grille d'hyperparamètres
    eff_negatif = y_train.value_counts().get(1, 0)
    eff_positif = y_train.value_counts().get(0, 1)
    ratio = eff_negatif / eff_positif if eff_positif > 0 else 1
    
    scale_weights = [1, 3, 5]
    if ratio != 1 and ratio not in scale_weights:
        scale_weights.append(ratio)
        
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 8],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0], # Proportion d’échantillons utilisée pour chaque arbre. Sert à éviter l’overfitting (bagging partiel)
        'colsample_bytree': [0.8, 1.0], # Proportion de variables utilisées pour chaque arbre. Réduit la corrélation entre arbres, améliore la généralisation
        'scale_pos_weight': scale_weights  # Poids donné à la classe minoritaire (positif). Utile si déséquilibre classe
    }

    # Initialisation du modèle
    xgb = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',  # éviter warning
        random_state=241,
        verbosity=0
    )

    # Initialisation du GridSearch
    grid_search = GridSearchCV(estimator=xgb,
                               param_grid=param_grid,
                               scoring='roc_auc',
                               cv=3,
                               verbose=0,
                               n_jobs=-1)

    # Entraînement
    grid_search.fit(X_train, y_train)

    # Meilleur modèle
    best_model = grid_search.best_estimator_
    score_grid_df = get_scores_grid(best_model, X_train)

    print("\t\t > Meilleurs paramètres trouvés :")
    print("\t\t\t", grid_search.best_params_)

    # Prédiction sur le test
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
    
    
    print("\t Modélisation XGBoost terminée.")
    print('---')
    
    # Sauvegarde du modèle
    return model_name, XGBClassifier(**grid_search.best_params_, random_state=241), score_grid_df
