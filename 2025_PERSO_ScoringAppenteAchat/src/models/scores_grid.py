# Importation des bibliothèques

import shap
import pandas as pd
import numpy as np



def get_scores_grid(model,
                    X_train: pd.DataFrame,
                    dict_encode_mapping: dict = None
                ) -> pd.DataFrame:



    def decode_modalities(df_score: pd.DataFrame, dict_mapping: dict) -> pd.DataFrame:
        if not dict_mapping:  # None ou {}
            return df_score.copy()

        df = df_score.copy()

        for var in df['Variable'].unique():
            mask = df['Variable'] == var

            if var in dict_mapping:
                # Décodage via le mapping
                mapping = dict_mapping[var]
                inverse_map = {v: k for k, v in mapping.items() if v != -1}
                df.loc[mask, 'Modalité'] = df.loc[mask, 'Modalité'].apply(
                    lambda x: inverse_map.get(int(x), "<inconnu>")
                )

            else:
                # Si c'est un bool encodé (0.0 / 1.0)
                valeurs_uniques = set(df.loc[mask, 'Modalité'])
                if valeurs_uniques <= {0.0, 1.0}:
                    df.loc[mask, 'Modalité'] = df.loc[mask, 'Modalité'].map({0.0: "Faux", 1.0: "Vrai"})

        return df


    
    def apply_shap_explainer(model, X_train: pd.DataFrame) -> pd.DataFrame:
        # Détection modèle d'arbre
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train)

        shap_values = explainer(X_train)

        # Gestion classification binaire (3D)
        if len(shap_values.values.shape) == 3:
            shap_vals_2d = shap_values.values[:, :, 1]  # classe positive
        else:
            shap_vals_2d = shap_values.values

        shap_df = pd.DataFrame(shap_vals_2d, columns=X_train.columns)

        results = []
        for col in X_train.columns:
            for modality in sorted(X_train[col].unique()):
                mask = X_train[col] == modality
                mean_val = shap_df.loc[mask, col].mean()
                results.append({
                    "Variable": col,
                    "Modalité": modality,
                    "shap_mean": mean_val
                })

        return pd.DataFrame(results).sort_values(["Variable", "Modalité"])


    # On force le typage numérique si nécessaire
    X_train_shap = X_train.copy()

    for col in X_train_shap.columns:
        # Convertir les booléens en float
        if X_train_shap[col].dtype == bool:
            X_train_shap[col] = X_train_shap[col].astype(float)
        else:
            # Convertir les autres colonnes (object ou int, float) en float
            X_train_shap[col] = pd.to_numeric(X_train_shap[col], errors='raise').astype(float)
    
    # Récupérer la df des contributions moyennes SHAP
    df_shap = apply_shap_explainer(model, X_train_shap)
    
    # Calcul importance par variable (somme des SHAP absolus)
    importance = df_shap.groupby('Variable')['shap_mean'].apply(lambda x: np.sum(np.abs(x))).reset_index()
    importance.columns = ['Variable', 'Importance']
    importance['Importance_norm'] = 100 * importance['Importance'] / importance['Importance'].sum()

    # Normalisation interne à chaque variable (min=0, max=1)
    df_shap['shap_shifted'] = df_shap.groupby('Variable')['shap_mean'].transform(lambda x: x - x.min())
    df_shap['shap_norm'] = df_shap.groupby('Variable')['shap_shifted'].transform(
        lambda x: 0 if x.max() == 0 else x / x.max()
    )

    # Pondération par importance
    df_score = df_shap.merge(importance[['Variable', 'Importance_norm']], on='Variable', how='left')
    df_score['Poids'] = df_score['shap_norm'] * df_score['Importance_norm']

    
    # Sélection et tri final
    df_score.rename(columns={'shap_mean': 'Contribution'}, inplace=True)
    df_score = df_score[['Variable', 'Modalité', 'Contribution', 'Poids']]
    df_score = df_score.sort_values(by=["Variable", "Modalité"]).reset_index(drop=True)
    
    
    if dict_encode_mapping is not None:
        df_score = decode_modalities(df_score, dict_encode_mapping)
    
    
    return df_score

