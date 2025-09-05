# Importation des bibliothèques
import shap
import pandas as pd
import numpy as np


def get_scores_grid(model, X_train: pd.DataFrame) -> pd.DataFrame:
    
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


    # Récupérer la df des contributions moyennes SHAP
    df_shap = apply_shap_explainer(model, X_train)
    
    df_result = []
    for var in df_shap['Variable'].unique():
        df_var = df_shap[df_shap['Variable'] == var].copy()
        
        # Décalage pour rendre positives les contributions (minimum à 0)
        min_val = df_var['shap_mean'].min()
        df_var['Contribution_decalee'] = df_var['shap_mean'] - min_val
        
        df_result.append(df_var)
    
    df_score = pd.concat(df_result)
    total = df_score['Contribution_decalee'].sum()
    
    # Normalisation pour que la somme totale soit 100
    df_score['Poids'] = (df_score['Contribution_decalee'] / total) * 100
    df_score['Poids'] = df_score['Poids'].round(2)
    
    # Renommer shap_mean en Contribution
    df_score.rename(columns={'shap_mean': 'Contribution'}, inplace=True)
    
    # Sélection et tri final
    df_score = df_score[['Variable', 'Modalité', 'Contribution', 'Poids']]
    df_score = df_score.sort_values(by=["Variable", "Modalité"]).reset_index(drop=True)
    
    return df_score

