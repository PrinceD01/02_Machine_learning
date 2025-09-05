# Setup environnement
## Importation des bibliothèques
import pandas as pd
import numpy as np

from datetime import datetime
from dateutil.relativedelta import relativedelta

from collections import Counter

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids


# Fonction de sous-échantillonnage
def apply_under_sampling(df: pd.DataFrame,
                         strate: str = 'FL_ACHAT_PRODUIT',
                         sampling_seuil_min: float = 0.2) -> pd.DataFrame:
    """
    Fonction de sous-échantillonnage : réduit la classe majoritaire pour qu'elle représente (1 - sampling_seuil_min) du total.

    Exemple : si sampling_seuil_min = 0.2 => la classe majoritaire sera ramenée à 80% max du total.

    Paramètres :
        df : pd.DataFrame
        strate : str - Nom de la variable cible
        sampling_seuil_min : float - Seuil de représentativité cible pour les minorités (donc 1 - sampling_seuil_min pour la majorité)

    Retour :
        df_undersampled : pd.DataFrame - Données sous-échantillonnées
    """
    # Séparation X / y
    y = df[strate]
    X = df.drop(columns=[strate])
    total = len(df)

    # Comptage des classes
    strate_counts = y.value_counts()
    majoritaire = strate_counts.idxmax()
    effectif_majo = strate_counts[majoritaire]
    minoritaire = strate_counts.idxmin()
    effectif_mino = strate_counts[minoritaire]
    

    # Définition des effectifs cibles pour chaque classe majoritaire
        # soit max à (1 - sampling_seuil_min) du total
    effectif_majo_cible = int(effectif_mino * (1-sampling_seuil_min) / sampling_seuil_min)

    # Vérification des effectifs cibles
    if effectif_majo <= effectif_majo_cible:
        print(f'\t\t > L\'effectif de la classe majoritaire est déjà inférieur au seuil ciblé ({(1 - sampling_seuil_min)*100}%).')
        return df.copy()

    # Application de ClusterCentroids
    cluster = ClusterCentroids(sampling_strategy={majoritaire: effectif_majo_cible}, random_state=241)
    X_res, y_res = cluster.fit_resample(X, y)

    # Reconstruction DataFrame final
    df_undersampled = pd.concat([
        pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True),
        pd.Series(y_res, name=strate).reset_index(drop=True)
    ], axis=1)

    return df_undersampled



# Foncton de sur-échantillonnage 
def apply_over_sampling(df: pd.DataFrame,
                        strate: str = 'FL_ACHAT_PRODUIT',
                        sampling_seuil_min: float = 0.2) -> pd.DataFrame:
    """
    Sur-échantillonnage des classes minoritaires pour qu'elles atteignent une certaine représentativité (ex: 20%).

    Paramètres :
        df : DataFrame d'entrée
        strate : nom de la variable cible
        sampling_seuil_min : sampling_seuil_min minimal de représentativité souhaité (ex: 0.2 = 20%)

    Retour :
        df_oversampled : DataFrame équilibrée avec BorderlineSMOTE
    """
    # Séparation X / y
    y = df[strate]
    X = df.drop(columns=[strate])
    total = len(df)

    # Identification de la classe sous-représentée
    strate_counts = y.value_counts()
    proportions = strate_counts / total
    classe_mino = proportions[proportions < sampling_seuil_min]

    # Vérification des effectifs
    if classe_mino.empty:
        print(f'\t\t > Aucune classe sous-représentée au seuil de {sampling_seuil_min*100}% détectée.')
        return df.copy()

    # Définition des effectifs cibles pour chaque classe minoritaire
        # soit max à sampling_seuil_min du total
    effectif_cible = int(total * sampling_seuil_min)
    sampling_strategy = {
        cls: max(strate_counts[cls], effectif_cible) for cls in classe_mino.index
    }

    # Vérification des effectifs cibles
    if any(strate_counts[cls] >= effectif_cible for cls in classe_mino.index):
        print(f'\t\t > L\'effectif de la classe minoritaire a atteint le seuil ciblé ({sampling_seuil_min*100}%).')
        return df.copy()

    # Sauvegarde des types d'origine
    initial_types = X.dtypes.to_dict()

    # Application de BorderlineSMOTE
    smote = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=5,
        m_neighbors=11,
        kind='borderline-1',
        random_state=241
    )
    X_res, y_res = smote.fit_resample(X, y)

    # Reconstruction du DataFrame final avec typage d'origine
    X_res = pd.DataFrame(X_res, columns=X.columns)
    for col, dtype in initial_types.items():
        if np.issubdtype(dtype, np.integer):
            X_res[col] = X_res[col].round().astype(dtype)
        elif np.issubdtype(dtype, np.bool_):
            X_res[col] = X_res[col].astype(bool)
        else:
            X_res[col] = X_res[col].astype(dtype)

    df_oversampled = pd.concat([
        X_res.reset_index(drop=True),
        pd.Series(y_res, name=strate).reset_index(drop=True)
    ], axis=1)

    return df_oversampled



# Fonction de ré-échantillonnage
def apply_sampling(df: pd.DataFrame,
                   strate: str = 'FL_ACHAT_PRODUIT',
                   sampling_seuil_min: float = 0.2,
                   sampling_mode: str = 'UNDER') -> pd.DataFrame:
    """
    Fonction flexible de ré-échantillonnage des classes :
    - 'UNDER' : applique uniquement ClusterCentroids sur la classe majoritaire
    - 'OVER'  : applique uniquement Borderline-SMOTE sur les classes minoritaires

    Paramètres :
        df : pd.DataFrame - Données d'entrée
        strate : str - Colonne cible
        sampling_seuil_min : float - Seuil de représentativité des minorités (ex: 0.2 -> max 0.8 pour la majorité)
        sampling_mode : str - 'OVER', 'UNDER'

    Retour :
        df_sampled : pd.DataFrame - Données ré-échantillonnées
    """
    
    if sampling_mode == 'OVER':
        df_sampled = apply_over_sampling(df, strate=strate, sampling_seuil_min=sampling_seuil_min)
    elif sampling_mode == 'UNDER':
        df_sampled = apply_under_sampling(df, strate=strate, sampling_seuil_min=sampling_seuil_min)
    else:
        df_sampled = df.copy()

        
    return df_sampled


