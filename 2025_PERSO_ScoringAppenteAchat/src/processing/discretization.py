import pandas as pd
import numpy as np
import scorecardpy as sc

def apply_discretisation_train(dt_train: pd.DataFrame, config: list):
    """
    Applique une discrétisation (binning) sur les colonnes numériques d'un DataFrame train
    et retourne à la fois le DataFrame discrétisé et les bornes à réutiliser pour le test.
    """

    df = dt_train.copy()

    # Identifier la variable cible dans le config
    cible = None
    for col_conf in config:
        if col_conf.get("label") == "Cible":
            cible = col_conf["name"]
            break

    if cible is None or cible not in df.columns:
        raise ValueError("Aucune variable cible trouvée dans le config ou absente de la dataframe")

    binning_dict = {}

    # Boucle sur chaque variable définie dans le JSON
    for col_conf in config:
        col = col_conf["name"]
        col_type = col_conf["type"]

        if col == cible:
            continue  # ne pas discrétiser la cible

        if col not in df.columns:
            continue

        # Numérique (robuste : int, float, etc.) mais pas bool
        if np.issubdtype(df[col].dtype, np.number) and df[col].dtype != bool:
            series = df[col]

            try:
                bins = sc.woebin(df[[col, cible]], y=cible, x=[col], stop_limit=0.05, bin_num=4, method="chimerge")
                binning = bins[col]

                # Bornes
                cuts = [-np.inf] + [float(v) for v in binning['breaks'].dropna()] + [np.inf]
                cuts = [round(c, 3) if np.isfinite(c) else c for c in cuts]

                # Labels
                labels = []
                for i in range(len(cuts) - 1):
                    left, right = cuts[i], cuts[i+1]
                    if left == -np.inf:
                        label = f"-inf; {right}"
                    elif right == np.inf:
                        label = f"{left}; +inf"
                    else:
                        label = f"{left}; {right}"
                    labels.append(label)

                # Application
                df[col] = pd.cut(series, bins=cuts, labels=labels, ordered=True)
                df[col] = df[col].astype("category")

                # Sauvegarde des bornes pour la phase test
                binning_dict[col] = {"cuts": cuts, "labels": labels}

            except Exception as e:
                print(f"Erreur sur la variable {col} : {e}")

    return df, binning_dict


def apply_discretisation_test(dt_test: pd.DataFrame, config: list, binning_dict: dict):
    """
    Applique une discrétisation au test selon les bornes apprises sur le train.
    """

    df = dt_test.copy()

    # Identifier la cible
    cible = None
    for col_conf in config:
        if col_conf.get("label") == "Cible":
            cible = col_conf["name"]
            break

    if cible is None or cible not in df.columns:
        raise ValueError("Aucune variable cible trouvée dans le config ou absente de la dataframe")

    # Appliquer les bins sauvegardés
    for col, bins_info in binning_dict.items():
        if col in df.columns:
            cuts = bins_info["cuts"]
            labels = bins_info["labels"]

            df[col] = pd.cut(df[col], bins=cuts, labels=labels, ordered=True)
            df[col] = df[col].astype("category")

    return df
