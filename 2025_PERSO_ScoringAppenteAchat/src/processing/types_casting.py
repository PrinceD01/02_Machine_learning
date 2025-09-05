import json
import pandas as pd

def cast_colmuns_types(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """
    Force les types des colonnes d'un DataFrame en fonction du fichier columns.json

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à typer
    config_path : str
        Chemin vers le fichier columns.json

    Returns
    -------
    df : pd.DataFrame
        DataFrame avec les colonnes typées
    """
    
    # Charger la config JSON
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for col_conf in config:
        col = col_conf["name"]
        col_type = col_conf["type"]
        order = col_conf["order"]

        if col not in df.columns:
            continue

        if col_type == "category":
            df[col] = df[col].astype("category")

        elif col_type == "pd.Categorical":
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)

        elif col_type == "bool":
            df[col] = df[col].astype(bool)

        elif col_type == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif col_type == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df
