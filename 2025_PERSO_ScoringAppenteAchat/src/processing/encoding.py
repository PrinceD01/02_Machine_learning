from typing import Any
from typing import List
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def encode_nominal(df: pd.DataFrame, nominal_cols: List[str]) -> pd.DataFrame:
    """
    Encodage nominal par OneHotEncoder.
    """
    df[nominal_cols] = df[nominal_cols].astype(str)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[nominal_cols])

    encoded_cols = encoder.get_feature_names_out(nominal_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index).astype(bool)

    df = df.drop(columns=nominal_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df

def encode_ordinal(df: pd.DataFrame, ordinal_cols: List[str], ordering_dict: dict) -> tuple[pd.DataFrame, dict]:
    """
    Encodage ordinal en respectant l'ordre donné.
    """
    categories = []
    for col in ordinal_cols:
        seen = set()
        ordered_unique = [x for x in ordering_dict[col] if not (x in seen or seen.add(x))]
        categories.append(ordered_unique)

    encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
    encoded = encoder.fit_transform(df[ordinal_cols])

    encoded_df = pd.DataFrame(encoded, columns=ordinal_cols, index=df.index).astype(int)

    dict_mapping = {}
    for i, col in enumerate(ordinal_cols):
        mapping = dict(zip(categories[i], range(len(categories[i]))))
        mapping["<inconnu>"] = -1
        dict_mapping[col] = mapping

    df = df.drop(columns=ordinal_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df, dict_mapping


def apply_encoding_from_json(df: pd.DataFrame, json_columns: List[dict]) -> tuple[pd.DataFrame, dict]:
    """
    Applique automatiquement les encodages nominal/ordinal selon le JSON.
    """
    nominal_cols = []
    ordinal_cols = []
    ordering_dict = {}

    for col_info in json_columns:
        col_name = col_info["name"]
        col_type = col_info["type"]

        # Nominales : "category"
        if col_type == "category" and col_name != "ID":
            nominal_cols.append(col_name)

        # Ordinales : "pd.Categorical" avec une clé "order"
        elif col_type == "pd.Categorical":
            ordinal_cols.append(col_name)
            ordering_dict[col_name] = col_info["order"]

    dict_mapping = {}

    if nominal_cols:
        df = encode_nominal(df, nominal_cols=nominal_cols)
    if ordinal_cols and ordering_dict:
        df, dict_mapping = encode_ordinal(df, ordinal_cols, ordering_dict)

    return df, dict_mapping
