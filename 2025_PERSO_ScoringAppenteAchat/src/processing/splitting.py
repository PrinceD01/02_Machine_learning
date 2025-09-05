
from typing import Tuple

def apply_splitting(df: pd.DataFrame,
                   target: str='FL_ACHAT_PRODUIT') -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Split train/val/test datasets
    # Création des datasets 
    data_train, data_test = train_test_split(df, test_size=0.3, stratify=df[target], random_state=241)

    return data_train, data_test

