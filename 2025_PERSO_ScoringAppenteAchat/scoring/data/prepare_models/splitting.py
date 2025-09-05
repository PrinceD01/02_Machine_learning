# Setup environnement
## Importation des bibliothèques
import pandas as pd
import numpy as np

from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split


def apply_splitting(df: pd.DataFrame,
                   strate: str='CLASSE_PMG_N') -> pd.DataFrame:

    print('---')
    print('    Lancement du split des données.')
    
    # Split train/val/test datasets
    data_train, dt = train_test_split(df, test_size=0.3, stratify=df[strate])
    data_val, data_test = train_test_split(dt, test_size=0.5, stratify=dt[strate])


    print('    Split des données terminé :')
    print('        - Dataset train | Nombre de lignes : ', data_train.shape[0])
    print('        - Dataset val   | Nombre de lignes : ', data_val.shape[0])
    print('        - Dataset test  | Nombre de lignes : ', data_test.shape[0])
    print('---')
    
    return data_train, data_val, data_test









