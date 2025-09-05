from turtle import pd


def prepare_cohort_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction de prétraitement des données brutes avant modélisation ou analyse.

    Étapes principales :
        1. Définition d'une fonction interne pour lister les valeurs manquantes
        2. Typage des variables catégorielles
        3. Traitement et imputation des valeurs manquantes :
            - Variables d'achat
            - Variables d'opportunités
            - Dates de facturation, commande, parution, etc.
            - Variables de blocage, arrêt, procédure collective
            - Variables liées au groupement et au département
            - Variables système (dates de création, MAJ CreditSafe)
        4. Ajout de variables indicatrices (flags) pour indiquer la présence de données manquantes
        5. Suppression des variables contenant plus de 25 % de valeurs manquantes
        6. Préparation finale du jeu de données

    Paramètres :
        df : pd.DataFrame
            Jeu de données d’entrée à prétraiter.

    Retour :
        pd.DataFrame : DataFrame nettoyée et enrichie, prête pour l’analyse ou la modélisation.
    """
    
    def listing_nan_var(df:pd.DataFrame):
        """
        Fonction permettant d'analyser la présence de valeurs manquantes dans un DataFrame

        Paramètres :
            df : pd.DataFrame
                Jeu de données à analyser
            verbose : bool (par défaut = True)
                Si True, affiche un graphique des pourcentages de NaN par variable

        Retour :
            missing_df : pd.DataFrame
                Tableau contenant :
                    - le nom des variables
                    - le pourcentage de valeurs manquantes
                    - le nombre de valeurs manquantes
                Trié par ordre décroissant du pourcentage de NaN
        """
        missing_values = df.isnull().mean() * 100
        missing_df = pd.DataFrame({
            'Variable': missing_values.index,
            'Prct': missing_values.values,
            'Count': (missing_values.values * df.shape[0] / 100).astype(int)
        })
        missing_df.sort_values(by='Prct', ascending=False, inplace=True)

        return missing_df

    # Variables de référence
    DATE_IMPUTER = pd.to_datetime("1900-01-01", format="%Y-%m-%d")
    DATE_HEURE_IMPUTER = pd.to_datetime("1900-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    SEUIL_ECART_ABS = 0.15
    
    # Données
    ## Pré-traitement
    dt = df.copy()

    ### Typage des variables
    # Traitement des catégorielles
    dt[['ID_COMPTE','DEPARTEMENT', 'MOTIF_ARRET']] = dt[['ID_COMPTE','DEPARTEMENT', 'MOTIF_ARRET']].astype('object')

    ### Valeurs manquantes
    missing_df = listing_nan_var(dt)
    #### Analyse préliminaire des NA

    # Achat produit
    dt['ACHAT_PRODUIT'] = dt['FL_ACHAT_PRODUIT'].replace(
        {0: 'Achat autre', 1: 'Achat cible', np.nan: 'Aucun achat'}
    )
    dt['FL_ACHAT_PRODUIT'] = dt['FL_ACHAT_PRODUIT'].fillna(0)

    # Variables produit acheté
    dt['REG_DAF4'] = dt['REG_DAF4'].fillna(dt['LIB_DAF2'])
    dt['LIB_DAF3'] = dt['LIB_DAF3'].fillna(dt['LIB_DAF2'])

    dt[['REG_DAF4', 'GAMME_PRODUIT', 'LIB_DAF3', 'LIB_DAF2']] = \
        dt[['REG_DAF4', 'GAMME_PRODUIT', 'LIB_DAF3', 'LIB_DAF2']].fillna('Aucun achat')

    # Dates de facturation, parution et commande
    dt[['DT_FACTURATION', 'DT_PARUTION', 'DT_COMMANDE']] = \
        dt[['DT_FACTURATION', 'DT_PARUTION', 'DT_COMMANDE']].fillna(DATE_IMPUTER)

    # Opportunité
    dt['FL_OPPORTUNITE'] = dt[['DT_OPPORTUNITE','ETAPE_OPPORTUNITE','TYPE_PRODUIT_OPPORTUNITE',
                            'MONTANT_OPPORTUNITE','GAIN_OPPORTUNITE','PROBABILITE_OPPORTUNITE']]\
                            .notna().any(axis=1).astype(int)

    dt['ETAPE_OPPORTUNITE'] = dt['ETAPE_OPPORTUNITE'].fillna('Aucune opportunité')
    dt['TYPE_PRODUIT_OPPORTUNITE'] = dt['TYPE_PRODUIT_OPPORTUNITE'].fillna('Aucune opportunité')
    dt[['GAIN_OPPORTUNITE','PROBABILITE_OPPORTUNITE']] = \
        dt[['GAIN_OPPORTUNITE','PROBABILITE_OPPORTUNITE']].fillna(0)

    dt.loc[dt['DT_OPPORTUNITE'].notna() & dt['MONTANT_OPPORTUNITE'].isna(), 'MONTANT_OPPORTUNITE'] = 0
    dt['MONTANT_OPPORTUNITE'] = dt['MONTANT_OPPORTUNITE'].fillna(0)
    dt['DT_OPPORTUNITE'] = dt['DT_OPPORTUNITE'].fillna(DATE_HEURE_IMPUTER)

    # Blocage
    dt['FL_BLOGAGE'] = dt['MOTIF_BLOCAGE'].notna().astype(int)
    dt['MOTIF_BLOCAGE'] = dt['MOTIF_BLOCAGE'].fillna('Aucun blocage')
    dt['FL_HISTO_BLOGAGE'] = dt['DT_BLOCAGE'].notna().astype(int)
    dt['DT_BLOCAGE'] = dt['DT_BLOCAGE'].fillna(DATE_IMPUTER)

    # Procédure collective
    dt['FL_PROCEDURE_COLLECTIVE'] = dt['PROCEDURE_COLLECTIVE'].notna().astype(int)
    dt['PROCEDURE_COLLECTIVE'] = dt['PROCEDURE_COLLECTIVE'].fillna('Aucune procédure collective')

    # Arrêt
    dt['FL_ARRET'] = dt['DT_ARRET'].notna().astype(int)
    dt['DT_ARRET'] = dt['DT_ARRET'].fillna(DATE_IMPUTER)

    # Groupement
    dt['FL_GROUPEMENT'] = dt['GROUPEMENT'].notna().astype(int)
    dt['GROUPEMENT'] = dt['GROUPEMENT'].fillna('Pas de groupement')
    dt['STATUT_GROUPEMENT'] = dt['STATUT_GROUPEMENT'].fillna('Pas de statut groupement')

    # Département
    dt['DEPARTEMENT'] = dt['DEPARTEMENT'].fillna(dt['CODE_POSTAL'].str[:2])

    # Dates système
    mask = dt['DT_CREATION_ENTREPRISE'].isna()
    dt.loc[mask, 'DT_CREATION_ENTREPRISE'] = dt.loc[mask, 'DT_ENREGISTREMENT_COMPTE'].astype(str).str[:-9]

    dt['FL_CREDITSAFE'] = dt['DT_MAJ_CREDITSAFE'].notna().astype(int)
    dt['DT_MAJ_CREDITSAFE'] = dt['DT_MAJ_CREDITSAFE'].fillna(DATE_IMPUTER)




    # Gestion des valeurs manquantes
    ## Remplacement des variables avec plus de 25% de valeurs manquantes par un flag
    missing_df = listing_nan_var(dt)
    missing_var = missing_df[missing_df['Prct'] >= 25]['Variable'].values
    for current_var in missing_var:
        dt['FL_'+current_var] = dt[current_var].notna().astype(int)
    dt = dt.drop(missing_var, axis=1) 


    ## Imputation par valeur de référence - <8%
    missing_df = listing_nan_var(dt)
    missing_df = missing_df[(missing_df['Prct'] < 8) & (missing_df['Prct'] > 0)]

    for current_var in missing_df['Variable']:
        serie = dt[current_var]

        if serie.dtype == 'object' or serie.dtype.name == 'category':  
            # Cas quali : mode
            if not serie.dropna().empty:
                mode_val = serie.mode(dropna=True)
                if not mode_val.empty:
                    serie = serie.fillna(mode_val[0])
                else:
                    serie = serie.fillna("Valeur manquante")  # fallback de sécurité
            else:
                serie = serie.fillna("Valeur manquante")

        elif pd.api.types.is_numeric_dtype(serie):  
            # Cas quanti : moyenne/médiane
            if not serie.dropna().empty:
                median_val = serie.median()
                mean_val = serie.mean()
                ecart_abs = abs(mean_val - median_val) / abs(median_val)

                if ecart_abs > SEUIL_ECART_ABS:
                    serie = serie.fillna(median_val)
                else:
                    serie = serie.fillna(mean_val)
            else:
                serie = serie.fillna(0)  # fallback de sécurité
                
        dt[current_var] = serie


    ## Imputation par knn - <25%
    missing_df = listing_nan_var(dt)
    missing_df = missing_df[(missing_df['Prct'] < 25) & (missing_df['Prct'] >= 8)]

    for current_var in missing_df['Variable']:
        serie = dt[current_var]
        
        if serie.dtype == 'object' or serie.dtype.name == 'category': 
            new_var = current_var + '_MISSING'
            dt[new_var] = dt[current_var].isna().astype(int)
            
        elif pd.api.types.is_numeric_dtype(serie):
            new_var = 'FL_' + current_var
            dt[new_var] = dt[current_var].notna().astype(int)
            
    dt = dt.drop(columns=missing_df['Variable'].values, axis=1) 


    data_cleaned = dt.copy()
    return data_cleaned




def cast_feature_types(df: pd.DataFrame,
                        config_path: str
                        )-> pd.DataFrame:
    """
    Fonction permettant de typer automatiquement les colonnes d’un DataFrame à partir des configurations définies.

    Paramètres :
        df : pd.DataFrame
            DataFrame source à typer
        config_path : str
            Chemin vers le fichier de configuration JSON

    Retour :
        dt  : pd.DataFrame
    """

    DATE_FORMATS = [
        "%Y-%m-%d %H:%M:%S",  # Format avec heure
        "%Y-%m-%d",           # Format sans heure
        "%d/%m/%Y %H:%M",     # Autre format possible
        "%d/%m/%Y"            # Autre format possible
    ]
    
    # Charger la config JSON
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Forçage des types
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

        elif col_type == "date":
            temp_dates = pd.Series(index=df[col].index, dtype='datetime64[ns]')

            for fmt in DATE_FORMATS: # Essai des différents formats
                mask = temp_dates.isna() # Convertir seulement les valeurs non encore converties
                if not mask.any():
                    break  # Toutes les valeurs sont converties
                try:
                    temp_dates[mask] = pd.to_datetime(df.loc[mask, col], format=fmt, errors='raise')
                except:
                    continue
            
            if temp_dates.isna().any(): # On utilise la conversion coercive pour les valeurs restantes non converties
                temp_dates[temp_dates.isna()] = pd.to_datetime(df.loc[temp_dates.isna(), col], errors='coerce')
            df[col] = temp_dates

    data_casting = df.copy()
    return data_casting
