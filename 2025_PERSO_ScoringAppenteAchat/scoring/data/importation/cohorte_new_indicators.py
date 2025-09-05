# Import du fichier environnement.py
from turtle import pd
import scoring.data.connector as conn
from scoring.data.importation.requetes_importation import historical_data, customer_relational_data



def build_indicators(df:pd.DataFrame, 
                        date:datetime.date
                    ) -> pd.DataFrame:
    """
    Fonction de création de nouveaux indicateurs à partir d’une base client cleaned et d’un historique de facturation issu de la base SISPUB.

    Étapes principales :
        1. Définition des dates de référence pour l’analyse.
        2. Création et exécution d’une requête SQL pour récupérer l’historique de facturation sur 24 mois glissants, en filtrant les comptes actifs uniquement.
        3. Traitement et typage des données issues de la base.
        4. Construction des nouveaux indicateurs par client, tels que :
            - Chiffre d'affaires annuel ADDITI et Classes PMG - N & N-1
            - RFM (Récence, Fréquence, Montant)
            - Co-appétence produit (GAMME_PRODUIT) selon l'étude des comportements clients
            - Suivi des relations commerciales
            - (Actions commerciales)
            - Récences (différences entre les différentes dates et la date d’étude)
            - Saisonnalités issues l'étude des comportements clients

    Paramètres :
        df : pd.DataFrame
            Dataframe client de référence (DTM) contenant les ID des comptes pour lesquels on veut enrichir les données.
        date : datetime.date
            Date d’étude — point de départ de l’observation pour la construction des indicateurs.

    Retour :
        data_new_indicators : pd.DataFrame - Dataframe enrichi avec les nouveaux indicateurs calculés pour chaque compte client.
    """

    # Variables de référence
    CNX = conn.sispub()
    DATE_ETUDE = date.strftime("%Y-%m-%d")
    DATE_HEURE_ETUDE = date.strftime("%Y-%m-%d %H:%M:%S")
    LAG_3M = 3
    LAG_6M = 6
    LAG_9M = 9
    LAG_12M = 12
    LAG_24M = 24
    DATE_PAST_3M = (date - relativedelta(months=LAG_3M)).strftime("%Y-%m-%d")   
    DATE_PAST_6M = (date - relativedelta(months=LAG_6M)).strftime("%Y-%m-%d")   
    DATE_PAST_9M = (date - relativedelta(months=LAG_9M)).strftime("%Y-%m-%d")   
    DATE_PAST_12M = (date - relativedelta(months=LAG_12M)).strftime("%Y-%m-%d") 
    DATE_PAST_24M = (date - relativedelta(months=LAG_24M)).strftime("%Y-%m-%d") 

    MEDIAN = {'Très petit': 10, 'Petit': 6, 'Moyen': 4, 'Gros': 2, 'Très gros': 1}
    MEAN = {'Très petit': 21, 'Petit': 11, 'Moyen': 7, 'Gros': 5, 'Très gros': 3}
    STD = {'Très petit': 37, 'Petit': 18, 'Moyen': 12, 'Gros': 12, 'Très gros': 7}
    SEUIL = {'Très petit': 84, 'Petit': 41, 'Moyen': 16, 'Gros': 15, 'Très gros': 5}

    # Importation of cleaned data
    data = df.copy()

    # New indicators
    # Initialisation du DTM avec les nouveaux indicateurs
    data['ID_COMPTE'] = data['ID_COMPTE'].astype(str)
    data_new_indicators = data.copy()

    # Chargement des historiques de facturations
    query = historical_data(date)
    histo_facturation = pd.DataFrame(CNX(query))

    # on ne garde que les id clients qui sont dans le DTM
    histo_facturation = histo_facturation[histo_facturation["ID_COMPTE"].isin(values=data["ID_COMPTE"].unique())]

    # Typage des colonnes 
    histo_facturation[["ID_COMMANDE", "ID_COMPTE", "NUM_ORDRE",
                       "REG_DAF1", "LIB_DAF1", "LIB_DAF2", "LIB_DAF3", "LIB_DAF4",
                       "GAMME_PRODUIT"]
                    ] = histo_facturation[["ID_COMMANDE", "ID_COMPTE", "NUM_ORDRE",
                                            "REG_DAF1", "LIB_DAF1", "LIB_DAF2", "LIB_DAF3", "LIB_DAF4",
                                            "GAMME_PRODUIT"]].astype(str)
    histo_facturation["DT_COMMANDE"] = pd.to_datetime(
        histo_facturation["DT_COMMANDE"], format="%Y-%m-%d %H:%M:%S"
    )
    histo_facturation["MONTANT_HT"] = histo_facturation["MONTANT_HT"].astype(float)



    ## 1. CA Additi & Classes PMG
    # Filtrer les données pour l'année d'analyse (N)
    ca_annuel_analyse = (
        histo_facturation[
            (histo_facturation['DT_COMMANDE'] > DATE_PAST_12M) & 
            (histo_facturation['DT_COMMANDE'] <= DATE_ETUDE)
        ]
        .groupby('ID_COMPTE')['MONTANT_HT']
        .sum()
        .reset_index()
        .rename(columns={'MONTANT_HT': 'CA_ANNUEL_ADDITI_N'})
    )
    # Ajouter la colonne CLASSE_PMG_N
    ca_annuel_analyse['CLASSE_PMG_N'] = pd.cut(
        ca_annuel_analyse['CA_ANNUEL_ADDITI_N'],
        bins=[-float('inf'), 0, 1500, 5000, 20000, 50000, float('inf')],
        labels=['CA négatif', 'Très petit', 'Petit', 'Moyen', 'Gros', 'Très gros'],
        right=False
    )
    ca_annuel_analyse['CLASSE_PMG_N'] = ca_annuel_analyse['CLASSE_PMG_N'].cat.add_categories(['Hors catégorie'])
    ca_annuel_analyse.loc[ca_annuel_analyse['CA_ANNUEL_ADDITI_N'] == 0, 'CLASSE_PMG_N'] = 'Hors catégorie'

    # Filtrer les données pour l'année de référence (N-1)
    ca_annuel_reference = (
        histo_facturation[
            (histo_facturation['DT_COMMANDE'] > DATE_PAST_24M) & 
            (histo_facturation['DT_COMMANDE'] <= DATE_PAST_12M)
        ]
        .groupby('ID_COMPTE')['MONTANT_HT']
        .sum()
        .reset_index()
        .rename(columns={'MONTANT_HT': 'CA_ANNUEL_ADDITI_N1'})
    )
    # Ajouter la colonne CLASSE_PMG_N-1
    ca_annuel_reference['CLASSE_PMG_N1'] = pd.cut(
        ca_annuel_reference['CA_ANNUEL_ADDITI_N1'],
        bins=[-float('inf'), 0, 1500, 5000, 20000, 50000, float('inf')],
        labels=['CA négatif', 'Très petit', 'Petit', 'Moyen', 'Gros', 'Très gros'],
        right=False
    )
    ca_annuel_reference['CLASSE_PMG_N1'] = ca_annuel_reference['CLASSE_PMG_N1'].cat.add_categories(['Hors catégorie'])
    ca_annuel_reference.loc[ca_annuel_reference['CA_ANNUEL_ADDITI_N1'] == 0, 'CLASSE_PMG_N1'] = 'Hors catégorie'

    

    # Filtrer les données pour l'année d'analyse (N) et les commandes de produits digitaux
    ca_annuel_digital = (
        histo_facturation[
            (histo_facturation['DT_COMMANDE'] > DATE_PAST_12M) & 
            (histo_facturation['DT_COMMANDE'] <= DATE_ETUDE) &
            (histo_facturation['REG_DAF1'] == 'D')
        ]
        .groupby('ID_COMPTE')['MONTANT_HT']
        .sum()
        .reset_index()
        .rename(columns={'MONTANT_HT': 'CA_DIGITAL_ANNUEL_ADDITI_N'})
    )
    
    # Gestion des cas particuliers
    ca_annuel_reference.loc[ca_annuel_reference['CA_ANNUEL_ADDITI_N1'] == 0, 'CLASSE_PMG_N1'] = 'Hors catégorie'
    ca_annuel_reference.loc[ca_annuel_reference['CA_ANNUEL_ADDITI_N1'] < 0, 'CLASSE_PMG_N1'] = 'CA négatif'
    
    ca_annuel_digital.loc[ca_annuel_digital['CA_DIGITAL_ANNUEL_ADDITI_N'].isnull(), 'CA_DIGITAL_ANNUEL_ADDITI_N'] = 0

    # Fusionner les deux tables pour obtenir df_CA
    df_CA = data[['ID_COMPTE']].drop_duplicates()
    df_CA = df_CA.merge(ca_annuel_analyse, on='ID_COMPTE', how='left')
    df_CA = df_CA.merge(ca_annuel_reference, on='ID_COMPTE', how='left')
    df_CA = df_CA.merge(ca_annuel_digital, on='ID_COMPTE', how='left')

    # Mettre à jour les valeurs manquantes selon le CA 
    df_CA.loc[((df_CA['CA_ANNUEL_ADDITI_N1'] <= 0) | (df_CA['CA_ANNUEL_ADDITI_N1'].isnull())) & (df_CA['CA_ANNUEL_ADDITI_N'] > 0), 'CA_ANNUEL_ADDITI_N1'] = 0
    df_CA.loc[((df_CA['CA_ANNUEL_ADDITI_N1'] <= 0) | (df_CA['CA_ANNUEL_ADDITI_N1'].isnull())) & (df_CA['CA_ANNUEL_ADDITI_N'] > 0), 'CLASSE_PMG_N1'] = df_CA.loc[df_CA['CA_ANNUEL_ADDITI_N1'] <= 0 & (df_CA['CA_ANNUEL_ADDITI_N'] > 0), 'CLASSE_PMG_N']

    df_CA.loc[(df_CA['CA_ANNUEL_ADDITI_N'].isnull()) & (df_CA['CA_ANNUEL_ADDITI_N1'] > 0), 'CA_ANNUEL_ADDITI_N'] = 0
    df_CA.loc[(df_CA['CLASSE_PMG_N'].isnull()) & (df_CA['CA_ANNUEL_ADDITI_N1'] > 0), 'CLASSE_PMG_N'] = 'Très petit'

    df_CA.loc[(df_CA['CLASSE_PMG_N'].isnull()) & (df_CA['CLASSE_PMG_N1'].isnull()), 'CLASSE_PMG_N'] = 'Hors catégorie'
    df_CA.loc[(df_CA['CLASSE_PMG_N'].isnull()) & (df_CA['CLASSE_PMG_N1'].isnull()), 'CLASSE_PMG_N1'] = 'Hors catégorie'
    df_CA.loc[(df_CA['CA_ANNUEL_ADDITI_N'].isnull()) & (df_CA['CA_ANNUEL_ADDITI_N1'].isnull()), 'CA_ANNUEL_ADDITI_N'] = 0
    df_CA.loc[(df_CA['CA_ANNUEL_ADDITI_N'].isnull()) & (df_CA['CA_ANNUEL_ADDITI_N1'].isnull()), 'CA_ANNUEL_ADDITI_N1'] = 0
    
    # data_new_indicators[data_new_indicators['CLASSE_PMG_N'].isnull()]
    df_CA.loc[(df_CA['CA_ANNUEL_ADDITI_N'].isnull()) & (df_CA['CA_ANNUEL_ADDITI_N1'] <= 0), 'CA_ANNUEL_ADDITI_N'] = 0
    df_CA.loc[(df_CA['CLASSE_PMG_N'].isnull()) & (df_CA['CA_ANNUEL_ADDITI_N1'] <= 0), 'CLASSE_PMG_N'] = df_CA.loc[(df_CA['CLASSE_PMG_N'].isnull()) & (df_CA['CA_ANNUEL_ADDITI_N1'] <= 0), 'CLASSE_PMG_N1']

    # Calcul de la part de digitalisation
    df_CA['PART_DIGITALISATION'] = df_CA['CA_DIGITAL_ANNUEL_ADDITI_N'] / df_CA['CA_ANNUEL_ADDITI_N'].replace(0, np.nan)
    df_CA['PART_DIGITALISATION'] = df_CA['PART_DIGITALISATION'].fillna(0)  # Remplacer les NaN par 0
    

    # Ajout des indicateurs des CA
    data_new_indicators = data_new_indicators.merge(df_CA, on='ID_COMPTE', how='left')


    ## 2. RFM
    ### Récence-Fréquence
    # Calcul des métriques pour chaque ID_COMPTE
    histo_facturation['DT_COMMANDE'] = pd.to_datetime(histo_facturation['DT_COMMANDE'], errors='coerce')

    rf_metrics = (
        histo_facturation.sort_values(by=['ID_COMPTE', 'DT_COMMANDE']).groupby('ID_COMPTE')
        .agg(
            NB_ACHAT=('NUM_ORDRE', 'nunique'),
            MOY_DELAI_ACHAT=('DT_COMMANDE', lambda x: x.diff().dt.days.MEAN()),
            VAR_DELAI_ACHAT=('DT_COMMANDE', lambda x: x.diff().dt.days.var()),
            RECENCE=('DT_COMMANDE', lambda x: (pd.to_datetime(DATE_ETUDE) - x.max()).days)
        )
        .reset_index()
    )

    # Remplir les valeurs manquantes avec 0 pour MOY_DELAI_ACHAT et VAR_DELAI_ACHAT
    rf_metrics['MOY_DELAI_ACHAT'] = rf_metrics['MOY_DELAI_ACHAT'].fillna(0)
    rf_metrics['VAR_DELAI_ACHAT'] = rf_metrics['VAR_DELAI_ACHAT'].fillna(0)

    # Calcul de SQRT_DELAI_ACHAT
    rf_metrics['SQRT_DELAI_ACHAT'] = rf_metrics['VAR_DELAI_ACHAT'].apply(sqrt).replace(0, 1)

    # Calcul de RECENCE_FREQUENCE
    rf_metrics['RECENCE_FREQUENCE'] = (
        (rf_metrics['RECENCE'] - 7 - rf_metrics['MOY_DELAI_ACHAT']) / rf_metrics['SQRT_DELAI_ACHAT']
    )

    # Calcul de INDICATEUR_RF
    conditions = [
        (rf_metrics['RECENCE_FREQUENCE'] < -0.5),
        (rf_metrics['RECENCE_FREQUENCE'] >= -0.5) & (rf_metrics['RECENCE_FREQUENCE'] <= 0),
        (rf_metrics['RECENCE_FREQUENCE'] > 0) & (rf_metrics['RECENCE_FREQUENCE'] <= 1.5),
        (rf_metrics['RECENCE_FREQUENCE'] > 1.5)
    ]
    choices = ['Très en avance', 'En avance', 'En retard', 'Très en retard']
    rf_metrics['INDICATEUR_RF'] = np.select(conditions, choices, default='Non calculé')

    # Jointure avec les ID_COMPTE de la table data pour s'assurer que tous les comptes sont inclus
    rf_metrics['ID_COMPTE'] = rf_metrics['ID_COMPTE'].astype(str)
    df_RF = data[['ID_COMPTE']].merge(rf_metrics, on='ID_COMPTE', how='left')
    df_RF = df_RF.drop('RECENCE_FREQUENCE', axis=1)

    # Remplissage des valeurs manquantes
    df_RF['NB_ACHAT'] = df_RF['NB_ACHAT'].fillna(0)

    val_incalculable = -1
    df_RF.loc[df_RF['NB_ACHAT'] < 2, ['MOY_DELAI_ACHAT', 'VAR_DELAI_ACHAT', 'RECENCE', 'SQRT_DELAI_ACHAT']] = val_incalculable
    df_RF.loc[df_RF['NB_ACHAT'] < 2, 'INDICATEUR_RF'] = 'Non calculé'

    df_RF['INDICATEUR_RF'] = df_RF['INDICATEUR_RF'].fillna('Non calculé')
        



    ### Montant
    # Calcul de SC_MONTANT
    df_Montant = df_CA
    df_Montant['SC_MONTANT'] = (df_Montant['CA_ANNUEL_ADDITI_N'] - df_Montant['CA_ANNUEL_ADDITI_N1']) / df_Montant['CA_ANNUEL_ADDITI_N1']

    # Calcul de INDICATEUR_MONTANT
    conditions = [
        (df_Montant['SC_MONTANT'] <= -0.4),
        (df_Montant['SC_MONTANT'] > -0.4) & (df_Montant['SC_MONTANT'] <= -0.15),
        (df_Montant['SC_MONTANT'] > -0.15) & (df_Montant['SC_MONTANT'] <= 0.15),
        (df_Montant['SC_MONTANT'] > 0.15),
        (df_Montant['SC_MONTANT'].isnull()) & (df_Montant['CA_ANNUEL_ADDITI_N'] > 0)
    ]
    choices = [
        'Très en retard',
        'En retard',
        'Stable',
        'En avance',
        'Client trop récent'
    ]
    df_Montant['INDICATEUR_MONTANT'] = np.select(conditions, choices, default='Non calculé')

    # Création de df_Montant
    df_Montant = df_Montant[['ID_COMPTE', 'INDICATEUR_MONTANT']]


    ## Récence-Fréquence-Montant
    # Fusionner df_RFM et df_Montant sur la colonne 'ID_COMPTE'
    df_RFM = df_RF.merge(df_Montant, on='ID_COMPTE', how='left')

    # Ajouter la colonne INDICATEUR_RFM avec les conditions spécifiées
    conditions = [
        # Relationnel
        ((df_RFM['INDICATEUR_MONTANT'] == 'En avance') & (df_RFM['INDICATEUR_RF'].isin(['Très en avance', 'En avance']))) |
        ((df_RFM['INDICATEUR_MONTANT'] == 'Client trop récent') & (df_RFM['INDICATEUR_RF'].isin(['Très en avance', 'En avance']))),
        
        # Push de parutions
        ((df_RFM['INDICATEUR_MONTANT'] == 'En avance') & (df_RFM['INDICATEUR_RF'].isin(['Très en retard', 'En retard']))) |
        ((df_RFM['INDICATEUR_MONTANT'] == 'Stable') & (df_RFM['INDICATEUR_RF'].isin(['Très en retard', 'En retard']))) |
        ((df_RFM['INDICATEUR_MONTANT'] == 'Client trop récent') & (df_RFM['INDICATEUR_RF'] == 'En retard')),
        
        # Push de CA
        ((df_RFM['INDICATEUR_MONTANT'].isin(['Très en retard', 'En retard'])) & (df_RFM['INDICATEUR_RF'].isin(['Très en avance', 'En avance']))),
        
        # Anti-Churn
        ((df_RFM['INDICATEUR_MONTANT'].isin(['Très en retard', 'En retard', 'Client trop récent'])) & 
        (df_RFM['INDICATEUR_RF'].isin(['Très en retard', 'En retard']))),
    
        # Trop récent
        (df_RFM['INDICATEUR_RF'].isin(['Non calculé', None])),
        
        # Abandonné
        (df_RFM['INDICATEUR_MONTANT'].isin(['Non calculé', None]))
    ]

    choices = [
        'Relationnel',
        'Push de parutions',
        'Push de CA',
        'Anti-Churn',
        'Trop récent',
        'Abandonné'
    ]

    df_RFM['INDICATEUR_RFM'] = np.select(conditions, choices, default='Non calculé')

    # Ajout des indicateurs RFM
    data_new_indicators = data_new_indicators.merge(df_RFM, on='ID_COMPTE', how='left')
    
    

    ## 3. Co-appétence
    # Liste des gammes et des classes
    gammes = histo_facturation['GAMME_PRODUIT'].unique()
    classes = list(MEDIAN.keys())

    # Création de la base avec tous les ID_COMPTE de data_new_indicators et les classes PMG
    ids = data_new_indicators['ID_COMPTE'].unique()
    df_co_appetence = data_new_indicators[['ID_COMPTE', 'CLASSE_PMG_N']].drop_duplicates()

    # Liste pour stocker les DataFrames temporaires
    liste_df = [df_co_appetence]

    # Boucle sur les gammes
    for gamme in gammes:
        gamme_col = gamme.replace(" ", "_")

        df_gamme = histo_facturation[histo_facturation['GAMME_PRODUIT'] == gamme]

        # Comptage des commandes et dernière date
        nb_gp = df_gamme.groupby('ID_COMPTE').size()
        dt_last = df_gamme.groupby('ID_COMPTE')['DT_COMMANDE'].max()

        # Création du DataFrame temporaire
        tmp_df = pd.DataFrame({'ID_COMPTE': ids})
        tmp_df[f'NB_GP_{gamme_col}'] = tmp_df['ID_COMPTE'].map(nb_gp).fillna(0).astype(int)

        tmp_df[f'DT_DERNIERE_GP_{gamme_col}'] = tmp_df['ID_COMPTE'].map(dt_last)
        tmp_df[f'DT_DERNIERE_GP_{gamme_col}'] = tmp_df[f'DT_DERNIERE_GP_{gamme_col}'].fillna(DATE_ETUDE)

        recence_col = f'RECENCE_GP_{gamme_col}'
        tmp_df[recence_col] = (pd.to_datetime(DATE_ETUDE) - tmp_df[f'DT_DERNIERE_GP_{gamme_col}']).dt.days

        tmp_df = tmp_df.merge(df_co_appetence[['ID_COMPTE', 'CLASSE_PMG_N']], on='ID_COMPTE', how='left')

        # Calcul du RF selon la classe si connue
        def calc_rf(row):
            classe = row['CLASSE_PMG_N']
            if classe in classes:
                return (row[recence_col] - 7 - MEDIAN[classe]) / STD[classe]
            else:
                return None

        rf_col = f'RF_GP_{gamme_col}'
        tmp_df[rf_col] = tmp_df.apply(calc_rf, axis=1)

        # Calcul du flag (fréquence basse) si classe connue
        def calc_fl(row):
            classe = row['CLASSE_PMG_N']
            if classe in classes:
                return int(row[recence_col] <= SEUIL[classe])
            else:
                return None

        fl_col = f'FL_RF_GP_{gamme_col}'
        tmp_df[fl_col] = tmp_df.apply(calc_fl, axis=1)

        # Calcul de l’indicateur si RF est défini
        def calc_indicateur(row):
            rf = row[rf_col]
            if pd.isna(rf):
                return None
            if rf < -0.5:
                return 'Très en avance'
            elif -0.5 <= rf <= 0:
                return 'En avance'
            elif 0 < rf <= 1.5:
                return 'En retard'
            else:
                return 'Très en retard'

        ind_col = f'INDICATEUR_RF_GP_{gamme_col}'
        tmp_df[ind_col] = tmp_df.apply(calc_indicateur, axis=1)

        # Ajout des nouvelles colonnes à la liste, en excluant les colonnes déjà présentes dans la base finale
        liste_df.append(tmp_df.drop(columns=['ID_COMPTE', 'CLASSE_PMG_N', recence_col]))

    # Fusion finale
    df_co_appetence = pd.concat(liste_df, axis=1)

    # Ajout des indicateurs Co-appétence sans réécraser CLASSE_PMG_N
    data_new_indicators = data_new_indicators.merge(df_co_appetence.drop(columns=['CLASSE_PMG_N']), on='ID_COMPTE', how='left')



    ## 4. Relation Client
    query = customer_relational_data(date)
    df_relation_client = pd.DataFrame(CNX(query))
    df_relation_client = df_relation_client[df_relation_client["ID_COMPTE"].isin(data["ID_COMPTE"].unique())]
    
    # Ajout des indicateurs de relation client
    data_new_indicators = data_new_indicators.merge(df_relation_client, on='ID_COMPTE', how='left')


    ## 5. Récences
    # Pour ne plus travailler avec des dates, on calcule la récence de chaque date par rapport à la date d'étude
    df_recence = pd.DataFrame()
    df_recence['ID_COMPTE'] = data['ID_COMPTE'].astype(str)
    
    # Seuil pour détecter une date factice comme 1900-01-01
    seuil_recence_max = 10000  # en jours, soit ~27 ans

    # Traitement des dates dans data
    columns_date = list(set(data.columns[data.columns.str.startswith('DT_')].tolist()))
    for col in columns_date:
        ind = col.replace('DT_', 'RECENCE_')
        recence = (pd.to_datetime(DATE_HEURE_ETUDE) - pd.to_datetime(data[col], format="%Y-%m-%d %H:%M:%S", errors='coerce')).dt.days
        recence = recence.apply(lambda x: seuil_recence_max if x >= seuil_recence_max else (-1 if x < 0 else x)) # on remplace les recences négatives par -1 pour éviter le data leakage et les dates imputées pr la valeur seuil
        df_recence[ind] = recence

    # Même traitement sur data_new_indicators
    columns_date = list(set(data_new_indicators.columns[data_new_indicators.columns.str.startswith('DT_')].tolist()))
    for col in columns_date:
        ind = col.replace('DT_', 'RECENCE_')
        recence = (pd.to_datetime(DATE_HEURE_ETUDE) - pd.to_datetime(data_new_indicators[col], format="%Y-%m-%d %H:%M:%S", errors='coerce')).dt.days
        recence = recence.apply(lambda x: seuil_recence_max if x >= seuil_recence_max else (-1 if x < 0 else x)) # on remplace les recences négatives par -1 pour éviter le data leakage et les dates imputées pr la valeur seuil
        df_recence[ind] = recence

    # Ajout des indicateurs de récence
    data_new_indicators = data_new_indicators.merge(df_recence, on='ID_COMPTE', how='left')



    ## 6. Régions
    # Dictionnaire : departement (code) -> région
    dict_dep_region: dict[str, str] = {
        '01': 'Auvergne-Rhône-Alpes',
        '02': 'Hauts-de-France',
        '03': 'Auvergne-Rhône-Alpes',
        '04': 'Provence-Alpes-Côte d\'Azur',
        '05': 'Provence-Alpes-Côte d\'Azur',
        '06': 'Provence-Alpes-Côte d\'Azur',
        '07': 'Auvergne-Rhône-Alpes',
        '08': 'Grand Est',
        '09': 'Occitanie',
        '10': 'Grand Est',
        '11': 'Occitanie',
        '12': 'Occitanie',
        '13': 'Provence-Alpes-Côte d\'Azur',
        '14': 'Normandie',
        '15': 'Auvergne-Rhône-Alpes',
        '16': 'Nouvelle-Aquitaine',
        '17': 'Nouvelle-Aquitaine',
        '18': 'Centre-Val de Loire',
        '19': 'Nouvelle-Aquitaine',
        '2A': 'Corse',
        '2B': 'Corse',
        '21': 'Bourgogne-Franche-Comté',
        '22': 'Bretagne',
        '23': 'Nouvelle-Aquitaine',
        '24': 'Nouvelle-Aquitaine',
        '25': 'Bourgogne-Franche-Comté',
        '26': 'Auvergne-Rhône-Alpes',
        '27': 'Normandie',
        '28': 'Centre-Val de Loire',
        '29': 'Bretagne',
        '30': 'Occitanie',
        '31': 'Occitanie',
        '32': 'Occitanie',
        '33': 'Nouvelle-Aquitaine',
        '34': 'Occitanie',
        '35': 'Bretagne',
        '36': 'Centre-Val de Loire',
        '37': 'Centre-Val de Loire',
        '38': 'Auvergne-Rhône-Alpes',
        '39': 'Bourgogne-Franche-Comté',
        '40': 'Nouvelle-Aquitaine',
        '41': 'Centre-Val de Loire',
        '42': 'Auvergne-Rhône-Alpes',
        '43': 'Auvergne-Rhône-Alpes',
        '44': 'Pays de la Loire',
        '45': 'Centre-Val de Loire',
        '46': 'Occitanie',
        '47': 'Nouvelle-Aquitaine',
        '48': 'Occitanie',
        '49': 'Pays de la Loire',
        '50': 'Normandie',
        '51': 'Grand Est',
        '52': 'Grand Est',
        '53': 'Pays de la Loire',
        '54': 'Grand Est',
        '55': 'Grand Est',
        '56': 'Bretagne',
        '57': 'Grand Est',
        '58': 'Bourgogne-Franche-Comté',
        '59': 'Hauts-de-France',
        '60': 'Hauts-de-France',
        '61': 'Normandie',
        '62': 'Hauts-de-France',
        '63': 'Auvergne-Rhône-Alpes',
        '64': 'Nouvelle-Aquitaine',
        '65': 'Occitanie',
        '66': 'Occitanie',
        '67': 'Grand Est',
        '68': 'Grand Est',
        '69': 'Auvergne-Rhône-Alpes',
        '70': 'Bourgogne-Franche-Comté',
        '71': 'Bourgogne-Franche-Comté',
        '72': 'Pays de la Loire',
        '73': 'Auvergne-Rhône-Alpes',
        '74': 'Auvergne-Rhône-Alpes',
        '75': 'Île-de-France',
        '76': 'Normandie',
        '77': 'Île-de-France',
        '78': 'Île-de-France',
        '79': 'Nouvelle-Aquitaine',
        '80': 'Hauts-de-France',
        '81': 'Occitanie',
        '82': 'Occitanie',
        '83': 'Provence-Alpes-Côte d\'Azur',
        '84': 'Provence-Alpes-Côte d\'Azur',
        '85': 'Pays de la Loire',
        '86': 'Nouvelle-Aquitaine',
        '87': 'Nouvelle-Aquitaine',
        '88': 'Grand Est',
        '89': 'Bourgogne-Franche-Comté',
        '90': 'Bourgogne-Franche-Comté',
        '91': 'Île-de-France',
        '92': 'Île-de-France',
        '93': 'Île-de-France',
        '94': 'Île-de-France',
        '95': 'Île-de-France',
        '97': 'Outre-Mer',
        '99' : 'Etranger'
    }

    # Création des colonnes de région et de code département
    df_region = pd.DataFrame()
    df_region['ID_COMPTE'] = data['ID_COMPTE'].astype(str)

    df_region['CD_DEPARTEMENT'] = data['DEPARTEMENT'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_region['CD_DEPARTEMENT'] = df_region['CD_DEPARTEMENT'].apply(lambda x: x if len(x) > 1 else '0' + x)
    df_region['CD_DEPARTEMENT'] = df_region['CD_DEPARTEMENT'].replace({'20': '2A'})

    df_region['REGION'] = df_region['CD_DEPARTEMENT'].map(dict_dep_region)
    df_region['REGION'] = df_region['REGION'].fillna('Autre')


    # Ajout des indicateurs de récence
    data_new_indicators = data_new_indicators.merge(df_region, on='ID_COMPTE', how='left')


    ## 9. Saisonnalité
    # Dictionnaire des saisons
    dict_saison = {
        1: 'HAUTE',
        2: 'HAUTE',
        3: 'MOYENNE',
        4: 'MOYENNE',
        5: 'MOYENNE',
        6: 'MOYENNE',
        7: 'BASSE',
        8: 'BASSE',
        9: 'MOYENNE',
        10: 'HAUTE',
        11: 'HAUTE',
        12: 'TRES HAUTE'
    }

    # Création des colonnes MOIS et SAISON
    df_saison = pd.DataFrame()
    df_saison['ID_COMPTE'] = data['ID_COMPTE'].astype(str)

    df_saison['MOIS_SAISON'] = data['_DATE_ETUDE_'].str[5:7].astype(int)
    df_saison['SAISON'] = df_saison['MOIS_SAISON'].map(dict_saison)
    
    df_saison['MOIS_SAISON'] = df_saison['MOIS_SAISON'].astype(str)
    
    # Ajout des indicateurs de saison
    data_new_indicators = data_new_indicators.merge(df_saison, on='ID_COMPTE', how='left')



    # Restrictions 
    ## Suppression des lignes où CLASSE_PMG_N est égal à 'CA négatif' ou 'Hors catégorie'
    data_new_indicators = data_new_indicators[~data_new_indicators['CLASSE_PMG_N'].isin(['CA négatif', 'Hors catégorie'])]


    # Return
    print(f"    Création des nouveaux indicateurs terminée | Nombre de lignes : {data_new_indicators.shape[0]} | Nombre de colonnes : {data_new_indicators.shape[1]}")
    print('---')
    return data_new_indicators

