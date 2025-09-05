
def apply_scaling(df: pd.DataFrame,
                  quanti_cols: list = []) -> pd.DataFrame:
    """
    Applique une standardisation (centrage-réduction) aux variables quantitatives spécifiées dans le DataFrame.
    """
    
    if quanti_cols:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[quanti_cols])
        
        scaled_df = pd.DataFrame(scaled_values, columns=quanti_cols, index=df.index)
        df = df.drop(columns=quanti_cols)
        df = pd.concat([df, scaled_df], axis=1)

    return df



def encode_nominal(df, nominal_cols):
    """
    Encode les variables nominales à l’aide d’un OneHotEncoder.
    """
    df[nominal_cols] = df[nominal_cols].astype(str)  # Assurer que les colonnes sont de type string
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[nominal_cols])
    
    # Création des noms de colonnes
    encoded_cols = encoder.get_feature_names_out(nominal_cols)
    
    # DataFrame des colonnes encodées
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index).astype(bool)
    
    # Supprimer les colonnes nominales originales et concaténer les encodées
    df = df.drop(columns=nominal_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df


def encode_ordinal(df, ordinal_cols, ordering_dict, verbose):
    """
    Encode les variables ordinales en conservant l’ordre de modalité défini.

    Paramètres :
        df : pd.DataFrame
            DataFrame contenant les colonnes ordinales à encoder.
        ordinal_cols : list
            Liste des colonnes ordinales à transformer.
        ordering_dict : dict
            Dictionnaire contenant pour chaque colonne une liste ordonnée des modalités.

    Retour :
        pd.DataFrame
            DataFrame avec les colonnes ordinales remplacées par leurs valeurs encodées.
    """
    # Créer la liste des ordres dans le bon ordre
    # categories = [ordering_dict[col] for col in ordinal_cols]
    categories = []
    for col in ordinal_cols:
        seen = set()
        ordered_unique = [x for x in ordering_dict[col] if not (x in seen or seen.add(x))]
        categories.append(ordered_unique)
    
    encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
    encoded = encoder.fit_transform(df[ordinal_cols])
    
    # Créer un DataFrame avec les colonnes ordinales encodées
    encoded_df = pd.DataFrame(encoded, columns=ordinal_cols, index=df.index).astype(int)
    
    # Correspondance des valeurs Avant VS Après encodage
    dict_mapping = {}
    for i, col in enumerate(ordinal_cols):
        mapping = dict(zip(categories[i], range(len(categories[i]))))
        mapping["<inconnu>"] = -1

        dict_mapping[col] = mapping

        if verbose:
            print(f"\t\t > '{col}' - Mapping appliqué : {mapping}")
            print()
        
    # Remplacer les colonnes originales
    df = df.drop(columns=ordinal_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df, dict_mapping


def apply_encoding(df,
                   nominal_cols=[], 
                   ordinal_cols=[], 
                   ordering_dict={},
                   verbose: bool = False):
    """
    Applique les encodages nominal (OneHot) et ordinal aux colonnes désignées d’un DataFrame.

    Cette fonction combine l’encodage one-hot pour les variables nominales avec l’encodage ordinal 
    pour les colonnes possédant un ordre de modalités, le tout dans un seul appel unifié.
    """
    if nominal_cols:
        df = encode_nominal(df, nominal_cols=nominal_cols)
    if ordinal_cols and ordering_dict:
        df, dict_mapping = encode_ordinal(df, ordinal_cols, ordering_dict, verbose)

    return df, dict_mapping


def apply_features_preprocessing(df_train: pd.DataFrame,
                                df_val: pd.DataFrame,
                                df_test: pd.DataFrame,
                                quanti_cols: list = [],
                                nominal_cols: list = [],
                                ordinal_cols: list = [],
                                ordering_dict: dict = {},
                                verbose: bool = False) -> pd.DataFrame:
    """
    Fonction pour prétraiter les features en appliquant l'encodage des variables nominales et ordinales.
    
    Paramètres :
        df : pd.DataFrame
            DataFrame contenant les données à prétraiter
        nominal_cols : list, par défaut []
            Liste des colonnes nominales à encoder
        ordinal_cols : list, par défaut []
            Liste des colonnes ordinales à encoder
        ordering_dict : dict, par défaut {}
            Dictionnaire avec pour chaque colonne ordinale la liste ordonnée des modalités
        verbose : bool, par défaut False
            Indicateur pour afficher les messages de progression
    
    Retour :
        pd.DataFrame
            DataFrame avec les features prétraitées
    """

        
    print('---')
    print('\t Lancement du pré-traitement des features.')
    
    # Application de la standardisation et de l'encodage
    df_train = apply_scaling(df_train, quanti_cols) # inutile quand data_discretization est appliqué
    df_train, dict_mapping = apply_encoding(df_train, nominal_cols, ordinal_cols, ordering_dict, verbose)

    df_val = apply_scaling(df_val, quanti_cols) # inutile quand data_discretization est appliqué
    df_val, _ = apply_encoding(df_val, nominal_cols, ordinal_cols, ordering_dict)

    df_test = apply_scaling(df_test, quanti_cols) # inutile quand data_discretization est appliqué
    df_test, _ = apply_encoding(df_test, nominal_cols, ordinal_cols, ordering_dict)

    # Gestion des colonnes particulières
    # Création de la liste des colonnes à supprimer (finissant par les suffixes donnés)
    suffixes = ['Group date future', 'Group date manquante'] # , 'Group incalculable'
    cols_to_drop = set()
    for df in [df_train, df_val, df_test]:
        for col in df.columns:
            if any(col.endswith(suffix) for suffix in suffixes):
                cols_to_drop.add(col)
    df_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    df_val = df_val.drop(columns=cols_to_drop, errors='ignore')
    df_test = df_test.drop(columns=cols_to_drop, errors='ignore')

    print("\t Pré-traitement des features terminé.")
    print('---')

    return df_train, df_val, df_test, dict_mapping