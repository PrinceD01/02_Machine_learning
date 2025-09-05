# Setup environnement
warnings.filterwarnings("ignore")



def calcul_entropy(df: pd.DataFrame, 
                   col_bin: str, 
                   target: str) -> float :
    """
    Fonction pour calculer l'entropie pondérée moyenne d'une variable cible selon une variable explicative.

    Paramètres :
        df : pd.DataFrame
            Jeu de données contenant les colonnes à analyser
        col_bin : str
            Nom de la colonne représentant les bins (généralement issue d’un découpage sur une variable)
        target : str
            Nom de la colonne cible (variable catégorielle) pour laquelle on mesure l’entropie conditionnelle

    Retour :
        float
            Entropie pondérée globale entre les bins et la variable cible
    """
    entropies = []
    for bin in df[col_bin].unique():
        sub = df[df[col_bin] == bin]
        probs = sub[target].value_counts(normalize=True)
        ent = entropy(probs, base=2)
        weight = len(sub) / len(df)
        entropies.append(ent * weight)
    return sum(entropies)

def format_bin_label(left, right) -> str:
    """
    Génère une étiquette textuelle pour représenter un intervalle (bin).

    Paramètres :
        left : float, int ou -np.inf
            Borne gauche de l'intervalle. -np.inf est représenté par '-inf'.
        right : float, int ou np.inf
            Borne droite de l'intervalle. +np.inf est représenté par '+inf'.

    Retour :
        str
            Une chaîne formatée de la forme "bin_<borne_gauche>_<borne_droite>",
            avec arrondi à 2 décimales et suppression des zéros inutiles.
    """
    def format_edge(x) -> str:
        if x == -np.inf:
            return '-inf'
        elif x == np.inf:
            return '+inf'
        else:
            return str(round(float(x), 2)).rstrip('0').rstrip('.') if isinstance(x, numbers.Number) else str(x)
    return f"bin_{format_edge(left)}_{format_edge(right)}"

def get_special_mask(df: pd.DataFrame, column: str) -> pd.Series:
    """Retourne un masque booléen des valeurs spéciales pour une colonne donnée."""
    val_future, val_missing, val_incalculable = -1, 10000, -1
    
    if column.startswith("RECENCE_"):
        return df[column].isin([val_future, val_missing])
    elif column in ['MOY_DELAI_ACHAT', 'VAR_DELAI_ACHAT', 'RECENCE', 'SQRT_DELAI_ACHAT']:
        return df[column] == val_incalculable
    return pd.Series(False, index=df.index)

def add_special_categories(df: pd.DataFrame, column: str, col_discretized: str) -> list[str]:
    """Ajoute les catégories spéciales en fonction du type de variable."""
    special_cats = []
    val_future, val_missing, val_incalculable = -1, 10000, -1

    if column.startswith("RECENCE_"):
        cats = ['Group date future', 'Group date manquante']
        df[col_discretized] = df[col_discretized].cat.add_categories(cats)
        df.loc[df[column] == val_future, col_discretized] = 'Group date future'
        df.loc[df[column] == val_missing, col_discretized] = 'Group date manquante'
        special_cats = cats

    elif column in ['MOY_DELAI_ACHAT', 'VAR_DELAI_ACHAT', 'RECENCE', 'SQRT_DELAI_ACHAT']:
        cat = 'Group incalculable'
        df[col_discretized] = df[col_discretized].cat.add_categories([cat])
        df.loc[df[column] == val_incalculable, col_discretized] = cat
        special_cats = [cat]

    return special_cats

def get_tree_thresholds(tree: DecisionTreeClassifier) -> list[float]:
    """
    Fonction pour extraire les seuils de découpage (thresholds) utilisés dans un arbre de décision entraîné.

    Étapes principales :
        1. Accède à l'attribut `.tree_` d'un arbre sklearn (ex. DecisionTreeClassifier ou DecisionTreeRegressor).
        2. Parcourt récursivement tous les nœuds internes de l’arbre.
        3. Récupère les valeurs des seuils (thresholds) utilisés pour effectuer les splits.
        4. Élimine les doublons et trie les seuils.

    Paramètres :
        tree : sklearn.tree.DecisionTreeClassifier ou DecisionTreeRegressor
            Arbre de décision préalablement entraîné avec `.fit()`

    Retour :
        thresholds : list
            Liste triée des seuils de split uniques utilisés dans l'arbre
    """
    tree_ = tree.tree_
    thresholds = []

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            thresholds.append(tree_.threshold[node])
            recurse(tree_.children_left[node])
            recurse(tree_.children_right[node])
    
    recurse(0)
    return sorted(set(thresholds))

def fit_tree_discretizer(X: pd.Series, y: pd.Series, max_modalities: int) -> list:
    """Entraîne un arbre supervisé et retourne les seuils optimaux."""
    tree = DecisionTreeClassifier(criterion='entropy', random_state=241)
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'min_samples_leaf': [0.01, 0.03, 0.05, 0.1],
        'class_weight': [None, 'balanced'],
        'max_leaf_nodes': list(range(2, max_modalities))
    }
    grid = GridSearchCV(tree, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X, y)

    return get_tree_thresholds(grid.best_estimator_)


def discretize_variable(df: pd.DataFrame,
                        column: str,
                        target: str,
                        max_modalities: int=5,
                        verbose: bool=False) -> tuple[pd.DataFrame, list, list, list]:
    """
    Discrétise une variable quantitative continue en un nombre fini de modalités qualitatives (bins),
    selon une approche supervisée à l'aide d'un arbre de décision optimisé via GridSearchCV.

    Cette méthode est particulièrement adaptée lorsque l’objectif est de capturer la relation entre une
    variable explicative quantitative et une variable cible qualitative (classification supervisée),
    tout en intégrant des traitements spécifiques pour les valeurs particulières (ex. : données manquantes ou incohérentes
    dans les variables de type 'RECENCE_' ou '_DELAI_ACHAT').

    Étapes du processus :
        1. Détection et gestion des modalités remarquables si la variable est de type 'RECENCE_' ou '_DELAI_ACHAT'.
        2. Exclusion des valeurs remarquables pour l'apprentissage supervisé.
        3. Si le nombre d'observations normales est insuffisant (< 200), la variable est discrétisée dans une seule modalité.
        4. Sinon, un arbre de décision (`DecisionTreeClassifier`) est entraîné sur la variable et la cible,
           avec une recherche de grille (`GridSearchCV`) sur les hyperparamètres pour optimiser la séparation supervisée.
        5. Les seuils de coupure sont extraits de l’arbre optimal pour définir les bornes des bins.
        6. La discrétisation est appliquée à l’ensemble des observations (y compris les valeurs remarquables),
           et une nouvelle variable catégorielle `CLASS_<column>` est ajoutée au DataFrame.

    Paramètres :
        df : pd.DataFrame
            Jeu de données d'entrée contenant la variable à discrétiser et la cible.
        column : str
            Nom de la variable quantitative à discrétiser.
        target : str
            Nom de la variable cible (binaire ou multiclasse) à utiliser pour superviser la discrétisation.
        max_modalities : int, facultatif (par défaut = 5)
            Nombre maximal de feuilles autorisé dans l'arbre de décision, ce qui limite le nombre de bins générés (max_bins = max_modalities).
        verbose : bool, facultatif (par défaut = False)
            Si True, affiche des détails sur les étapes de la discrétisation (seuils, entropie, répartition des classes...).

    Retour
        tuple :
            - df : pd.DataFrame
                Le DataFrame enrichi d’une nouvelle colonne catégorielle `CLASS_<column>` représentant les bins.
            - bins : list
                Liste des bornes numériques utilisées pour la discrétisation (incluant -inf et +inf).
            - labels : list
                Liste des libellés des bins (ex. : ['bin_1', 'bin_2', ...]).
            - special_cats : list
                Liste des modalités supplémentaires ajoutées pour les valeurs remarquables
    """

    if verbose:
        print(f"\t\t Discrétisation de la variable '{column}'")

    col_discretized = f'CLASS_{column}'
    mask_special = get_special_mask(df, column)
    df_normal = df.loc[~mask_special, [column, target]].copy()

    # Cas 1 : pas assez d'observations pour apprendre
    if df_normal.shape[0] < 200:
        if verbose:
            msg = "Aucune observation" if df_normal.empty else f"Pas assez d'observations (n={df_normal.shape[0]})"
            print(f"\t\t\t {msg} pour discrétiser '{column}'")

        df[col_discretized] = 'bin_1'
        df[col_discretized] = df[col_discretized].astype('category')
        special_cats = add_special_categories(df, column, col_discretized)

        return df.drop(columns=[col_discretized]), [-np.inf, np.inf], ['bin_1'], special_cats

    # Cas 2 : apprentissage normal
    df_normal[column] = df_normal[column].replace([np.inf, -np.inf], np.nan).fillna(df_normal[column].median())
    thresholds = fit_tree_discretizer(df_normal[[column]], df_normal[target], max_modalities)

    bins = [-np.inf] + thresholds + [np.inf]
    labels = [format_bin_label(left, right) for left, right in zip(bins[:-1], bins[1:])]
    df[col_discretized] = pd.cut(df[column], bins=bins, labels=labels)

    # Ajout des catégories spéciales
    df[col_discretized] = df[col_discretized].astype('category')
    special_cats = add_special_categories(df, column, col_discretized)

    if verbose:
        print("\t\t\t Seuils utilisés :", bins)
        print("\t\t\t Répartition des classes :")
        print(pd.crosstab(df[col_discretized], df[target], normalize='index').round(2) * 100)

    return df.drop(columns=[col_discretized]), bins, labels, special_cats



def apply_discretization_from_bins(df: pd.DataFrame,
                                    column: str,
                                    bins: list,
                                    labels: list,
                                    special_cats: list = None) -> pd.DataFrame:
    """
    Applique une discrétisation sur un DataFrame à partir de seuils (bins) et labels
    déjà appris lors d'une étape précédente de discrétisation supervisée.

    Paramètres :
        df : pd.DataFrame
            Jeu de données sur lequel appliquer la discrétisation.
        column : str
            Nom de la colonne quantitative à discrétiser.
        bins : list
            Liste des seuils à utiliser pour les intervalles (issus d'un apprentissage précédent).
        labels : list
            Liste des étiquettes à associer à chaque intervalle de bins.
        special_cats : list, optionnel (par défaut = None)
            Liste des catégories remarquables à réinjecter (ex : 'Group date future', 'Group date manquante').

    Retour :
        pd.DataFrame
            Le DataFrame original avec une nouvelle colonne discrétisée : 'CLASS_<column>'
    """
    col_discretized = f'CLASS_{column}'
    df[col_discretized] = pd.cut(df[column], bins=bins, labels=labels)
    df[col_discretized] = df[col_discretized].astype('category')

    if special_cats:
        for cat in special_cats:
            df[col_discretized] = df[col_discretized].cat.add_categories([cat])

    add_special_categories(df, column, col_discretized)

    return df.drop(columns=[col_discretized])


def apply_discretization(df_train: pd.DataFrame,
                         df_val: pd.DataFrame,
                         df_test: pd.DataFrame,
                         columns: list[str],
                         target: str,
                         max_modalities: int=5,
                         verbose: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applique la discrétisation supervisée (guidée par la cible) sur une liste de colonnes quantitatives
    pour trois jeux de données : train, validation et test. Utilise les seuils appris sur le train pour
    transformer val et test.

    Paramètres :
        df_train : pd.DataFrame
            Jeu de données d'entraînement contenant les colonnes à discrétiser.
        df_val : pd.DataFrame
            Jeu de données de validation.
        df_test : pd.DataFrame
            Jeu de données de test.
        columns : list of str
            Noms des colonnes quantitatives à transformer.
        target : str
            Nom de la variable cible utilisée pour guider les discrétisations.
        max_modalities : int, optionnel (par défaut = 5)
            Nombre maximum de feuilles pour l'arbre de décision donc de bins pour la discrétisation.
        verbose : bool, optionnel (par défaut = False)
            Si True, affiche les informations détaillées pour chaque variable.

    Retour :
        tuple de trois pd.DataFrame
            Les DataFrames df_train, df_val et df_test mis à jour avec les nouvelles colonnes discrétisées.
    """
    
    print('---')
    print("\t Lancement de la discrétisation des variables.")

    for col in columns:
        df_train, bins_trained, labels_trained, special_cats_trained = discretize_variable(df_train, col, target, max_modalities, verbose)
        df_val = apply_discretization_from_bins(df_val, col, bins_trained, labels_trained, special_cats_trained)
        df_test = apply_discretization_from_bins(df_test, col, bins_trained, labels_trained, special_cats_trained)
        
    print("\t Discrétisation terminée.")
    print('---')
    return df_train, df_val, df_test
