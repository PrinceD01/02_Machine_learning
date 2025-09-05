
def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Prépare les features pour statsmodels (dummies + constante)."""
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return sm.add_constant(X)


def get_criterion(X: pd.DataFrame, 
                                    y: pd.Series, 
                                    criterion: str = "BIC"
                                ) -> float:
    """
    Ajuste une régression logistique (statsmodels) et retourne AIC ou BIC.
    """
    X_with_const = _prepare_features(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=HessianInversionWarning)
        model = sm.Logit(y, X_with_const).fit(disp=False, method="lbfgs", maxiter=1000)

    if criterion.upper() == "BIC":
        return model.bic
    elif criterion.upper() == "AIC":
        return model.aic
    else:
        raise ValueError("criterion doit être 'AIC' ou 'BIC'")


def calcul_GINI(y_true: pd.Series, y_scores: np.ndarray) -> float:
    """
    Calcule le coefficient de Gini.
    """
    auc = roc_auc_score(y_true, y_scores)
    return 2 * auc - 1


def remove_quasi_constant_features(df: pd.DataFrame, threshold_constant: float = 0.99) -> List[str]:
    """Supprime les variables quasi-constantes (ex: 99% des valeurs identiques)."""
    return [
        col for col in df.columns
        if df[col].value_counts(normalize=True).iloc[0] < threshold_constant
    ]


def remove_multicollinearity(df: pd.DataFrame, threshold_corr: float = 0.7) -> List[str]:
    """
    Supprime les variables fortement corrélées entre elles.
    Retourne la liste des colonnes conservées.
    """
    df_dummies = pd.get_dummies(df, drop_first=True)
    corr_matrix = df_dummies.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop_dummies = [col for col in upper_tri.columns if any(upper_tri[col] > threshold_corr)]

    # Map dummy → original
    dummy_to_original = {
        dummy: orig
        for dummy in df_dummies.columns
        for orig in df.columns
        if dummy.startswith(orig + "_") or dummy == orig
    }

    dropped_originals = {dummy_to_original[d] for d in to_drop_dummies if d in dummy_to_original}
    return [col for col in df.columns if col not in dropped_originals]



# Sélection Forward
def select_features(df_train: pd.DataFrame,
                        target: str = "FL_ACHAT_PRODUIT",
                        criterion: str = "BIC",
                        nb_max_features: int = 30,
                        threshold_corr: float = 0.7,
                        threshold_constant: float = 0.99,
                    ) -> List[str]:
    """
    Sélectionne les variables explicatives step-by-step selon critère hybride :
    - BIC (ou AIC) doit s'améliorer
    - GINI doit augmenter
    """
    selected_features: List[str] = []
    all_features = [f for f in df_train.columns if f != target]
    df_X = df_train[all_features]

    # Nettoyage
    df_X = df_X[remove_quasi_constant_features(df_X, threshold_constant)]
    remaining_features = remove_multicollinearity(df_X, threshold_corr)
    y = df_train[target]

    current_criterion = np.inf
    current_gini = 0.0

    for _ in range(min(nb_max_features, len(remaining_features))):
        best_feature, best_criterion, best_gini = None, current_criterion, current_gini

        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X = df_train[features_to_test]
            crit_value = get_criterion(X, y, criterion)
            model = LogisticRegression(solver="lbfgs", C=0.1, penalty="l2",
                                            max_iter=1000, class_weight="balanced", random_state=241
                                        )
            model.fit(X, y)
            gini = calcul_GINI(y, model.predict_proba(X)[:, 1])

            if crit_value < best_criterion and gini > best_gini:
                best_feature, best_criterion, best_gini = feature, crit_value, gini

        if best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            current_criterion, current_gini = best_criterion, best_gini
        else:
            break

    return selected_features


def consolidate_selection(df_train: pd.DataFrame,
                            target: str = "FL_ACHAT_PRODUIT",
                            n_splits: int = 5,
                            criterion: str = "BIC",
                            nb_max_features: int = 30,
                            threshold_corr: float = 0.7,
                            threshold_constant: float = 0.99,
                            verbose: bool = False,
                        ) -> List[str]:
    """Stabilise la sélection de variables via validation croisée stratifiée."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=241)
    all_selected = {}

    for i, (train_idx, _) in enumerate(skf.split(df_train, df_train[target])):
        fold_train = df_train.iloc[train_idx]
        selected = select_features(
            fold_train, target=target, criterion=criterion,
            nb_max_features=nb_max_features,
            threshold_corr=threshold_corr,
            threshold_constant=threshold_constant
        )
        all_selected[i] = selected
        if verbose:
            print(f"\t Split {i+1}/{n_splits} - Variables sélectionnées : {selected}")

    flat_list = [var for sel in all_selected.values() for var in sel]
    counts = Counter(flat_list)

    threshold = n_splits // 2 + 1
    consolidated = [var for var, count in counts.items() if count >= threshold]

    if verbose:
        print(f"\t Variables consolidées (>= {threshold} splits) : {consolidated}\n")

    return consolidated


# =====================================================
# Ajout métier & pipeline global
# =====================================================

def add_business_features(features_selected: List[str], business_features: List[str]) -> List[str]:
    """Ajoute des variables métier si elles ne sont pas déjà présentes."""
    return features_selected + [f for f in business_features if f not in features_selected]


def apply_features_selection(df_train: pd.DataFrame,
                                df_val: pd.DataFrame,
                                df_test: pd.DataFrame,
                                business_features: Optional[List[str]] = None,
                                target: str = "FL_ACHAT_PRODUIT",
                                criterion: str = "BIC",
                                nb_max_features: int = 30,
                                n_splits: int = 5,
                                threshold_corr: float = 0.8,
                                threshold_constant: float = 0.99,
                                verbose: bool = False,
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Étape 1 – Sélection Forward - ajout step by step de variables
        - Critère de  sur critère hybride : BIC (ou AIC) + GINI

    Étape 2 - Stabilisation de la sélection 
        - on réexécute la sélection sur différents splits du train 
        - on ne garde que les variables sélectionnées dans plus de la moitié des splits

    Étape 3 - Ajout de variables sur critère métier
        - on force l'ajout de certaines variables métiers (possibilité de créer de variable composite pour maximiser leur apport significatif)
    """
    
    print("---\t Lancement de la sélection des features.")

    # Étape 1 : Sélection Forward + Étape 2 : Consolidation de la sélection
    features_selected = consolidate_selection(df_train, target=target, n_splits=n_splits, threshold_corr=threshold_corr, threshold_constant=threshold_constant, verbose=verbose)
    
    # Étape 3 : Ajout de variables métier
    if business_features is not None:
        features_selected = add_business_features(features_selected, business_features)
    
    # Sélection des features dans les DataFrames
    if target not in features_selected: features_selected.append(target)  # On ajoute la cible si elle n'est pas déjà présente
    df_train = df_train[features_selected]
    df_val = df_val[features_selected]
    df_test = df_test[features_selected]


    print(f"\t   > Nombre de variables sélectionnées : {len(features_selected)} | Variables : {features_selected}")

    print("\t Sélection des features terminé.")
    print('---')

    return df_train, df_val, df_test
