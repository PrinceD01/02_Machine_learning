
def calcul_woe_iv(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """
    Calcule le WoE (Weight of Evidence) et l'IV (Information Value) pour chaque modalité d'une variable catégorielle
    par rapport à une cible binaire (0 ou 1). Utilisé en scoring, modélisation logistique ou réduction supervisée.
    """
    EPS = 1e-6  # pour éviter les divisions par zéro


    df_temp = df[[feature, target]].copy()
    grouped = df_temp.groupby(feature, observed=True)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']


    # Distributions
    grouped['dist_bad'] = grouped['bad'] / (grouped['bad'].sum() + EPS)
    grouped['dist_good'] = grouped['good'] / (grouped['good'].sum() + EPS)


    # Calcul WoE & IV
    grouped['woe'] = np.log((grouped['dist_good'] + EPS) / (grouped['dist_bad'] + EPS))
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']


    return grouped.reset_index()

def get_label_new_group(grp1: str, grp2: str) -> str:
    """Fusionne deux labels de groupes de modalités en conservant l'ordre et l'unicité."""
    merged_parts = sorted(set(grp1.split(" && ") + grp2.split(" && ")))
    return " && ".join(merged_parts)

def build_grouping(df: pd.DataFrame, column: str, target: str, n_groups: int) -> dict:
    """
    Regroupement classique des modalités d’une variable catégorielle à l’aide d’un algorithme de fusion hiérarchique 
    basé sur la similarité des WoE (Weight of Evidence), afin de réduire le nombre de modalités à un nombre cible.
    """
    woe_table = calcul_woe_iv(df, column, target).sort_values('woe')
    modalities = list(woe_table[column])
    woe_dict = dict(zip(woe_table[column], woe_table['woe']))
    total_dict = dict(zip(woe_table[column], woe_table['total']))

    group_assignments = {mod: mod for mod in modalities}
    group_woe = woe_dict.copy()
    heap = [(abs(woe_dict[mod1] - woe_dict[mod2]), (mod1, mod2))
        for i, mod1 in enumerate(modalities)
        for mod2 in modalities[i+1:]]
    for item in heap:
        heappush([], item)


    while len(set(group_assignments.values())) > n_groups and heap:
        _, (mod1, mod2) = heappop(heap)
        if mod1 not in group_assignments or mod2 not in group_assignments:
            continue

        grp1, grp2 = group_assignments[mod1], group_assignments[mod2]
        if grp1 == grp2:
            continue

    # Fusion
    new_group = get_label_new_group(grp1, grp2)
    total1 = sum(total_dict[m] for m in group_assignments if group_assignments[m] == grp1)
    total2 = sum(total_dict[m] for m in group_assignments if group_assignments[m] == grp2)
    new_total = total1 + total2
    new_woe = (group_woe[grp1] * total1 + group_woe[grp2] * total2) / new_total


    for mod in group_assignments:
        if group_assignments[mod] in {grp1, grp2}:
            group_assignments[mod] = new_group

    group_woe[new_group] = new_woe
    total_dict[new_group] = new_total


    for other_group in set(group_assignments.values()) - {new_group}:
        diff = abs(group_woe[new_group] - group_woe[other_group])
        heappush(heap, (diff, (new_group, other_group)))

    return group_assignments

def fallback_build_grouping(df: pd.DataFrame, column: str, target: str, max_modalities: int, woe_threshold: float) -> dict:
    """
    Fusionne les modalités d'une variable catégorielle en groupes homogènes en utilisant la 
    proximité des valeurs de WoE (Weight of Evidence), comme solution de repli si les méthodes
    principales échouent.
    """
    woe_table = calcul_woe_iv(df, column, target).sort_values('woe')
    modalities = list(woe_table[column])
    woe_dict = dict(zip(woe_table[column], woe_table['woe']))
    total_dict = dict(zip(woe_table[column], woe_table['total']))

    group_assignments = {mod: mod for mod in modalities}
    group_woe = woe_dict.copy()
    heap = []

    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            mod1, mod2 = modalities[i], modalities[j]
            diff = abs(woe_dict[mod1] - woe_dict[mod2])
            if diff <= woe_threshold:
                heappush(heap, (diff, (mod1, mod2)))

    while heap:
        # Stop si on a atteint max_modalities + 1 groupes ou moins
        if len(set(group_assignments.values())) <= max_modalities + 1:
            break
            
        diff, (mod1, mod2) = heappop(heap)
        if diff > woe_threshold:
            break
        if mod1 not in group_assignments or mod2 not in group_assignments:
            continue
        grp1, grp2 = group_assignments[mod1], group_assignments[mod2]
        if grp1 == grp2:
            continue
        
        parts1 = grp1.split(" && ")
        parts2 = grp2.split(" && ")
        merged_parts = sorted(set(parts1 + parts2))
        new_group = " && ".join(merged_parts)
        
        total1 = sum(total_dict[mod] for mod in group_assignments if group_assignments[mod] == grp1)
        total2 = sum(total_dict[mod] for mod in group_assignments if group_assignments[mod] == grp2)
        new_total = total1 + total2
        new_woe = (group_woe[grp1] * total1 + group_woe[grp2] * total2) / new_total

        for mod in group_assignments:
            if group_assignments[mod] == grp1 or group_assignments[mod] == grp2:
                group_assignments[mod] = new_group
                
        group_woe[new_group] = new_woe
        total_dict[new_group] = new_total
            
        existing_groups = set(group_assignments.values())
        for other_group in existing_groups:
            if other_group == new_group:
                continue
            diff = abs(group_woe[new_group] - group_woe[other_group])
            if diff <= woe_threshold:
                heappush(heap, (diff, (new_group, other_group)))

        return group_assignments


def regroup_modalities(df: pd.DataFrame,
                        column: str,
                        target: str,
                        max_modalities: int = 5,
                        verbose: bool = False,
                        woe_threshold: float = 0.2
                    ) -> tuple[pd.DataFrame, str, list]:
    """
    Regroupe les modalités d'une variable catégorielle selon la similarité de leur WoE.
    Retourne : df mis à jour, nom de la nouvelle colonne, liste des groupes.
    """
    iv_origin = calcul_woe_iv(df, column, target)['iv'].sum()
    new_column = f'GROUP_{column}'
    if verbose:
        print(f"Regroupement '{column}' → IV origine={iv_origin:.4f}")

    best_iv, best_grouping, best_k = -np.inf, None, None

    for k in range(2, min(max_modalities, df[column].nunique()) + 1):
        mapping_k = build_grouping(df, column, target, k)
        df_temp = df.assign(**{new_column: df[column].map(mapping_k)})
        iv_k = calcul_woe_iv(df_temp, new_column, target)['iv'].sum()
        if verbose:
            print(f" > k={k} → IV={iv_k:.4f}")
        if iv_k > best_iv:
            best_iv, best_grouping, best_k = iv_k, mapping_k, k


    if best_iv >= iv_origin:
        if verbose:
            print(f"Choix : {best_k} groupes (IV={best_iv:.4f})")
        df[new_column] = df[column].map(best_grouping)
    else:
        if verbose:
            print("Pas de gain → tentative fallback")
    if df[column].nunique() > max_modalities:
        fallback_mapping = fallback_build_grouping(df, column, target, max_modalities, woe_threshold)
        df_temp = df.assign(**{new_column: df[column].map(fallback_mapping)})
        iv_fb = calcul_woe_iv(df_temp, new_column, target)['iv'].sum()
        if iv_fb > iv_origin:
            if verbose:
                print(f"Fallback accepté : {iv_fb:.4f}")
            df[new_column] = df[column].map(fallback_mapping)
        else:
            df[new_column] = df[column]
    else:
        df[new_column] = df[column]


    df[column] = df[new_column].astype('category')
    df.drop(columns=[new_column], inplace=True, errors='ignore')
    return df.copy(), new_column, list(df[column].cat.categories)


def regroup_modalities_from_mappings(df: pd.DataFrame,
                                        old_column: str,
                                        new_column: str,
                                        modalities_grouped: list[str]) -> pd.DataFrame:
    """
    Applique un regroupement de modalités pré-calculé sur un autre DataFrame (validation/test).
    """
    
    modality_map = {mod: group for group in modalities_grouped for mod in group.split(' && ')}
    df[new_column] = df[old_column].map(modality_map)
    df[old_column] = df[new_column].astype('category')
    df.drop(columns=[new_column], inplace=True, errors='ignore')
    
    return df.copy()


def apply_regroup_modalities(df_train: pd.DataFrame,
                            df_val: pd.DataFrame,
                            df_test: pd.DataFrame,
                            columns: list[str],
                            target: str,
                            max_modalities: int = 5,
                            woe_threshold: float = 0.2,
                            verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applique un regroupement supervisé de modalités sur plusieurs colonnes catégorielles, en s'appuyant sur le WOE (Weight of Evidence).

    Cette fonction réalise un regroupement supervisé sur les colonnes catégorielles du DataFrame d'entraînement,
    puis applique les mêmes regroupements aux jeux de validation et de test pour assurer la cohérence du pipeline.

    Étapes principales :
        - Pour chaque colonne catégorielle :
            - Application d’un regroupement supervisé sur df_train via la fonction `regroup_modalities`.
            - Application du mapping obtenu sur df_val et df_test via `regroup_modalities_from_mappings`.
        - Affichage des détails pour chaque regroupement si verbose=True.
   """
    
    print('---')
    print("\t Lancement du regroupement des modalités catégorielles.")
   
    for col in columns:
        df_train, new_column, modalities_grouped = regroup_modalities(
            df=df_train,
            column=col,
            target=target,
            max_modalities=max_modalities,
            woe_threshold=woe_threshold,
            verbose=verbose
        )
        df_val = regroup_modalities_from_mappings(
            df=df_val,
            old_column=col,
            new_column=new_column,
            modalities_grouped=modalities_grouped
        )
        df_test = regroup_modalities_from_mappings(
            df=df_test,
            old_column=col,
            new_column=new_column,
            modalities_grouped=modalities_grouped
        )
    print("\t Regroupement des modalités terminé.")
    print('---')
    
    return df_train, df_val, df_test
