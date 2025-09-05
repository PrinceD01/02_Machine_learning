# Importation des bibliothèques
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_curve



def evaluate_model(model,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    y_pred_proba: np.ndarray,
                    score_grid: pd.DataFrame,
                    model_name: str = "Modèle",
                    threshold_decision: float = 0.5,
                    beta: float = 1.0,
                    lift_prct: int = 10
                )   -> None:
    """
    Évalue un modèle de classification binaire avec visualisation complète.
    """

    def lift_at_top_x(y_true, y_scores, lift_prct=10):
        df = pd.DataFrame({'y_true': y_true, 'y_scores': y_scores})
        df = df.sort_values(by='y_scores', ascending=False)
        top_n = int(len(df) * lift_prct / 100)
        top_df = df.iloc[:top_n]
        return top_df['y_true'].mean() / df['y_true'].mean()

    sns.set(style="whitegrid")
    # 1. Métriques personnalisées
    y_pred_seuil = (y_pred_proba >= threshold_decision).astype(int)
    cm = confusion_matrix(y_test, y_pred_seuil)
    TN, FP, FN, TP = cm.ravel()
    taux_faux_positifs = FP / (FP + TN)
    fbeta = fbeta_score(y_test, y_pred_seuil, beta=beta)
    lift_x = lift_at_top_x(y_test, y_pred_proba, lift_prct=lift_prct)
    precision = precision_score(y_test, y_pred_seuil)
    recall = recall_score(y_test, y_pred_seuil)
    prevalence = y_test.mean()
    lift_global = precision / prevalence
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred_seuil)

    metriques = pd.DataFrame({
        "Lift global": [lift_global],
        "Taux faux positifs (FPR)": [taux_faux_positifs],
        f"F{beta}-score": [fbeta],
    })

    print("\n\t\t > Grille de poids des modalités :")
    for line in score_grid[['Variable', 'Modalité', 'Poids']].to_string().split('\n'): # != 0
        print("\t\t\t", line)
    
    print(f"\n\t\t > Évaluation du modèle: {model_name}")
    for i, row in metriques.round(4).iterrows():
        for line in row.to_string(index=True).split('\n'):
            print("\t\t\t ", line)
            
    # 2. Visualisations
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Courbes - {model_name.upper()}", fontsize=16)

    # 2.1. Courbe de Lift par décile + % TP sur y_pred = 1
    df_lift = pd.DataFrame({"y": y_test, "proba": y_pred_proba, "pred": y_pred_seuil})
    df_lift["decile"] = pd.qcut(df_lift["proba"].rank(method='first'), 10, labels=False, duplicates='drop') + 1  # 1 à 10

    # Calculs
    lift_df = df_lift.groupby("decile").agg(
        taux_positif=('y', 'mean'),
        nb_positifs=('y', 'sum'),
        effectif=('y', 'count'),
        nb_pred_1=('pred', lambda x: (x == 1).sum()),
        nb_TP=('y', lambda x: x.sum())
    ).sort_index(ascending=False)

    ## Taux global et lift
    taux_global = df_lift['y'].mean()
    lift_df["lift"] = lift_df["taux_positif"] / taux_global
    ## Pourcentage de vrais positifs dans les prédictions à 1 (TP / préd = 1)
    lift_df["pct_TP_pred1"] = 100 * lift_df["nb_positifs"] / lift_df["nb_pred_1"].replace(0, np.nan)

    # Tracé Lift
    ax1 = axs[0]
    deciles = range(1, len(lift_df) + 1)
    ax1.plot(deciles, lift_df["lift"], color='darkblue', lw=2.5, marker='o', label=f"Lift@{lift_prct}% = {lift_x:.2f}", zorder=2)
    ax1.hlines(1, 1, 10, colors='gray', linestyles='--', label="Aléatoire (lift = 1)", zorder=1)
    ax1.set_title(f"Lift & TP d'acheteurs - {model_name}")
    ax1.set_xlabel("Déciles (1 = top scores)")
    ax1.set_ylabel("Lift")
    ax1.set_xticks(deciles)
    ax1.set_ylim(0, 10)
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left")

    # Courbe Prct TP / PRED1
    ax2 = ax1.twinx()
    ax2.plot(deciles, lift_df["pct_TP_pred1"], color="darkviolet", lw=2, linestyle='-', marker='x', label=f"Seuil de décision : {threshold_decision}")
    ax2.fill_between(deciles, lift_df["pct_TP_pred1"], color="darkviolet", alpha=0.15)
    ax2.set_ylabel("Prct de bonnes prédictions en acheteur")
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right")

    # 2.2. Courbe Precision-Recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    axs[1].plot(thresholds, precisions[:-1], label="Précision", color='darkblue')
    axs[1].plot(thresholds, recalls[:-1], label="Rappel", color='darkviolet')
    axs[1].set_title(f"Courbe Precision-Recall")
    axs[1].set_xlabel("Seuil de décision")
    axs[1].set_ylabel("Score")
    axs[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # 3. Visualisation des matrices de confusion selon le seuil de décision
    seuils = [0.5, 0.6, 0.7, 0.8]
    fig, axs = plt.subplots(1, len(seuils), figsize=(20, 5))
    fig.suptitle(f"Matrices de confusion - {model_name.upper()}", fontsize=16)
    for i, seuil in enumerate(seuils):
        # Prédictions binaires selon le seuil
        y_pred_seuil = (y_pred_proba >= seuil).astype(int)
        # Calcul matrice de confusion
        cm_seuil = confusion_matrix(y_test, y_pred_seuil)
        # Heatmap matrice confusion
        sns.heatmap(cm_seuil, annot=True, fmt='d', cmap="Blues", ax=axs[i])
        axs[i].set_title(f"Seuil = {seuil}")
        axs[i].set_xlabel("Prédit")
        axs[i].set_ylabel("Réel")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # 4. Importance des variables & modalités
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 ligne, 2 colonnes
    fig.suptitle(f"{model_name.upper()}", fontsize=16)

    ## Importance des variables
    if hasattr(model, 'coef_'):
        importance = pd.Series(np.abs(model.coef_[0]), index=X_test.columns)
    elif hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=X_test.columns)
    else:
        importance = None
        print("Aucune importance de variable accessible pour ce modèle.")

    if importance is not None:
        # Supprimer les variables avec une importance nulle
        importance = importance[importance != 0]
        # Trier décroissant
        importance = importance.sort_values(ascending=False)
        # Tracé
        sns.barplot(y=importance.index, x=importance.values, ax=axes[0], palette="viridis", orient='h')
        axes[0].set_title(f"Importance des variables")
        axes[0].set_xlabel("Importance")
        axes[0].set_ylabel("Variables")
        axes[0].tick_params(axis='y')
        # Etiquettes de valeurs à droite des barres
        for i, v in enumerate(importance.values):
            axes[0].text(v + 0.01 * max(importance.values), i, f"{v:.2f}",
                         va='center', fontsize=9)

    ## Poids par modalité (score_grid)
    df_plot = score_grid.copy()
    df_plot = df_plot[df_plot['Poids'] != 0]  # On supprime les poids nuls
    df_plot['Label'] = df_plot['Variable'].astype(str) + " - " + df_plot['Modalité'].astype(str)  # Étiquette affichée
    df_plot = df_plot.sort_values(by='Poids', ascending=False)  # Tri décroissant
    # Couleurs selon le signe
    colors = df_plot['Poids'].apply(lambda x: 'darkblue' if x > 0 else 'darkviolet')
    # Graphique
    axes[1].barh(df_plot['Label'], df_plot['Poids'], color=colors)
    axes[1].set_title("Poids par modalité")
    axes[1].set_xlabel("Poids")
    axes[1].set_ylabel("Variable - Modalité")
    axes[1].invert_yaxis()  # plus grands en haut
    # Légende manuelle
    legend_elements = [
        Patch(facecolor='darkblue', label='Poids positif'),
        Patch(facecolor='darkviolet', label='Poids négatif')
    ]
    axes[1].legend(handles=legend_elements, title="Signe du poids", loc="best")

    plt.tight_layout()
    plt.show()


    # Fermeture de toutes les figures
    plt.close('all')

