Ce dossier regroup l'nesemble du scoring d'appétence produits développé pour ADDITI.

L'ensemble des informations sur ce projet sont contenues dans la Jira SIPADATA-2603


``` 
├── 2025_PERSO_ScoringAppenteAchat
│   │
│   ├── analyses
│   │   ├── Analyse des comportements d'achat.ipynb                      
│   │   └── Statistiques exploratoires bivariées.ipynb
│   │
│   ├── config
│   │   └── columns.json                                <- spécification de config des colonnes
│   │
│   └── scoring                                                                         
│   │   ├── data                                    <- importation et traitement des données
│   │   │   ├── importation
│   │   │   │   ├── requete_importation.py
│   │   │   │   ├── cohorte_preprocessing.py
│   │   │   │   ├── cohorte_new_indicators.py
│   │   │   │   └── cohorte_class.py
│   │   │   │
│   │   │   ├── prepare_models                          
│   │   │   │   ├── splitting.py
│   │   │   │   ├── discretization.py
│   │   │   │   ├── regroup_modalities.py
│   │   │   │   ├── features_selection.py
│   │   │   │   ├── sampling.py
│   │   │   └── └── data_preparation.py                                                     
│   │   │
│   │   ├── models                                  <- scripts python d'entraînement de modèles
│   │   │   ├── logistic_regression.py
│   │   │   ├── random_forest.py
│   │   │   ├── XGBoost.py
│   │   │   ├── scores_grid.py
│   │   │   ├── evaluate_models.py
│   │   └── └── run_models.py
│   │                                                                                                       
│   ├── outputs                           <- scripts python pour générer des graphiques
│   │   ├── notebooks
│   │   │   ├── Modélisation finale - LR - CIBL'AD.ipynb
│   │   │   ├── Modélisation finale - RF - CIBL'AD.ipynb
│   │   │   ├── Modélisation finale - RF - AD4.ipynb
│   │   │   └── Modélisation finale - RF - La Manchette.ipynb
│   │   │
│   │   ├── reports
│   │   │   └── Présentation projet.pdf  
│   │   │                                                  
│   │   ├── streamlit 
│   │   │   │                                                 
│   │   └── └── evaluate_models.py                                                  
│   │                                                           
│   ├── .gitignore                                  <- fichiers et répertoires que git doit ignorer
│   ├── .gitlab-ci.yml                              <- CI
│   ├── .pre-commit-congig.yaml                     <-
│   │                                                                               
│   ├── .sonar-projet.properties                    <- propriétés sonar par défaut
│   │                                                                   
│   ├── CHANGELOG.md                                <-
│   ├── commitlint.config.js                        <-
│   ├── git-precommit-checks.config.js              <-
│   │                                                                           
│   ├── requirements-dev.txt                        <-
│   ├── requirements.txt                            <-
│   │                                                                       
│   └── setup.py                                    <-
└── README.md                                       <- README propre au projet

```