Ce dossier regroup l'nesemble du scoring d'appétence produits développé pour ADDITI.

L'ensemble des informations sur ce projet sont contenues dans la Jira SIPADATA-2603


``` 
├── 2025_PERSO_ScoringAppenteAchat
│   │                                                               
│   ├── config
│   │   └── columns.json                                <- spécification de config des colonnes
│   │                                                                                                                  
│   ├── analyses
│   │   ├── Analyse des comportements d'achat.ipynb                      
│   │   └── Statistiques exploratoires bivariées.ipynb                                                  
│   │                                                                           
│   └── scoring                                                                         
│   │   ├── data                                    <- importation et traitement des données
│   │   │   ├── requete                                                         
│   │   │   ├── cohorte_new_indicators.py           <- build_features
│   │   │   ├── cohorte_preprocessing.py            <- process features
│   │   │   └── cohorte.py                          <- class cohorte
│   │   │       
│   │   ├── features                                <- scripts python liés au traitement de la donnée
│   │   │   ├── splitting.py                                                        
│   │   │   ├── discretization.py                                               
│   │   │   ├── regrouping modalities.py                                            
│   │   │   ├── features_selection.py                                               
│   │   │   ├── encoding.py                                                             
│   │   │   └── sampling.py                                                         
│   │   │                                                                       
│   │   ├── models                                  <- scripts python d'entraînement de modèles
│   │   │   │                                       et de calcul des prédictions
│   │   │   │                                       intègre déjà la partie MLFlow et Pycaret
│   │   │   ├── train_model.py                                                                  
│   │   │   └── predict_model.py
│   │   │                                                                                                       
│   │   └── visualisation                           <- scripts python pour générer des graphiques
│   │   │   ├── scores_grid.py                                                  
│   │   │   └── evaluate_models.py                                                  
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