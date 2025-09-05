
├── requirements.txt                            <-
│
├── config                                      <- répertoire de config du projet
│   └── columns.json                            <- spécification de config des colonnes
│
├── data                                        <- répertoire de stockage des données
│   └── customer_data.csv                               
│           
├── outputs
│   ├── streamlit
│   │   └── streamlit_app.py
│   └── visualisation
│
├── src                                                                        
│   ├── processing                                <- scripts python liés au traitement de la donnée
│   │   ├── splitting.py                                                        
│   │   ├── discretization.py                                               
│   │   ├── features_selection.py                                               
│   │   ├── encoding.py                                                             
│   │   └── sampling.py                                                         
│   ├── models                                  <- scripts python d'entraînement de modèles et de calcul des prédictions
│   │   ├── evaluate_models.py                                                                  
│   │   ├── scores_grid.py    
│   │   ├── logistic_regression.py                                                                  
│   │   └── random_forest.py                                                                                            
│   │
│   └── main.py
└── README.md                                       <- README propre au projet
