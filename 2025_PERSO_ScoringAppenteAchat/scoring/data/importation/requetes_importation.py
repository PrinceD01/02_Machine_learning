
def get_products(date:datetime.date,
                    product_name:str
                ) -> tuple:
    """
    Requête d'importation de la liste des ID des solutions liées au produit à cibler
    Pour plus de contexte : chez Additi, 1 Gamme de produit = plusieurs produits 
                                        et 1 produit = plusieurs solutions
    Donc, cibler 1 produit revient à cibler toutes ses solutions associées.

    Parameters :
        date : date de l'étude - date à laquelle on extrait les données
        produit_name : nom du produit à cibler
        
    Return :
        tuple : la liste des solutions liées au produit à cibler
    """
    import scoring.data.connector as conn

    CNX = conn.sispub()
    DATE_SEUIL_MIN = date.strftime("%Y-%m-%d")
    DATE_SEUIL_MAX = date.today().replace(day=1)
  
    query = f"""
    WITH PRODUITS_FILTERED AS (
        SELECT 
            CA.Id_offre,
            OFFRE.Libelle,
            COUNT(CA.reference) AS NB,
            CASE 
                WHEN (OFFRE.Libelle COLLATE Latin1_General_CS_AS LIKE '%{product_name}%' AND LOWER(OFFRE.Libelle) NOT LIKE '%hors%')
                THEN 1 
                ELSE 0 
            END AS FL_PRODUIT_CIBLE
        FROM FAIT_APUB_PRESTATION_ANALYTIQUE AS CA
        LEFT JOIN (
            SELECT DISTINCT
                Id_offre,
                Libelle
            FROM DIM_OFFRE_PUB
        ) AS OFFRE ON OFFRE.Id_offre = CA.Id_offre
        WHERE CA.Date_facturation >= '{DATE_SEUIL_MIN}' 
            AND CA.Date_facturation <= '{DATE_SEUIL_MAX}'
            AND (CA.Id_etat_prestation IN ('APF', 'VLD') OR (CA.Id_etat_prestation = 'ATT' AND CA.Flag_motif_attente = 2048))
            AND CA.flag_regularisation <> 1 
        GROUP BY CA.Id_offre, OFFRE.Libelle
    )
    
    SELECT Id_offre
    FROM PRODUITS_FILTERED
    WHERE FL_PRODUIT_CIBLE = 1 AND NB > 0
    """
    
    return tuple(pd.DataFrame(CNX(query)))


def cohort_data(date:datetime.date,
                        produits:tuple,
                        lag_prediction:int=3
                    ) -> str:
    """
    Requête d'importation des données d'une cohorte
    
    Parameters :
        date : date de l'étude - date à laquelle on extrait les données
        produits : tuple des ID des produits ciblés (ex: CIBLAD+ ou AD4)
        lag_prediction : délai de prédiction (en mois) - délai pendant lequel on observe et enregistre les achats
        
    Return :
        query : la requête SQL
    """

    # Date de l'étude - date à laquelle on extrait les données
    # date = datetime.strptime(date, "%Y-%m-%d") # date = '2024-06-01'
    DATE_ETUDE = date.strftime("%Y-%m-%d")
    DATE_HEURE_ETUDE = date.strftime("%Y-%m-%d %H:%M:%S")

    # Date de la prédiction
    date_prediction = (date + relativedelta(months=lag_prediction+1)).strftime("%Y-%m-%d") 

    # délai d'observation (en mois)
    LAG_24M = 24
    DATE_PAST_24M = (date - relativedelta(months=LAG_24M)).strftime("%Y-%m-%d")

    # Requête de récupération des données
    query = f"""
        WITH 

        -- on récupère l'opportunité la plus récente de chaque compte créée entre date_creation_compte_1 et date_creation_compte_2
        LAST_OPPORTUNITE AS (
            SELECT 
                tOPPORTUNITE.ID_COMPTE_LIE
                , tOPPORTUNITE.DT_CREATION
                , tOPPORTUNITE.ETAPE
                , tOPPORTUNITE.TYPE_PRODUIT
                , tOPPORTUNITE.MONTANT
                , tOPPORTUNITE.GAIN
                , CAST(tOPPORTUNITE.PROBABILITE AS FLOAT) AS PROBABILITE_NUM
            FROM DIM_APUB_RCP_OPPORTUNITES AS tOPPORTUNITE
            WHERE tOPPORTUNITE.DT_CREATION = (
                SELECT MAX(DT_CREATION)
                FROM DIM_APUB_RCP_OPPORTUNITES
                WHERE ID_COMPTE_LIE = tOPPORTUNITE.ID_COMPTE_LIE
                    AND DT_CREATION <= '{date_prediction}')
        ),

        -- on trie les lignes pour qu'ensuite on sélectionne l'opportunité la plus 'significative' en regardant simplement la probabilité de réalisation de l'opportunité
        LAST_OPPORTUNITE_ORDERED AS (
            SELECT 
                *
                -- On ajoute une colonne d'ordonnement où on utilise RANK() qui permet d'attribuer un numéro de classement aux lignes en fonction de PROBABILITE_NUM. Si plusieurs lignes ont la même valeur, elles reçoivent le même rang.
                , RANK() OVER (
                    PARTITION BY ID_COMPTE_LIE, DT_CREATION 
                    ORDER BY PROBABILITE_NUM DESC
                ) AS _NUM_LIGNE_
            FROM LAST_OPPORTUNITE
        ),

        -- CA filtré sur la période de prédiction et les produits cibles et on identifie les achats de produits CIBLAD après DATE_ETUDE
        CA_FILTERED AS (
            SELECT
                tCA.Id_offre
                , tCA.Id_societe
                , tCA.Id_tiers_facture
                , tCA.reference
                , tCA.Numero_panier
                , tCA.date_valeur
                , tCA.Date_facturation
                , tCA.Date_commande_jour
                , tCA.Id_exploitation
                -- Flag d'appartenance aux produits cibles
                , CASE WHEN tCA.Id_Offre IN {produits} THEN 1 ELSE 0 END AS FL_ACHAT_PRODUIT
            FROM FAIT_APUB_PRESTATION_ANALYTIQUE AS tCA
        
            -- on ne considère que les facturations faites entre DATE_ETUDE et date_prediction
            WHERE tCA.Date_commande_jour >= '{DATE_ETUDE}'
            AND tCA.Date_commande_jour < '{date_prediction}'
            
            -- filtres pour récupérer les factures valides uniquement
            -- AND tCA.flag_validite_facturation = 1 -- obsolète
            AND (tCA.Id_etat_prestation IN ('APF', 'VLD') OR (tCA.Id_etat_prestation = 'ATT' AND tCA.Flag_motif_attente = 2048))
            AND tCA.flag_regularisation <> 1
        ),

        -- on trie les lignes pour qu'ensuite on sélectionne l'achat le plus 'significatif'
            -- L'achat le plus significatif par compte est défini selon la logique suivante :
                -- 1. Priorité aux produits ciblés (FL_ACHAT_PRODUIT=1) | s'il prend au moins une fois le produit ciblé parmi ses achats, on ne considère que cet achat
                -- 2. Plus ancienne date de commande | s'il prend plus d'une fois le produit ciblé, on ne considère que la première commande <=> premier achat
                -- 3. Plus ancienne date de parution | s'il pour la même commande, le produit est livré en plusieurs phases, on ne considère que la premièred date de parution du produit
        CA_FILTERED_ORDERED AS (
            SELECT 
                SS_COMPTE.ID
                , fCA.FL_ACHAT_PRODUIT
                , fCA.date_valeur
                , fCA.Date_facturation
                , fCA.Date_commande_jour
                , fCA.Id_exploitation
                , fCA.Id_societe
                -- on ajoute un numéro de ligne pour chaque achat en fonction du tri fait puis, on ne gardera que la première ligne qui est notre achat significatif
                , ROW_NUMBER() OVER (
                    PARTITION BY SS_COMPTE.ID -- pour diviser le résultat en sous-groupes, un pour chaque ID
                    ORDER BY fCA.FL_ACHAT_PRODUIT DESC, -- 1
                            fCA.Date_commande_jour ASC, -- 2
                            fCA.date_valeur ASC -- 2
                ) AS _NUM_LIGNE_
            FROM CA_FILTERED AS fCA 
            INNER JOIN FAI_APUB_RCP_SOUS_COMPTE AS SS_COMPTE
                ON SS_COMPTE.ID_PCMO = fCA.Id_tiers_facture 
                AND SS_COMPTE.CD_FILIALE = fCA.Id_societe
        ),
        
        -- on ne récupère que les comptes qui ne sont pas Inactif ou Prospect à 'DATE_ETUDE'
        STATUT_FILTERED AS (
            SELECT
                SC.ID_UNIQUE_RCP,
                SC.STATUT_PERE,
                SC.DT_MOIS,
                ROW_NUMBER() OVER (
                    PARTITION BY SC.ID_UNIQUE_RCP, {int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])}
                    ORDER BY SC.DT_MOIS DESC
                ) AS rn,
                {int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])} AS dt_etude
            FROM DIM_APUB_RCP_STATUT_COMPTE SC
            
            WHERE SC.STATUT_PERE NOT IN ('Prospect', 'Inactif')
            AND SC.DT_MOIS < ({int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])})
        )
            
        SELECT
            CONVERT(VARCHAR, '{DATE_ETUDE}', 112) + ' | ' + CAST(COMPTE.ID AS VARCHAR)                  AS _ID_
            , '{DATE_ETUDE}'                                                                            AS _DATE_ETUDE_
            , COMPTE.ID																					AS ID_COMPTE

            -- table FAIT_APUB_PRESTATION_ANALYTIQUE
            , CA.FL_ACHAT_PRODUIT																		AS FL_ACHAT_PRODUIT
            , CA.Date_commande_jour                                                                     AS DT_COMMANDE
            , CA.date_valeur																			AS DT_PARUTION
            , CA.Date_facturation																		AS DT_FACTURATION

            -- table V_REGROUPEMENT_EXPLOITATION_DAF4
            , DAF.lib_daf2                                                                              AS LIB_DAF2
            , DAF.lib_daf3                                                                              AS LIB_DAF3
            , DAF.lib_daf4                                                                              AS REG_DAF4
            , DAF.GAMME_PRODUIT                                                                         AS GAMME_PRODUIT
            
        ---------- 1. CARACTÉRISTIQUES DU COMPTE ----------
            -- Localisation
            , COMPTE.CD_DEPARTEMENT																	    AS DEPARTEMENT
            , COMPTE.CD_POSTAL                                                                          AS CODE_POSTAL
            -- Statut juridique et administratif
            , COMPTE.LB_STATUT_SOCIETE																    AS STATUT_SOCIETE
            , COMPTE.FORME_JURIDIQUE																    AS FORME_JURIDIQUE
            , COMPTE.CD_TYPE_FACTURE                                                                    AS TYPE_FACTURE
            , MODE_PAIEMENT.LB_MODE_DE_PAIEMENT 													    AS MODE_PAIEMENT
            , COMPTE.CD_FREQUENCE_FACTURATION                                                           AS FREQUENCE_FACTURATION
            , COMPTE.CD_DELAI_PAIEMENT                                                                  AS DELAI_PAIEMENT
            , COMPTE.CD_SOURCE_CREATION                                                                 AS SOURCE_CREATION
            , COMPTE.MOTIF_BLOCAGE                                                                      AS MOTIF_BLOCAGE -- to check
            , COMPTE.LB_ASC 			            												    AS MOTIF_ARRET -- to check
            , COMPTE.PROCEDURE_COLLECTIVE                                                               AS PROCEDURE_COLLECTIVE -- to check
            , ST.STATUT_PERE																	        AS STATUT_ACTIVITE -- to check (on a viré les prospects)
            , COMPTE.STATUT_BENEFICIAIRE                                                                AS STATUT_BENEFICIAIRE -- to check
            -- Données financières et taille
            , COMPTE.CA_ETABLISSEMENT																    AS CA_ENTREPRISE
            , COMPTE.NB_EMPLOYES																	    AS NOMBRE_EMPLOYES
            -- Segments et groupes
            , PROFIL_CLIENT.LB_PROFIL_CLIENT                                                            AS PROFIL_CLIENT       
            , COMPTE.SEGMENT                                                                            AS SEGMENT
            , COMPTE.SOUS_SEGMENT                                                                       AS SOUS_SEGMENT
            , COMPTE.LB_SEGMENT_BUMP                                                                    AS SEGMENT_BUMP     
            , COMPTE.STATUT_GROUPEMENT                                                                  AS STATUT_GROUPEMENT
            , COMPTE.GROUPEMENT                                                                         AS GROUPEMENT
            -- Contacts et communication
            , COMPTE.NB_CONTACT_AVEC_EMAIL                                                              AS NB_CONTACT_AVEC_EMAIL
            , COMPTE.NB_CONTACT_AVEC_TEL                                                                AS NB_CONTACT_AVEC_TEL
            , COMPTE.NB_CONTACT_COMPTE                                                                  AS NB_CONTACT_COMPTE
            -- Identifiants et références  
            , COMPTE.RECORD_TYPE_NAME                                                                   AS RECORD_TYPE_NAME
            
            -- Données temporelles
            , COMPTE.DT_CREATION												          				AS DT_ENREGISTREMENT_COMPTE
            , COMPTE.DT_CREATION_SIRET														      		AS DT_CREATION_ENTREPRISE
            , COMPTE.DT_MAJ_CREDITSAFE                                                                  AS DT_MAJ_CREDITSAFE
            , COMPTE.DT_ASC 				            												AS DT_ARRET -- to check | passer en flag selon la date d'appétence avec l'idée du délai de prédiction
            , COMPTE.DT_BLOCAGE                                                                         AS DT_BLOCAGE

            -- table DIM_APUB_RCP_OPPORTUNITES
            , OPPORTUNITE.DT_CREATION                                          	                        AS DT_OPPORTUNITE
            , OPPORTUNITE.ETAPE                                                                         AS ETAPE_OPPORTUNITE
            , OPPORTUNITE.TYPE_PRODUIT                                                                  AS TYPE_PRODUIT_OPPORTUNITE
            , OPPORTUNITE.MONTANT                                                                       AS MONTANT_OPPORTUNITE
            , OPPORTUNITE.GAIN                                                                          AS GAIN_OPPORTUNITE
            , OPPORTUNITE.PROBABILITE                                                                   AS PROBABILITE_OPPORTUNITE
        
            -- table DIM_APUB_RCP_CONTACT_COMPTE
            , CONTACT_COMPTE.NB_CONTACT_DIRECT                                                          AS NB_CONTACT_DIRECT

            -- table DIM_APUB_RCP_STATUT_BENEFICIAIRE
            , STATUT_DETAIL.STATUT_PERE                                                                 AS STATUT_DETAIL_PERE
            , STATUT_DETAIL.TYPE_SUPPORT                                                                AS DETAIL_TYPE_SUPPORT
            , STATUT_DETAIL.TYPE_PRODUIT                                                                AS DETAIL_TYPE_PRODUIT

        FROM FAI_APUB_RCP_COMPTE AS COMPTE
        
        -- Libellés des modes de paiement
        LEFT JOIN (
            SELECT DISTINCT
                CD_MODE_DE_PAIEMENT
                , LB_MODE_DE_PAIEMENT
            FROM DIM_APUB_RCP_PAIEMENT
        ) AS MODE_PAIEMENT ON MODE_PAIEMENT.CD_MODE_DE_PAIEMENT = COMPTE.CD_MODE_DE_PAIEMENT

        -- Libellés des profils clients actifs
        LEFT JOIN (
            SELECT DISTINCT
                CD_PROFIL_CLIENT
                , LB_PROFIL_CLIENT
            FROM DIM_APUB_RCP_PROFIL_CLIENT
            WHERE ACTIF = 1
        ) AS PROFIL_CLIENT ON PROFIL_CLIENT.CD_PROFIL_CLIENT = COMPTE.CD_PROFIL_CLIENT
        
        -- Statut actualisé des comptes
        LEFT JOIN STATUT_FILTERED AS ST 
            ON ST.ID_UNIQUE_RCP = COMPTE.ID 
                AND ST.dt_etude = ({int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])})
                AND ST.rn = 1

        ---------- 2. OPPORTUNITÉS COMMERCIALES ----------
        -- on ajoute les informations sur la dernière opportunité commerciale enregistrée (càd le dernier devis proposé) pour le compte
        LEFT JOIN (
            SELECT 
                ID_COMPTE_LIE
                , DT_CREATION
                , ETAPE
                , PROBABILITE_NUM AS PROBABILITE

                -- on constate qu'une opportunité peut être faite sur la base de plusieurs types de produits différents donc,
                -- on redéfinit le type de produit
                , CASE 
                    -- si un unique type de produit, on le garde
                    WHEN COUNT(DISTINCT TYPE_PRODUIT) = 1 THEN MAX(TYPE_PRODUIT) 
                    -- si on en a plus, on définit selon les cas le type de produit
                    WHEN COUNT(DISTINCT TYPE_PRODUIT) > 1
                        AND (MAX(TYPE_PRODUIT) IN ('2-PRINT', '3-D+P') OR MAX(TYPE_PRODUIT) IN ('1-DIGITAL', '3-D+P') OR MAX(TYPE_PRODUIT) IN ('4-AUTRE', '3-D+P') OR MAX(TYPE_PRODUIT) IN ('2-PRINT', '1-DIGITAL')) 
                    THEN '3-D+P'
                    WHEN COUNT(DISTINCT TYPE_PRODUIT) > 1 
                        AND MAX(TYPE_PRODUIT) IN ('2-PRINT', '4-AUTRE') 
                    THEN '2-PRINT'
                    WHEN COUNT(DISTINCT TYPE_PRODUIT) > 1
                        AND MAX(TYPE_PRODUIT) IN ('1-DIGITAL', '4-AUTRE') 
                    THEN '1-DIGITAL'
                    ELSE MAX(TYPE_PRODUIT) 
                END AS TYPE_PRODUIT

                -- on recalcule le montant et le gain en sommant les montants et les gains des différentes opportunités significatives faites
                , SUM(MONTANT) AS MONTANT
                , SUM(GAIN) AS GAIN
            FROM LAST_OPPORTUNITE_ORDERED
            WHERE _NUM_LIGNE_ = 1 -- pour ne garder pour chaque ID que l' ou les opportunités la plus significative (plus aboutie)
            GROUP BY ID_COMPTE_LIE, DT_CREATION, ETAPE, PROBABILITE_NUM
        ) AS OPPORTUNITE ON COMPTE.ID_SOURCE = OPPORTUNITE.ID_COMPTE_LIE


        ---------- 4. CONTACTS ASSOCIÉS AU COMPTE ----------
        LEFT JOIN (
            SELECT
            ID_COMPTE
            , SUM( COALESCE( DIRECTE,0 ) ) AS NB_CONTACT_DIRECT
            FROM DIM_APUB_RCP_CONTACT_COMPTE
            GROUP BY ID_COMPTE
        ) AS CONTACT_COMPTE ON COMPTE.ID = CONTACT_COMPTE.ID_COMPTE

        
        ---------- 3. DETAILS CONTACTS ASSOCIÉS AU COMPTE ----------
        -- CANCELLED


        ---------- 5. PRODUITS ACHETÉS ----------
        LEFT JOIN (
            SELECT ID
                , FL_ACHAT_PRODUIT
                , date_valeur
                , Date_facturation
                , Date_commande_jour
                , Id_exploitation
                , Id_societe
            FROM CA_FILTERED_ORDERED
            WHERE _NUM_LIGNE_ = 1 -- pour ne garder pour chaque ID que l'achat le plus significatif
        ) AS CA ON CA.ID = COMPTE.ID


        ---------- 6. CLASSIFICATION DES PRODUITS (DAF) ----------
        -- Jointure sur l'exploitation pour le regroupement DAF
        LEFT JOIN (
            SELECT
                Id_exploitation
                , Id_societe
                , Regroupement22
            FROM DIM_REGROUPEMENT_EXPLOITATION_LIGNE_PUB
            WHERE Regroupement22 IS NOT NULL
        ) AS LIGNE_DAF ON CA.Id_exploitation = LIGNE_DAF.Id_exploitation AND CA.Id_societe = LIGNE_DAF.Id_societe

        -- Ajout des libellés DAF et de la classification par gamme
        LEFT JOIN (
            SELECT
                reg_daf4
                , reg_daf3
                , reg_daf2
                , lib_daf2
                , lib_daf3
                , lib_daf4
                , CASE 
                    WHEN reg_daf3 IN ('DDIEDT','DDIMVI','DFTFTW','DDICLF','DDIEXTE','DDIPRT','DDISIP','DSVVIT','DMTDAT') THEN 'DISPLAY'
                    WHEN reg_daf3 IN ('PPCTVM','PNRTVM') THEN 'DIVERTO'
                    WHEN reg_daf3 IN ('DMDEMD') THEN 'EMAILING'
                    WHEN reg_daf3 IN ('DPGEXT','DPGEDT','DPGTEC','DPGMVI','DPGPRT','DPGCLF','DPGSIP') THEN 'PROG BU'
                    WHEN reg_daf3 IN ('DSVCOM') THEN 'CM'
                    WHEN reg_daf3 IN ('DWMAD4') THEN 'AD4'
                    WHEN reg_daf3 IN ('PPCPUB','PNRPUB','PPCSIP') THEN 'PUB CO'
                    WHEN reg_daf3 IN ('DWMREF') THEN 'REF PAYANT'
                    WHEN reg_daf3 IN ('DWMSOC','DDISOC','DSVSOC') THEN 'SOCIAL'
                    WHEN reg_daf3 IN ('PPCESP','PPCDOF','PPCGRT','PPCPSP','PPCDFS','PPCJDN','PNRESP') THEN 'SUPPLEMENT PUB CO'
                    ELSE 'AUTRE'
                END AS GAMME_PRODUIT
            FROM V_REGROUPEMENT_EXPLOITATION_DAF4
        ) AS DAF ON LIGNE_DAF.Regroupement22 = DAF.reg_daf4
        

        ---------- 7. STATUT BENEFICIAIRE ----------
        LEFT JOIN (
            SELECT
                ID_UNIQUE_RCP
                , DT_MOIS
                , STATUT_PERE
                , TYPE_SUPPORT
                , TYPE_PRODUIT
            FROM DIM_APUB_RCP_STATUT_BENEFICIAIRE
            -- on s'assure de prendre le statut bénéficiaire à DATE_ETUDE (format YYYYMM)
            WHERE DT_MOIS = {int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])}
        ) AS STATUT_DETAIL ON COMPTE.ID = STATUT_DETAIL.ID_UNIQUE_RCP

        
        -- Filtre sur les clients à considérer
        WHERE
            COMPTE.DT_CREATION <= '{DATE_HEURE_ETUDE}' -- On ne garde que les comptes créés avant la date d'étude
            AND COMPTE.LB_SOURCE = 'SFDC' -- On ne garde que les comptes de source 'SFDC'
            AND ST.STATUT_PERE NOT IN ('Prospect', 'Client Inactif') AND ST.STATUT_PERE IS NOT NULL -- On exclut les clients prospects et inactifs
            AND COMPTE.CD_BU <> 'UAUT' -- On exclut les comptes de la BU Auto
            AND ( CASE 
                -- On exclut les comptes passés en ASC avant la date de parution du produit cible acheté et ceux sans date renseignée
                WHEN COMPTE.LB_ASC = '1' AND COMPTE.DT_ASC <= CA.date_valeur THEN 1
                WHEN COMPTE.LB_ASC = '1' AND COMPTE.DT_ASC IS NULL THEN 1
                -- On exclut les comptes bloqués pour certains motifs avant la date de parution du produit cible acheté et ceux sans date renseignée
                WHEN COMPTE.MOTIF_BLOCAGE IN ('DBL', 'DOUBLON', 'IMP', 'INA', 'JUD') AND COMPTE.DT_BLOCAGE <= CA.date_valeur THEN 1
                WHEN COMPTE.MOTIF_BLOCAGE IN ('DBL', 'DOUBLON', 'IMP', 'INA', 'JUD') AND COMPTE.DT_BLOCAGE IS NULL THEN 1
                ELSE 0
            END ) = 0
            -- On exclut les comptes sans historique d'achat jusqu'à la date de prédiction => compte à considérer comme inactif sur la période
            AND COMPTE.ID NOT IN (
                SELECT C.ID
                FROM FAI_APUB_RCP_COMPTE AS C
                INNER JOIN FAI_APUB_RCP_SOUS_COMPTE AS SC ON C.ID = SC.ID
                INNER JOIN FAIT_APUB_PRESTATION_ANALYTIQUE AS PA ON PA.Id_tiers_facture = SC.ID_PCMO AND PA.Id_societe = SC.CD_FILIALE
                WHERE 
                    -- Validité de la facturation
                    (PA.Id_etat_prestation IN ('APF', 'VLD') OR (PA.Id_etat_prestation = 'ATT' AND PA.Flag_motif_attente = 2048))
                    AND PA.flag_regularisation <> 1 
                    AND PA.Date_commande_jour < '{date_prediction}'
                    AND PA.Date_commande_jour >= '{DATE_PAST_24M}'
                    -- Validité du compte
                    AND C.DT_CREATION <= '{DATE_HEURE_ETUDE}'
                    AND C.LB_SOURCE = 'SFDC'
                    AND C.STATUT NOT IN ('Prospect', 'Client Inactif')
                    AND C.CD_BU <> 'UAUT'
                GROUP BY C.ID
                HAVING COUNT( PA.reference ) < 1
            )
    """
    return query


def historical_data(date:datetime.date) -> str:
    """
    Requête d'importation de l'historique de facturation des comptes
    
    Parameters :
        date : date de l'étude - date à laquelle on extrait les données

    Return :
        query : la requête SQL
    """
    # Variables de référence
    DATE_ETUDE = date.strftime("%Y-%m-%d")
    LAG_24M = 24
    DATE_PAST_24M = (date - relativedelta(months=LAG_24M)).strftime("%Y-%m-%d")

    
    query = f"""
        WITH -- on ne récupère que les comptes qui ne sont pas Inactif ou Prospect à 'DATE_ETUDE'
            STATUT_FILTERED AS (
                SELECT
                    SC.ID_UNIQUE_RCP,
                    SC.STATUT_PERE,
                    SC.DT_MOIS,
                    ROW_NUMBER() OVER (
                        PARTITION BY SC.ID_UNIQUE_RCP, {int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])}
                        ORDER BY SC.DT_MOIS DESC
                    ) AS rn,
                    {int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])} AS dt_etude
                FROM DIM_APUB_RCP_STATUT_COMPTE SC
                
                WHERE SC.STATUT_PERE NOT IN ('Prospect', 'Inactif')
                AND SC.DT_MOIS <= ({int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])})
            )
            
        SELECT DISTINCT
                COMPTE.ID AS ID_COMPTE,
                CA.Numero_panier AS ID_COMMANDE,
                CA.Date_commande_jour AS DT_COMMANDE,
                CA.reference AS NUM_ORDRE,
                COALESCE( CA.Montant_hors_taxe, 0) AS MONTANT_HT,
                DAF.reg_daf1 AS REG_DAF1,
                DAF.lib_daf1 AS LIB_DAF1,
                DAF.lib_daf2 AS LIB_DAF2,
                DAF.lib_daf3 AS LIB_DAF3,
                DAF.lib_daf4 AS LIB_DAF4,
                CASE 
                    WHEN DAF.reg_daf3 IN ('DDIEDT','DDIMVI','DFTFTW','DDICLF','DDIEXTE','DDIPRT','DDISIP','DSVVIT','DMTDAT') THEN 'DISPLAY'
                    WHEN DAF.reg_daf3 IN ('PPCTVM','PNRTVM') THEN 'DIVERTO'
                    WHEN DAF.reg_daf3 IN ('DMDEMD') THEN 'EMAILING'
                    WHEN DAF.reg_daf3 IN ('DPGEXT','DPGEDT','DPGTEC','DPGMVI','DPGPRT','DPGCLF','DPGSIP') THEN 'PROG BU'
                    WHEN DAF.reg_daf3 IN ('DSVCOM') THEN 'CM'
                    WHEN DAF.reg_daf3 IN ('DWMAD4') THEN 'AD4'
                    WHEN DAF.reg_daf3 IN ('PPCPUB','PNRPUB','PPCSIP') THEN 'PUB CO'
                    WHEN DAF.reg_daf3 IN ('DWMREF') THEN 'REF PAYANT'
                    WHEN DAF.reg_daf3 IN ('DWMSOC','DDISOC','DSVSOC') THEN 'SOCIAL'
                    WHEN DAF.reg_daf3 IN ('PPCESP','PPCDOF','PPCGRT','PPCPSP','PPCDFS','PPCJDN','PNRESP') THEN 'SUPPLEMENT PUB CO'
                    ELSE 'AUTRE'
                END AS GAMME_PRODUIT
            FROM FAI_APUB_RCP_COMPTE AS COMPTE
            LEFT JOIN FAI_APUB_RCP_SOUS_COMPTE AS SSCOMPTE
                ON SSCOMPTE.ID = COMPTE.ID
            LEFT JOIN FAIT_APUB_PRESTATION_ANALYTIQUE AS CA
                ON CA.Id_tiers_facture = SSCOMPTE.ID_PCMO AND CA.Id_societe = SSCOMPTE.CD_FILIALE
            LEFT JOIN DIM_REGROUPEMENT_EXPLOITATION_LIGNE_PUB AS LIGNEDAF
                ON LIGNEDAF.Id_exploitation = CA.Id_exploitation AND LIGNEDAF.Id_societe = CA.Id_societe
            LEFT JOIN V_REGROUPEMENT_EXPLOITATION_DAF4 AS DAF
                ON LIGNEDAF.Regroupement22 = DAF.reg_daf4
                
            -- Statut actualisé des comptes
            LEFT JOIN STATUT_FILTERED AS ST 
                ON ST.ID_UNIQUE_RCP = COMPTE.ID 
                    AND ST.dt_etude = ({int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])})
                    AND ST.rn = 1

            WHERE
                ST.STATUT_PERE NOT IN ('Prospect', 'Client Inactif') AND ST.STATUT_PERE IS NOT NULL -- On exclut les clients prospects et inactifs

                AND (CA.Id_etat_prestation IN ('APF', 'VLD') OR (CA.Id_etat_prestation = 'ATT' AND CA.Flag_motif_attente = 2048))
                AND CA.flag_regularisation <> 1 
                AND CA.Date_commande_jour < '{DATE_ETUDE}'
                AND CA.Date_commande_jour >= '{DATE_PAST_24M}'
                AND DAF.reg_daf2 NOT IN ('DCL','PAN')
    """
    
    return query


def customer_relational_data(date:datetime.date) -> str:
    """
    Requête d'importation de l'historique de relations clients du compte

    Parameters :
        date : date de l'étude - date à laquelle on extrait les données

    Return :
        query : la requête SQL
    """
    # Variables de référence
    DATE_ETUDE = date.strftime("%Y-%m-%d")
    DATE_HEURE_ETUDE = date.strftime("%Y-%m-%d %H:%M:%S")

    LAG_3M = 3
    LAG_6M = 6
    LAG_9M = 9
    LAG_12M = 12
    DATE_PAST_3M = (date - relativedelta(months=LAG_3M)).strftime("%Y-%m-%d")   
    DATE_PAST_6M = (date - relativedelta(months=LAG_6M)).strftime("%Y-%m-%d")   
    DATE_PAST_9M = (date - relativedelta(months=LAG_9M)).strftime("%Y-%m-%d")   
    DATE_PAST_12M = (date - relativedelta(months=LAG_12M)).strftime("%Y-%m-%d") 
    
    query = f"""
        WITH
        -- on ne récupère que les comptes qui ne sont pas Inactif ou Prospect à 'DATE_ETUDE'
            STATUT_FILTERED AS (
                SELECT
                    SC.ID_UNIQUE_RCP,
                    SC.STATUT_PERE,
                    SC.DT_MOIS,
                    ROW_NUMBER() OVER (
                        PARTITION BY SC.ID_UNIQUE_RCP, {int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])}
                        ORDER BY SC.DT_MOIS DESC
                    ) AS rn,
                    {int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])} AS dt_etude
                FROM DIM_APUB_RCP_STATUT_COMPTE SC
                
                WHERE SC.STATUT_PERE NOT IN ('Prospect', 'Inactif')
                AND SC.DT_MOIS <= ({int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])})
            ),
            
        -- Table des activités assimilables aux relations clients
        ACTIVITE_RC AS (
            SELECT DISTINCT
                ACT.ID AS ID_ACTIVITE,
                ACT.ID_COMPTE AS ID_COMPTE_SRC,
                ACT.DT_CREATION AS DT_ACTIVITE,
                ACT.HEURE_CREATION AS DTH_ACTIVITE
            FROM DIM_APUB_RCP_ACTIVITE AS ACT
        
            WHERE
                -- Filtre sur les types d'activités faisant pour les relations clients
                ACT.TYPE IN ('Rdv téléphonique','Appel sortant','Call','Fidélisation','Rdv Physique','Appel entrant','Email','Envoi devis','Fabrication','Meeting','Other','Rappel téléphonique','Réalisation propo','Recouvrement','Relance')
                -- les flag s'arretant à 9 mois on prend un scope de 12 mois
                -- si quand on travaillera sur DT_LAST_RELATION on ne trouve rine dans ce scope on prendra une date de ref éloignée
                AND ACT.DT_CREATION <= '{DATE_ETUDE}'
                AND ACT.DT_CREATION > '{DATE_PAST_12M}' 
        ),
        
        -- Table des commandes valides
        COMMANDE_RC AS (
            SELECT DISTINCT
                COMPTE.ID AS ID_COMPTE,
                CA.Numero_panier AS ID_COMMANDE,
                CA.Date_commande_jour AS DT_COMMANDE,
                CA.reference AS NUM_ORDRE
            FROM FAIT_APUB_PRESTATION_ANALYTIQUE AS CA
            
            INNER JOIN FAI_APUB_RCP_SOUS_COMPTE AS SSCOMPTE
                ON SSCOMPTE.ID_PCMO = CA.Id_tiers_facture
                    AND SSCOMPTE.CD_FILIALE = CA.Id_societe
            INNER JOIN FAI_APUB_RCP_COMPTE AS COMPTE
                ON SSCOMPTE.ID = COMPTE.ID
            
            INNER JOIN DIM_REGROUPEMENT_EXPLOITATION_LIGNE_PUB AS LIGNEDAF
                ON LIGNEDAF.Id_exploitation = CA.Id_exploitation
                    AND LIGNEDAF.Id_societe = CA.Id_societe
            INNER JOIN V_REGROUPEMENT_EXPLOITATION_DAF4 AS DAF
                ON LIGNEDAF.Regroupement22 = DAF.reg_daf4
            
            WHERE
                CA.Date_commande_jour >= '{DATE_PAST_12M}' AND CA.Date_commande_jour < '{DATE_ETUDE}' -- scope
                
                -- Filtre sur les commandes valides
                AND ( CA.Id_etat_prestation IN ( 'APF' , 'VLD' ) 
                    OR ( CA.Id_etat_prestation = 'ATT' AND CA.Flag_motif_attente = 2048 ))
                AND CA.flag_regularisation <> 1
                
                -- Filtre sur les clients valides
                AND COMPTE.DT_CREATION < '{DATE_HEURE_ETUDE}'
                AND COMPTE.LB_SOURCE = 'SFDC'
                AND COMPTE.STATUT NOT IN ('Prospect', 'Client Inactif')
                AND COMPTE.CD_BU <> 'UAUT'
                
                -- Filtre sur les produits à exclure
                AND DAF.reg_daf2 <> 'DCL'
        ),
        
        -- Table des opportunités valides
        OPPORTUNITE_RC AS (
            SELECT DISTINCT
                OPP.ID AS ID_OPPORTUNITE,
                OPP.ID_COMPTE_LIE AS ID_COMPTE_SRC,
                OPP.DT_CREATION AS DT_OPPORTUNITE
            FROM DIM_APUB_RCP_OPPORTUNITES AS OPP
            WHERE OPP.DT_CREATION < '{DATE_ETUDE}'
                AND OPP.DT_CREATION >= '{DATE_PAST_12M}' -- scope
        )
        
        -- Requête finale
        SELECT 
            COMPTE.ID AS ID_COMPTE
            
            -- Flag RC 3 mois
            , CASE 
                WHEN
                    MAX(CASE WHEN ACT.DT_ACTIVITE >= '{DATE_PAST_3M}' THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN CMD.DT_COMMANDE >= '{DATE_PAST_3M}' THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN OPP.DT_OPPORTUNITE >= '{DATE_PAST_3M}' THEN 1 ELSE 0 END) = 1
                THEN 1
                ELSE 0
            END AS FL_RELATION_CLIENT_3M

            -- Flag 6 mois
            , CASE 
                WHEN 
                    MAX(CASE WHEN ACT.DT_ACTIVITE < '{DATE_PAST_3M}' AND ACT.DT_ACTIVITE >= '{DATE_PAST_6M}' THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN CMD.DT_COMMANDE < '{DATE_PAST_3M}' AND CMD.DT_COMMANDE >= '{DATE_PAST_6M}' THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN OPP.DT_OPPORTUNITE > '{DATE_PAST_3M}' AND OPP.DT_OPPORTUNITE >= '{DATE_PAST_6M}' THEN 1 ELSE 0 END) = 1
                THEN 1
                ELSE 0
            END AS FL_RELATION_CLIENT_6M

            -- Flag 9 mois
            , CASE 
                WHEN 
                    MAX(CASE WHEN ACT.DT_ACTIVITE < '{DATE_PAST_6M}' AND ACT.DT_ACTIVITE >= '{DATE_PAST_9M}' THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN CMD.DT_COMMANDE < '{DATE_PAST_6M}' AND CMD.DT_COMMANDE >= '{DATE_PAST_9M}' THEN 1 ELSE 0 END) = 1
                    OR MAX(CASE WHEN OPP.DT_OPPORTUNITE < '{DATE_PAST_6M}' AND OPP.DT_OPPORTUNITE >= '{DATE_PAST_9M}' THEN 1 ELSE 0 END) = 1
                THEN 1
                ELSE 0
            END AS FL_RELATION_CLIENT_9M

            -- Date de dernière relation client
            , MAX( LAST_RC.DT_LAST_RC ) AS DT_DERNIERE_RELATION_CLIENT 

        FROM FAI_APUB_RCP_COMPTE AS COMPTE
        
        LEFT JOIN ACTIVITE_RC AS ACT ON COMPTE.ID_SOURCE = ACT.ID_COMPTE_SRC
        LEFT JOIN COMMANDE_RC AS CMD ON COMPTE.ID = CMD.ID_COMPTE
        LEFT JOIN OPPORTUNITE_RC AS OPP ON COMPTE.ID_SOURCE = OPP.ID_COMPTE_SRC
        -- Statut actualisé des comptes
        LEFT JOIN STATUT_FILTERED AS ST 
            ON ST.ID_UNIQUE_RCP = COMPTE.ID 
                AND ST.dt_etude = ({int(DATE_ETUDE[:4] + DATE_ETUDE[5:7])})
                AND ST.rn = 1

        CROSS APPLY (
            SELECT MAX(MyDate) AS DT_LAST_RC
            FROM (VALUES 
                (COALESCE(ACT.DTH_ACTIVITE, '{DATE_PAST_12M}')),
                (COALESCE(CMD.DT_COMMANDE, '{DATE_PAST_12M}')),
                (COALESCE(OPP.DT_OPPORTUNITE, '{DATE_PAST_12M}'))
            ) AS Dates(MyDate)
        ) AS LAST_RC
        
        WHERE ST.STATUT_PERE NOT IN ('Prospect', 'Client Inactif') AND ST.STATUT_PERE IS NOT NULL -- On exclut les clients prospects et inactifs
        GROUP BY COMPTE.ID
    """
    
    return query