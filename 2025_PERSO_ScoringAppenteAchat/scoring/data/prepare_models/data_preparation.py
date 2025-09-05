# Importation des fonctions nécessaires
from splitting import apply_splitting
from discretization import apply_discretization
from regroup_modalities import apply_regroup_modalities
from features_selection import apply_features_selection
from features_preprocessing import apply_features_preprocessing
from sampling import apply_sampling


class DataPreparationPipeline:
    def __init__(self,
                 target: str = 'FL_ACHAT_PRODUIT',
                 id_col: str = '_ID_',
                 strate: str = 'FL_ACHAT_PRODUIT',
                 max_modalities: int = 5,
                 woe_threshold: float = 0.2,
                 criterion: str = 'BIC',
                 business_cols: list = None,
                 n_splits: int = 5,
                 threshold_corr: float = 0.7,
                 threshold_constant: float = 0.99,
                 nb_max_features: int = 30,
                 sampling_seuil_min: float = 0.5,
                 sampling_mode: str = 'NONE',
                 verbose: bool = False):
        
        self.target: str = target
        self.id_col: str = id_col
        self.strate: str = strate
        self.max_modalities: int = max_modalities
        self.woe_threshold: float = woe_threshold
        self.criterion: str = criterion
        self.business_cols: list | None = business_cols
        self.n_splits: int = n_splits
        self.threshold_corr: float = threshold_corr
        self.threshold_constant: float = threshold_constant
        self.nb_max_features: int = nb_max_features
        self.sampling_seuil_min: float = sampling_seuil_min
        self.sampling_mode: str = sampling_mode
        self.verbose: bool = verbose
        self.dict_encode_mapping: dict = None  # pour stocker le mapping des encodages

        # Définition dynamique du pipeline
        self.pipeline_steps = [
            self.splitting,
            self.discretization,
            self.regroup_modalities,
            self.features_selection,
            self.features_preprocessing,
            self.sampling,
            self.align_columns
        ]


    def fit_transform(self, df: pd.DataFrame):
        data = df
        for step in self.pipeline_steps:
            data = step(data)
        return *data, self.dict_encode_mapping


    # --- Étapes du pipeline ---
    
    def splitting(self, df) -> tuple:
        self.df_train, self.df_val, self.df_test = apply_splitting(df=df, target=self.target)
        return (self.df_train, self.df_val, self.df_test)

    def discretization(self, data) -> tuple:
        df_train, df_val, df_test = data
        quantitative_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        df_train, df_val, df_test = apply_discretization(
            df_train, df_val, df_test,
            columns=quantitative_cols,
            target=self.target,
            max_modalities=self.max_modalities,
            verbose=self.verbose
        )
        return (df_train, df_val, df_test)

    def regroup_modalities(self, data) -> tuple:
        df_train, df_val, df_test = data
        nominal_cols = [col for col in df_train.columns if is_categorical_dtype(df_train[col]) and not df_train[col].cat.ordered]
        df_train, df_val, df_test = apply_regroup_modalities(
            df_train=df_train, df_val=df_val, df_test=df_test,
            columns=nominal_cols,
            target=self.target,
            max_modalities=self.max_modalities,
            woe_threshold=self.woe_threshold,
            verbose=self.verbose
        )
        return (df_train, df_val, df_test)

    def features_selection(self, data) -> tuple:
        df_train, df_val, df_test = data
        business_features = list(set([self.strate] + self.business_cols))
        df_train, df_val, df_test = apply_features_selection(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            business_features=business_features,
            target=self.target,
            criterion=self.criterion,
            nb_max_features=self.nb_max_features,
            n_splits=self.n_splits,
            threshold_corr=self.threshold_corr,
            threshold_constant=self.threshold_constant,
            verbose=self.verbose
        )
        return (df_train, df_val, df_test)

    def features_preprocessing(self, data) -> tuple:
        df_train, df_val, df_test = data
        binary_cols = df_train.select_dtypes(include='bool').columns.tolist()
        quantitative_cols = [col for col in df_train.select_dtypes(include=[np.number]).columns if col not in binary_cols]

        ordinal_cols = []
        order_dict = {}
        for col in df_train.columns:
            if is_categorical_dtype(df_train[col]) and df_train[col].cat.ordered:
                ordinal_cols.append(col)
                order_dict[col] = df_train[col].cat.categories.tolist()

        cols_to_exclude = set(binary_cols) | set(quantitative_cols) | set(ordinal_cols)
        nominal_cols = [col for col in df_train.columns if col not in cols_to_exclude and col != self.id_col]

        df_train, df_val, df_test, self.dict_encode_mapping = apply_features_preprocessing(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            quanti_cols=quantitative_cols,
            nominal_cols=nominal_cols,
            ordinal_cols=ordinal_cols,
            ordering_dict=order_dict,
            verbose=self.verbose
        )
        return (df_train, df_val, df_test)

    def sampling(self, data) -> tuple:
        df_train, df_val, df_test = data
        df_train = apply_sampling(
            df=df_train,
            strate=self.strate,
            sampling_seuil_min=self.sampling_seuil_min,
            sampling_mode=self.sampling_mode
        )
        return (df_train, df_val, df_test)

    def align_columns(self, data) -> tuple:
        df_train, df_val, df_test = data
        cols_to_keep = df_train.columns
        df_val = df_val.reindex(columns=cols_to_keep, fill_value=0)
        df_test = df_test.reindex(columns=cols_to_keep, fill_value=0)
        return (df_train, df_val, df_test)
