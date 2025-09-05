from scoring.data.importation.requetes_importation import cohort_data, get_products
import scoring.data.connector as conn
from scoring.data.importation.cohorte_new_indicators import build_indicators
from scoring.data.importation.cohorte_preprocessing import prepare_cohort_features, cast_feature_types


class CohorteBuilder:
    def __init__(self, start_date: datetime.date, produits: tuple, lag_prediction: int = 3, cohorte_size: int = 4, config_columns: str = "scoring/config/columns.json"):
        self.start_date: datetime.date = start_date
        self.produits: tuple = produits
        self.lag_prediction: int = lag_prediction
        self.cohorte_size: int = cohorte_size
        self.config_columns = config_columns


    def get_cohortes_system(self) -> pd.DataFrame:
        print(f'\t Lancement de la création des cohortes | Date : {self.start_date} | Taille de cohorte : {self.cohorte_size}.')
        
        DATE_MAX = date.today().replace(day=1) - relativedelta(months=self.lag_prediction)
        DATE_MAX = (DATE_MAX.replace(day=1) + relativedelta(months=1) - timedelta(days=1))
        
        current_date = self.start_date.replace(day=1)
        cohortes = []
        cpt = 1
        
        while (current_date <= DATE_MAX):
            print(f'\t   > Cohorte : {cpt} | Date : {current_date}')
            cohorte = self.get_cohorte(current_date)
            cohortes.append(cohorte)
            current_date += relativedelta(months=self.cohorte_size)
            cpt += 1
        
        if cohortes:
            return pd.concat(cohortes, ignore_index=True)
        else:
            return pd.DataFrame()

        
    def get_cohorte(self, cohort_date: datetime.date) -> pd.DataFrame:
        CNX = conn.sispub()
        query = cohort_data(self.start_date, self.produits, self.lag_prediction)
        data = pd.DataFrame(CNX(query))
        cohorte = self.prepare_cohorte(data, cohort_date)
        return cohorte


    def prepare_cohorte(self, cohorte: pd.DataFrame, cohort_date: datetime.date) -> pd.DataFrame:
        try:
            cohorte = prepare_cohort_features(cohorte) # def prepare_cohort_features(df: pd.DataFrame) -> pd.DataFrame:
            cohorte = build_indicators(cohorte, cohort_date) # def build_indicators(df:pd.DataFrame, date:datetime.date) -> pd.DataFrame:
            cohorte = cast_feature_types(cohorte, self.config_columns) # def cast_feature_types(df: pd.DataFrame, config_path: str)-> pd.DataFrame:
        except Exception as e:
            print(f"Error during cohorte preparation: {e}")
        
        return cohorte if not cohorte.empty else pd.DataFrame()

