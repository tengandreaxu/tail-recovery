import pandas as pd
import itertools
from create_datasets.constants import MATURITY_BUCKETS, MONEYNESS, RETURNS_FOR_STATISTICS

import logging
logging.basicConfig(level=logging.INFO)

class DataPreprocessor(object):
    """
    This class preprocess Raw Orats Data:
        1. Historical quotes (options data)
        2. Dividends
        3. Earningss
        4. Stock Price History
    """

    def __init__(self):
        self.logger = logging.getLogger('DataPreprocessor')
        self.all_moneyness = MONEYNESS
        self.all_maturities = MATURITY_BUCKETS
        self.all_returns = RETURNS_FOR_STATISTICS
        self.option_types = [-1, 1]
        self.iv_types = ['caskiv', 'cbidiv', 'paskiv', 'pbidiv']
        self.volume_columns = self.build_grids_columns_type('volume')
        self.open_interest_columns = self.build_grids_columns_type('open_interest')
        self.iv_columns = self.flatten([self.build_grids_columns_type(x) for x in self.iv_types])
        self.rolling_volume_columns = self.build_grids_columns_type('rolling_20_volume')
        self.rolling_open_interest_columns = self.build_grids_columns_type('rolling_20_open_interest')
        self.rolling_volume_mean_per_year_columns = self.build_grids_columns_type('rolling_mean_volume_per_year')
        self.rolling_open_interest_mean_per_year_columns = self.build_grids_columns_type('rolling_open_interest_mean_per_year')
        self.returns_columns = self.build_return_columns()

        self.atm_iv_columns = self.build_atm_iv_columns()

    def drop_duplicates(self, df: pd.DataFrame, unique_column: str):
        df = df.groupby(unique_column).first()
        return df

    def flatten(self, list2d:list)-> list:
        return  list(itertools.chain(*list2d))

    def build_grids_columns_type(self, type_: str)->list:
        """
        type in ['volume', 'option_interest', 'caskiv', 'paskiv', 'cbidiv', 'pbidiv']
        Build building blocks given the "type" of the column
        """
        columns = []
        if type_ == 'caskiv' or type_ == 'cbidiv':
            option_types = [1]
        elif type_ == 'paskiv' or type_ == 'pbidiv':
            option_types = [-1]
        else:
            option_types = self.option_types

        for moneyness in self.all_moneyness:
            for maturity_bucket in self.all_maturities:
                for option_type in option_types:
                    column = f"{type_}_moneyness={moneyness}_maturity_bucket={maturity_bucket}_option_type={option_type}"
                    columns.append(column)
        return columns
    def build_return_columns(self)->list:
        """
        return the mean returns columns
        """
        return_columns = ['return_t=0']
        for r in self.all_returns:
            column =f"return_t={r}"
            return_columns.append(column)
        return return_columns

    def build_atm_iv_columns(self)-> list:
        """
        returns the atm iv columns for each option type and ask/bid
        """
        atm_ivs = []
        for iv_type in self.iv_types:
            for maturity in self.all_maturities:
                if iv_type == 'caskiv' or iv_type == 'cbidiv':
                    option_type = 1
                else:
                    option_type = -1
                column = f"{iv_type}_moneyness=0.0_maturity_bucket={maturity}_option_type={option_type}"
                atm_ivs.append(column)
        return atm_ivs
