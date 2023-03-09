import os
import itertools
from enum import Enum
import numpy as np
import pandas as pd

##################
# params Enum
##################

SEED = 12345


class Optimizer(Enum):
    SGD = 1
    SGS_DECAY = 2
    ADAM = 3
    RMS_PROP = 4


class Loss(Enum):
    LOG_LIKE = 1
    LOG_LIKE_RESTRICTED = 2
    LOG_LIKE_PARETO = 3
    MAE = 4
    MSE = 5
    SIGN = 6
    R2 = 7
    MSE_VOLA = 8
    PRB_QUANTILE = 9
    PARETO_FIX_SIG_DIST = 10
    PARETO_SIG_DIST = 11
    PARETO_FIX_U = 12
    PRB_SIGMA_DIST = 13
    PARETO_ONLY_SIGMA = 14
    PARETO_ONLY_QUANTILE = 15
    EX_GDP_SIGMA = 16


class DataType(Enum):
    SIMULATED = 1
    Real1 = 2


class BoostType(Enum):
    NONE = 1
    UNIFORM = 2
    NEXT_DOOR_RETURN = 3
    EXTREME_QUANTILE = 4
    NEGATIVE_DOUBLE = 5
    NORMAL_HEURISTIC = 6
    REDUNDANT = 7


##################
# params classes
##################


class ParamsData:
    def __init__(self):
        self.data_type = DataType.Real1
        self.data_path = None
        self.t = 5
        self.test_cut_year = 2018
        self.test_cut_month = -1
        self.test_nb_year = 20

        self.clip_up = 3.0
        self.clip_down = -3.0

        self.ivs = 0
        self.open_interests = 0
        self.realized = -1
        self.moments = -1
        self.other = 0
        self.volumes = 0

        self.moneyness_only = (
            []
        )  # empty list means all. Else we keep only moneyness in the list
        self.opt_type_only = (
            0  # 0 means all, -1 mean keeps only put, +1 means keeps only call
        )
        # either 'bid' or 'ask' or None
        self.iv_type_only = None
        self.lag_predictor = 0

        #####################
        # Data boosting
        #####################
        self.data_boost = False
        self.data_boost_indexes = np.array([])
        # par the PRB SIGMA DIST loss
        self.prb_sig_dist = (
            -1.0
        )  # the sigma distance to be classified into one category or another
        self.prb_sig_est_window = (
            252  # the estimation window done to estimate this sigma
        )
        self.loss = Loss.EX_GDP_SIGMA
        self.prb_quantile = 0.9
        self.data_path = ""


class ParamsModels:
    def __init__(self):

        self.save_dir = "./model_save/"
        self.log_dir = "./model_log/"
        self.res_dir = "./res"

        self.normalize = False
        self.E = 50

        self.layers = [64, 32, 16]
        self.batch_size = 512
        self.activation = "swish"
        self.opti = Optimizer.ADAM
        self.loss = Loss.EX_GDP_SIGMA
        self.learning_rate = 0.001
        self.validation_split = 0.05

        self.nb_normal = 1

        self.boost = BoostType.NONE

        self.dropout = 0.0
        self.min_std = 0.001


class Params:
    def __init__(self):
        self.name = ""
        self.seed = SEED
        self.model = ParamsModels()
        self.data = ParamsData()

        self.process = None

    def set_name(self, args):
        name = args.name

        if args.create_name:
            datasize = args.data_path.split("/")[4]
            options = "all"
            iv_type = "all"
            moneyness = "all"
            if args.otm_only:
                moneyness = "otm_only"
            if args.otm_atm_only:
                moneyness = "otm_atm_only"
            if args.call_only:
                options = "call_only"
            if args.put_only:
                options = "put_only"
            if args.atm_only:
                moneyness = "atm_only"
            if args.itm_only:
                moneyness = "itm_only"
            if args.itm_atm_only:
                moneyness = "itm_atm_only"

            name = f"lag={args.lag}_horizon={args.horizon}_earnings={args.has_earnings}_moneyness={moneyness}_std={args.std}_data={datasize}_options={options}_iv_type={iv_type}"

        self.name = name

    def print_values(self):
        """
        Print all parameters used in the model
        """
        for key, v in self.__dict__.items():
            try:
                print("########", key, "########", flush=True)
                for key2, vv in v.__dict__.items():
                    print(key2, ":", vv, flush=True)
            except:
                print(v, flush=True)

    def save(self, save_dir: str, file_name: str = "parameters.p"):
        # simple save function that allows loading of deprecated parameters object
        df = pd.DataFrame(columns=["key", "value"])

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    temp = pd.DataFrame(
                        data=[str(key) + "_" + str(key2), vv], index=["key", "value"]
                    ).T
                    df = df.append(temp)

            except:
                temp = pd.DataFrame(data=[key, v], index=["key", "value"]).T
                df = df.append(temp)

        file_ = os.path.join(save_dir, file_name)

        df.to_pickle(file_, protocol=4)

    def load(self, load_dir: str, file_name="parameters.p"):
        # simple load function that allows loading of deprecated parameters object

        df = pd.read_pickle(os.path.join(load_dir, file_name))

        # First check if this is an old pickle version, if so transform it into a df
        if type(df) != pd.DataFrame:
            loaded_par = df
            df = pd.DataFrame(columns=["key", "value"])
            for key, v in loaded_par.__dict__.items():
                try:
                    for key2, vv in v.__dict__.items():
                        temp = pd.DataFrame(
                            data=[str(key) + "_" + str(key2), vv],
                            index=["key", "value"],
                        ).T
                        df = df.append(temp)

                except:
                    temp = pd.DataFrame(data=[key, v], index=["key", "value"]).T
                    df = df.append(temp)

        no_old_version_bug = True

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    t = df.loc[df["key"] == str(key) + "_" + str(key2), "value"]
                    if t.shape[0] == 1:
                        tt = t.values[0]
                        self.__dict__[key].__dict__[key2] = tt
                    else:
                        if no_old_version_bug:
                            no_old_version_bug = False
                            print(
                                "#### Loaded parameters object is depreceated, default version will be used"
                            )
                        print(
                            "Parameter",
                            str(key) + "." + str(key2),
                            "not found, using default: ",
                            self.__dict__[key].__dict__[key2],
                        )
            except:
                t = df.loc[df["key"] == str(key), "value"]
                if t.shape[0] == 1:
                    tt = t.values[0]
                    self.__dict__[key] = tt
                else:
                    if no_old_version_bug:
                        no_old_version_bug = False
                        print(
                            "#### Loaded parameters object is deprecated, default version will be used"
                        )
                    print(
                        "Parameter",
                        str(key),
                        "not found, using default: ",
                        self.__dict__[key],
                    )
