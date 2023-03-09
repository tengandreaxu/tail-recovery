import argparse
import os
import tensorflow as tf
import pandas as pd
from training.trainer import Trainer
from parameters.parameters import Params, Loss, BoostType
from parameters.parameters import DataType
import logging

logging.basicConfig(level=logging.INFO)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", dest="name", type=str, help="name of the result folder name"
    )
    parser.set_defaults(name="LASSO_ROLLING")

    parser.add_argument(
        "--res-dir",
        dest="res_dir",
        type=str,
        help="name of the root result folder name",
    )
    parser.set_defaults(res_dir="res/")

    parser.add_argument(
        "--data-path",
        dest="data_path",
        type=str,
        help="specify if you want to override the data folder path",
        required=True,
    )
    parser.set_defaults(data_path=None)

    parser.add_argument("--std", dest="std", type=float, help="sigma distance")
    parser.set_defaults(std=-2.0)

    parser.add_argument(
        "--lasso-only",
        dest="lasso_only",
        action="store_true",
        help="either run lasso only or all models, default is False",
    )
    parser.set_defaults(lasso_only=False)

    parser.add_argument(
        "--call-only",
        dest="call_only",
        action="store_true",
        help="either use call only or not",
    )
    parser.set_defaults(call_only=False)

    parser.add_argument(
        "--put-only",
        dest="put_only",
        action="store_true",
        help="either use put only or not",
    )
    parser.set_defaults(put_only=False)

    parser.add_argument(
        "--create-name",
        dest="create_name",
        action="store_true",
        help="either craft the result folder name from the parameters used",
    )
    parser.set_defaults(create_name=False)

    parser.add_argument(
        "--nnet-only",
        dest="nnet_only",
        action="store_true",
        help="use neural network only",
    )
    parser.set_defaults(nnet_only=False)

    parser.add_argument(
        "--otm-only",
        dest="otm_only",
        action="store_true",
        help="either use only otm, deep otm only",
    )
    parser.set_defaults(otm_only=False)

    parser.add_argument(
        "--atm-only", dest="atm_only", action="store_true", help="either use only atm"
    )
    parser.set_defaults(atm_only=False)
    parser.add_argument(
        "--itm-only",
        dest="itm_only",
        action="store_true",
        help="either use only itm, deep itm only",
    )
    parser.set_defaults(itm_only=False)

    parser.add_argument(
        "--ols-only",
        dest="ols_only",
        action="store_true",
        help="either use ols only or not",
    )
    parser.set_defaults(ols_only=False)

    parser.add_argument(
        "--otm-atm-only",
        dest="otm_atm_only",
        action="store_true",
        help="either use only otm, atm, deep otm only",
    )
    parser.set_defaults(otm_atm_only=False)

    parser.add_argument(
        "--itm-atm-only",
        dest="itm_atm_only",
        action="store_true",
        help="either use only itm, atm, deep itm only",
    )
    parser.set_defaults(itm_atm_only=False)

    parser.add_argument("--loss", dest="loss", type=int, help="type of loss")
    parser.set_defaults(loss=Loss.EX_GDP_SIGMA)

    parser.add_argument(
        "--has-earnings",
        dest="has_earnings",
        action="store_true",
        help="either the training set has earnings or not",
    )
    parser.set_defaults(has_earnings=False)

    parser.add_argument(
        "--horizon", dest="horizon", type=int, help="prediction horizon "
    )
    parser.set_defaults(horizon=1)
    parser.add_argument(
        "--lag",
        dest="lag",
        action="store_true",
        help="either lag the return or not by 1",
    )
    parser.set_defaults(lag=False)

    args = parser.parse_args()

    if args.otm_only and args.otm_atm_only:
        logging.error("Please specify either otm_only or otm_atm_only, not both")
        exit(1)

    if args.horizon != 1 and args.horizon != 5:
        logging.error("please specify a value in {1, 5}")
    return args


def run_rolling_ols(trainer, par):
    """
    Single rolling:
    for each month we recalibrate
    the models with all available past data
    "expanding window"
    """
    for year in range(2009, 2021):
        for m in range(1, 13):
            if year * 100 + m <= 202008:
                print("run", year, m)

                # update X_train
                par.data.test_cut_month = m
                par.data.test_cut_year = year
                trainer.update_par(par)
                trainer.data.resplit_sample()
                trainer.ols_month_by_month()
            else:
                print("skip", year, m)


def run_lasso_and_nnet(par: Params):
    par.data.ivs = 1
    par.data.volumes = -1
    par.data.open_interests = -1
    par.data.other = -1
    par.data.moments = 1
    par.data.realized = -1

    if args.otm_only:
        par.data.moneyness_only = [-2, -1]
    if args.otm_atm_only:
        par.data.moneyness_only = [-2, -1, 0]
    if args.atm_only:
        par.data.moneyness_only = [0]
    if args.itm_atm_only:
        par.data.moneyness_only = [0, 1, 2]
    if args.itm_only:
        par.data.moneyness_only = [1, 2]

    trainer = Trainer(par)

    trainer.save_par_in_res()
    trainer.load_all_data(
        keep_all=True,
        lag=args.lag,
    )

    for year in range(2009, 2021):
        for m in range(1, 13):
            if year * 100 + m <= 202008:
                print("=" * 47)
                print("{:<15} {:<15}".format(f"year={year}", f"month={m}"))
                print("=" * 47)
                par.data.test_cut_month = m
                par.data.test_cut_year = year
                trainer.update_par(par)
                trainer.data.resplit_sample()

                if not args.lasso_only:
                    # save monthly nnet
                    dir_ = os.path.join(trainer.model.res_dir, "nnet")
                    os.makedirs(dir_, exist_ok=True)
                    save_name = str((year * 100) + m)
                    # Neural Net

                    trainer.train_model()

                    trainer.model.save(dir_, save_name)
                    trainer.exGDP_month_pred()
                    trainer.full_dataset_month_pred()
                if not args.nnet_only:
                    trainer.lasso_month_by_month(
                        save_ols_parameters=save_ols_parameters
                    )

            else:
                print("skip", year, m)


if __name__ == "__main__":
    args = get_args()

    ##################
    # configure the parameters
    ##################
    par = Params()
    par.model.res_dir = args.res_dir
    par.set_name(args)

    save_ols_parameters = True
    if args.loss == 8:
        loss = Loss.MSE_VOLA
        save_ols_parameters = False
    elif args.loss == 1:
        loss = Loss.LOG_LIKE
        save_ols_parameters = False
    else:
        loss = Loss.EX_GDP_SIGMA
    logging.info(f"Result dir={par.name}")

    par.model.learning_rate = 0.0005
    par.model.layers = [64, 32, 16]
    par.model.batch_size = 512
    par.model.nb_normal = 1
    par.model.dropout = 0.4

    # both, call_only or put_only
    par.data.opt_type_only = 0
    if args.call_only:
        par.data.opt_type_only = 1
    if args.put_only:
        par.data.opt_type_only = -1

    if args.put_only and args.call_only:
        logging.error("either call only or put only not both")
        exit(1)

    par.data.data_path = args.data_path
    par.seed = 1
    par.data.t = args.horizon
    par.model.loss = loss
    par.data.prb_sig_dist = args.std

    tf.random.set_seed(par.seed)

    ##################
    # create the trainer and launch the training steps
    ##################

    if (not args.lasso_only and not args.nnet_only) or args.ols_only:
        par.data.ivs = -1
        par.data.volumes = -1
        par.data.open_interests = -1
        par.data.other = -1
        par.data.moments = 1
        par.data.realized = -1
        par.print_values()
        trainer = Trainer(par)
        trainer.save_par_in_res()
        save_columns_dir = os.path.join(args.res_dir, par.name, "ols_columns/")
        os.makedirs(save_columns_dir, exist_ok=True)
        trainer.load_all_data(
            keep_all=True,
            save_columns=save_columns_dir,
            lag=args.lag,
        )
        run_rolling_ols(trainer, par)

    ##################
    # run second par with the lasso
    ##################
    if args.ols_only:
        logging.info("=" * 47)
        logging.info("Running for OLS only exiting...")
        logging.info("=" * 47)
        exit(0)

    run_lasso_and_nnet(par)
