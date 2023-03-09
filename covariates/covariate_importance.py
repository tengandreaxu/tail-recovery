import os
import argparse
import tensorflow as tf
from parameters.parameters import Params, Loss, BoostType, DataType
from training.trainer import Trainer

import logging

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    """
    We want to understand the covariate importance,
    by using the same methodology used in Figure 4
    in Empirical Asset Pricing via Machine Learning, 2019,
    Bryan Kelly.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--res-folder",
        dest="res_folder",
        type=str,
        help="name of the result folder name",
    )
    parser.set_defaults(res_folder="LASSO_ROLLING")

    parser.add_argument("--name", dest="name", type=str, help="result folder name")
    parser.set_defaults(name=None)
    parser.add_argument(
        "--data-path",
        dest="data_path",
        type=str,
        help="specify if you want to override the data folder path",
    )
    parser.set_defaults(data_path=None)

    parser.add_argument("--std", dest="std", type=float, help="sigma distance")
    parser.set_defaults(std=-2.0)

    parser.add_argument(
        "--horizon", dest="horizon", type=int, help="prediction horizon "
    )
    parser.set_defaults(horizon=5)

    parser.add_argument(
        "--lag",
        dest="lag",
        action="store_true",
        help="either lag the return or not by 1",
    )
    parser.set_defaults(lag=True)

    parser.add_argument(
        "--quantile",
        dest="quantile",
        type=float,
        help="quantile cut for the cross section threshold",
    )
    parser.set_defaults(quantile=0)

    parser.add_argument(
        "--predictions", dest="predictions", type=str, help="the predictions res folder"
    )
    parser.set_defaults(predictions=os.environ["RES_DIR"])

    parser.add_argument("--loss", dest="loss", type=int, help="which loss type to use")
    parser.set_defaults(loss=0)

    args = parser.parse_args()

    ##################
    # configure the parameters
    ##################
    par = Params()
    par.model.res_dir = args.predictions
    if args.name:
        name = args.name
    else:
        name = args.res_folder.split("/")[1]

    logging.info(f"FIX_NAME={name}")
    par.name = name
    par.model.learning_rate = 0.0005
    par.model.layers = [64, 32, 16]
    par.model.batch_size = 512
    par.model.nb_normal = 3
    par.model.dropout = 0.4

    # OTM ATM only
    par.data.moneyness_only = [-2, -1, 0]
    # both, call_only or put_only
    par.data.opt_type_only = 0

    if args.data_path:
        par.data.data_path = args.data_path
    par.seed = 1

    loss = Loss.EX_GDP_SIGMA

    ##################
    # Configure paramter grid
    ##################

    par.data.data_path = args.data_path
    par.seed = 1
    par.data.t = args.horizon
    par.model.loss = loss
    par.data.ivs = 1
    par.data.volumes = -1
    par.data.prb_sig_dist = args.std

    # par.data.data_path = 'data/'
    par.data.open_interests = par.data.volumes

    # np.random.seed(par.seed)
    tf.random.set_seed(par.seed)

    par.data.ivs = 1
    par.data.volumes = -1
    par.data.open_interests = -1
    par.data.other = -1
    par.data.moments = 1
    par.data.realized = -1

    trainer = Trainer(par)
    trainer.load_all_data(keep_all=True, lag=args.lag)

    covariates = trainer.data.columns

    for covariate in covariates:
        trainer.load_all_data(
            keep_all=True,
            lag=args.lag,
            zero_column=covariate,
            quantile_threshold=args.quantile,
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

                    save_name = str((year * 100) + m)
                    saved_model = os.path.join(
                        args.predictions, name, "nnet", save_name
                    )
                    trainer.model.model = tf.keras.models.load_model(
                        saved_model, compile=False
                    )
                    trainer.exGDP_month_pred(
                        overwrite_dir=f"covariates_importance/{covariate}/"
                    )

                else:
                    print("skip", year, m)
