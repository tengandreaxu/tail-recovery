import os
import argparse
import tensorflow as tf
import numpy as np

from parameters.parameters import Params
from training.trainer import Trainer
from util.dataframes_handling import load_all_pickles_in_folder

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", "-fn", type=str, dest="name", help="res directory fixed name"
    )
    parser.set_defaults(name="LASSO_ROLLING")

    parser.add_argument("--lag", action="store_true", dest="lag")
    parser.set_defaults(lag=False)

    parser.add_argument("--is-pos", action="store_true", dest="is_pos")
    parser.set_defaults(is_pos=False)

    parser.add_argument(
        "--is-small",
        action="store_true",
        dest="is_small",
        help="either is the small dataframe or not",
    )

    parser.add_argument(
        "--res-dir", dest="res_dir", type=str, help="the root result dir"
    )
    parser.set_defaults(res_dir=os.environ["RES_DIR"])

    args = parser.parse_args()

    par = Params()
    par.name = args.name
    par.load(args.res_dir + args.name)

    np.random.seed(par.seed)
    tf.random.set_seed(par.seed)

    trainer = Trainer(par)
    trainer.model.res_dir = args.res_dir + args.name
    # load DNN predictions
    final = load_all_pickles_in_folder(trainer.model.res_dir + "/rolling_pred/")

    lasso_folder = os.path.join(trainer.model.res_dir, "rolling_lasso")

    if os.path.exists(lasso_folder):
        lasso_df = load_all_pickles_in_folder(lasso_folder)
    else:
        lasso_df = None

    ols_folder = os.path.join(trainer.model.res_dir, "rolling_ols")

    if os.path.exists(ols_folder):
        ols_df = load_all_pickles_in_folder(ols_folder)
    else:
        ols_df = None
    breakpoint()
    logging.info(f"Results in {args.name}")

    trainer.analyze_exGDP(
        final,
        lasso_df=lasso_df,
        ols_df=ols_df,
        with_lag=args.lag,
        is_pos=args.is_pos,
        is_small=args.is_small,
    )
