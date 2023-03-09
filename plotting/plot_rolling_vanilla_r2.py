import argparse
import os
import matplotlib.pyplot as plt
from plot_function import pylab

from analysis_rolling import load_all_pickles_in_folder
from trainer import Trainer
from parameters import Params

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--results', nargs="+", default=[], required=True, \
        dest='results', help='the list of results to plot')
    
    parser.add_argument('--names', nargs="+", dest='names', \
        help='the list of labels to show on the plot')

    parser.add_argument('--output-dir', dest='output_dir', type=str)

    args = parser.parse_args()

    results = dict()

    for result, name in zip(args.results, args. names):
        
        try:
            results[name] = load_all_pickles_in_folder(result)
            logging.info(f'Loaded name: {name}, result: {result}')
        except:
            pass

    trainer = Trainer(Params())

    to_plot = dict()
    for result, color in zip(results.keys(), ['brown', 'black', 'green']):
        df = results[result]
        rolling = trainer.compute_rolling_r2(df, use_normal_r2=True)
        plt.plot(rolling.index, rolling.values, color=color,label=result)
    
    file_name = 'vanilla_r2.png'
    if args.output_dir:

        file_name = os.path.join(args.output_dir, file_name)
    plt.legend()
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    plt.close()


