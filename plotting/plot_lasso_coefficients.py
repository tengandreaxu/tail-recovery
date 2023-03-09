import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from plot_function import pylab, params

def smooth_months(df: pd.DataFrame):

    df = df.reset_index()
    output = []
    i = 0
    for index, row in df.iterrows():
        start = i - 12
        start = max(0, start)

        sub_df = df[start:(i+1)] # non inclusive
        output.append(
            {
                'date': df.iloc[i]['index'], 
                'coef': sub_df.coef.sum()/ sub_df.shape[0]
            }
        )

        i += 1
    df = pd.DataFrame(output)
    df.index = df.date
    df.pop('date')
    return df

def plot_curves(
    dfs: list,
    y: str,
    title: str, 
    ylabel: str, 
    xlabel: str, 
    grid: bool,\
    save_path: str, 
    colors: list, 
    linestyles: list, 
    labels: list,
):
    """
    """
    for df, color, linestyle, label in zip(dfs, colors, linestyles, labels):

        plt.plot(
            df.index, 
            df[y], 
            color=color,
            linestyle=linestyle, 
            label=label, 
        )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(grid)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    """
    Plot history of rolling lasso betas
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--is-pos', action='store_true', dest='is_pos',\
        help='either the results are for the positive tail or not')
    parser.set_defaults(is_pos=False)

    parser.add_argument('--data-folder', type=str, dest='data_folder',\
        help='lasso betas folder', required=True)

    args = parser.parse_args()
    files = os.listdir(args.data_folder)
    files.sort()

    puts_plot = dict()
    calls_plot = dict()
    historicals_plot = dict()
    for file_ in files:
        df = pd.read_csv(os.path.join(args.data_folder, file_))

        calls = df[df['columns'].str.startswith('c')]
        puts = df[df['columns'].str.startswith('p')]
        historical = df[df['columns'].str.startswith('rolling')]

        puts_plot[file_] = puts.coef.sum()
        calls_plot[file_] = calls.coef.sum()
        historical = historical[(historical.coef < 1) & (historical.coef > -1)]
        historicals_plot[file_] = historical.coef.sum()

    calls = pd.DataFrame.from_dict(calls_plot, orient='index', columns=['coef'])
    puts = pd.DataFrame.from_dict(puts_plot, orient='index', columns=['coef'])
    historical = pd.DataFrame.from_dict(historicals_plot, orient='index', columns=['coef'])

    calls.index = pd.to_datetime(calls.index, format='%Y%m')
    puts.index = pd.to_datetime(puts.index, format='%Y%m')
    historical.index = pd.to_datetime(historical.index, format='%Y%m')

    calls = smooth_months(calls)
    puts = smooth_months(puts)
    historical = smooth_months(historical)

    output_folder =  '/'.join(args.data_folder.split('/')[:-1])

    plot_curves(
        dfs=[calls, puts, historical],
        y='coef',
        title='',
        ylabel='Coefficient',
        xlabel='Year',
        grid=True,
        save_path=os.path.join(output_folder, 'lasso_betas.png'),
        colors=['black', 'brown', 'grey'],
        linestyles=['solid', 'dotted', 'dashed'],
        labels=['Calls', 'Puts', 'Historical']
    )