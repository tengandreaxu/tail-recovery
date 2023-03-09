import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Plotter:
    """the plotter takes care of any plotting function"""

    def __init__(
        self,
        axes_labelsize: Optional[int] = 14,
        axes_labelweight: Optional[str] = "bold",
        xtick_labelsize: Optional[int] = 12,
        ytick_labelsize: Optional[int] = 12,
        axes_titlesize: Optional[int] = 12,
    ):
        self.LINE_STYLE: list = [
            "solid",
            "dotted",
            "dashed",
            "dashdot",
            "dashed",
            "solid",
            "dotted",
        ]
        self.COLOR: list = [
            "black",
            "brown",
            "grey",
            "lightsalmon",
            "sienna",
            "navy",
            "darkgreen",
            "dimgray",
            "silver",
            "rosybrown",
            "darkred",
            "red",
            "tomato",
            "peru",
        ]
        pylab.rcParams.update(
            {
                "axes.labelsize": axes_labelsize,
                "axes.labelweight": axes_labelweight,
                "xtick.labelsize": xtick_labelsize,
                "ytick.labelsize": ytick_labelsize,
                "axes.titlesize": axes_titlesize,
            }
        )

    def plot_histogram_with_curve(
        self,
        hist: list,
        curve: Tuple[list, list],
        bins: int,
        density: bool,
        file_name: str,
    ):
        """Plots an histogram and a curve on top of it"""
        plt.hist(hist, bins=bins, density=density)
        plt.title("in sample")
        plt.plot(curve[0], curve[1])
        plt.savefig(file_name)
        plt.close()

    def plot_multiple_lines(
        self,
        dfs: list,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        save_name: str,
        labels: list,
        colors: list,
        linestyles: list,
        ylim=None,
    ):
        """
        Given a list of dfs, plot stated x,y

        """
        if not colors:
            colors = self.COLOR
        if not linestyles:
            linestyles = self.LINE_STYLE

        for df, label, color, linestyle in zip(dfs, labels, colors, linestyles):
            if x == "index":
                plt.plot(df.index, df[y], label=label, linestyle=linestyle, color=color)
            else:
                plt.plot(df[x], df[y], label=label, linestyle=linestyle, color=color)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        if ylim:
            plt.ylim(ylim)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()

    def plot_single_curve(
        self,
        x,
        y,
        title: str,
        ylabel: str,
        xlabel: str,
        grid: bool,
        save_path: str,
        color: str,
        linestyle: str,
        label: str,
        marker=None,
        ylim=None,
    ):
        """
        This function plots and saves a single curve, it can be tuned to handle
        multiple curves - in a for loop for example.
        """
        plt.plot(x, y, color=color, linestyle=linestyle, label=label, marker=marker)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(grid)
        if ylim:
            plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_multiple_lines_from_columns(
        self,
        df: pd.DataFrame,
        title: str,
        xlabel: str,
        ylabel: str,
        save_name: str,
        output_dir: str,
        marker=None,
    ):

        """
        Useful when given a df which represents a time series
        we want to plot all columns
        """
        i = 0
        for column in df.columns:
            color = self.COLOR[i]
            df.loc[:, column].plot(color=color, label=column, marker=marker)
            i += 1
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.legend()
        plt.title(title)
        plt.grid()
        plt.savefig(output_dir + save_name)
        plt.close()

    def plot_multiple_lines_given_row_and_columns(
        self,
        df: pd.DataFrame,
        row: str,
        columns: list,
        labels: list,
        colors: list,
        linestyles: list,
        title: str,
        xlabel: str,
        ylabel: str,
        save_name: str,
        format_dates: Optional[bool] = False,
    ):

        """
        Useful when given a df which represents a time series
        we want to plot a subset of columns
        """

        for column, label, color, linestyle in zip(columns, labels, colors, linestyles):
            df[[row, column]].plot(color=color, label=label, linestyle=linestyle)

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.legend()
        plt.title(title)
        plt.grid()
        if format_dates:
            ax = plt.gca()
            self.format_datetime_into_years(ax)
        plt.savefig(save_name)
        plt.close()

    def format_datetime_into_years(self, ax):
        ## setting the x axis format
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    def remove_scientific_notation_yaxis(plt):
        plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)

    def scatter_plot(
        self,
        x,
        y,
        title: str,
        ylabel: str,
        xlabel: str,
        grid: bool,
        save_path: str,
        color: str,
        marker=None,
    ):
        """
        This function plots and saves a single curve, it can be tuned to handle
        multiple curves - in a for loop for example.
        """
        plt.scatter(x, y, color=color, marker=marker)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(grid)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_histogram(
        self,
        df: pd.Series,
        xlabel: str,
        ylabel: str,
        grid: bool,
        save_path: str,
        color: str,
        linestyle: str,
        label: str,
        bins: int,
    ):
        plt.hist(df, bins=bins, density=True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(grid)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def get_xticks(self, index):
        x = math.ceil(len(index) / 25)
        return index[0::x]

    def rotate_xticks_labels(self, ax, degrees: int):
        # rotate dates
        for tick in ax.get_xticklabels():
            tick.set_rotation(degrees)
