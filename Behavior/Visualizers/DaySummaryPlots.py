import matplotlib.pyplot as plt

from Behavior.Visualizers.PairwiseAnalyses import PairWiseRoi
from Behavior.Visualizers.PairwiseAnalyses import PairWiseProjectionDensity
from Behavior.Visualizers.PairwiseAnalyses import PairWiseSpeedDensity

from Behavior.General.TracksFilter import filterTracksForAnalyses

import numpy as np
import seaborn as sns

def day_summary_plots(exp_pairs, titles, legends):
    number_of_exps = len(exp_pairs)

    plt.style.use("dark_background")
    fig, axs = plt.subplots(number_of_exps, 3)
    sns.set_context("talk")


    axs = np.atleast_2d(axs)

    for i, exp_pair in enumerate(exp_pairs):

        exp_pair[0]._tracks = filterTracksForAnalyses(exp_pair[0]._tracks, minDistance=100)
        exp_pair[1]._tracks = filterTracksForAnalyses(exp_pair[1]._tracks, minDistance=100)

        # ROI plot
        plt.sca(axs[i][0])
        PairWiseRoi('ATR+',  exp_pair[0], 'ATR-', exp_pair[1], showShow=False, show_count=False)
        #plt.title(titles[i])

        # Projection plot
        plt.sca(axs[i][1])
        PairWiseProjectionDensity('ATR+',  exp_pair[0], 'ATR-', exp_pair[1], showShow=False)
        #plt.title(titles[i])

        # Speed plot
        plt.sca(axs[i][2])
        PairWiseSpeedDensity('ATR+',  exp_pair[0], 'ATR-', exp_pair[1], showShow=False)
        #plt.title(titles[i])

    #fig.tight_layout()
    plt.show()


def main():
    plots_pairs = []
    plot_legends = []
    plot_titles = []

    exp1 = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_NO_IAA3x5.avi_11.17.19/exp.npy')[0]
    exp2 = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_NO_ATR_TRAIN_60M_NO_IAA3x5.avi_11.16.35/exp.npy')[0]
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+', 'ATR-'))
    plot_titles.append('0m')

    exp1 = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_D60_NO_IAA3x5.avi_14.24.06/exp.npy')[0]
    exp2 = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_NO_ATR_TRAIN_60M_D60_NO_IAA3x5.avi_14.23.14/exp.npy')[0]
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+', 'ATR-'))
    plot_titles.append('0m')

    exp1 = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_D120_NO_IAA3x5.avi_17.27.06/exp.npy')[0]
    exp2 = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_NO_ATR_TRAIN_60M_D120_NO_IAA3x5.avi_17.26.18/exp.npy')[0]
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+', 'ATR-'))
    plot_titles.append('0m')





    day_summary_plots(plots_pairs, plot_legends, plot_titles)

if __name__ == "__main__":
    main()

