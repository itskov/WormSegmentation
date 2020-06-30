import matplotlib.pyplot as plt

from Behavior.Visualizers.PairwiseAnalyses import PairWiseRoi
from Behavior.Visualizers.PairwiseAnalyses import PairWiseProjectionDensity
from Behavior.Visualizers.PairwiseAnalyses import PairWiseSpeedDensity

from Behavior.General.TracksFilter import filterTracksForAnalyses

import numpy as np
import seaborn as sns

def day_summary_plots(exp_pairs, titles, legends, paper=False, show=True):
    number_of_exps = len(exp_pairs)

    fig, axs = plt.subplots(1, number_of_exps)
    if not paper:
        plt.style.use("dark_background")
        sns.set_context("talk")
    else:
        sns.set_context("paper")



    axs = np.atleast_2d(axs)

    for i, exp_pair in enumerate(exp_pairs):

        # ROI plot
        plt.sca(axs[0][i])
        PairWiseRoi('ATR+',  exp_pair[0], 'ATR- (Ctrl)', exp_pair[1], showShow=False, show_count=False, freq=120)
        #plt.title(titles[i])
        plt.legend(fontsize='xx-small', title_fontsize='40')


    #fig.tight_layout()
    #fig.savefig('/home/itskov/Dropbox/dayfigs.png')

    if show:
        plt.show()


def main():
    plots_pairs = []
    plot_legends = []
    plot_titles = []

    exp1 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D120_NO_IAA3x5.avi_12.57.03/exp.npy')[0]
    exp2 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D120_NO_IAA3x5.avi_12.56.12/exp.npy')[0]
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D180_NO_IAA3x5.avi_15.29.04/exp.npy')[0]
    exp2 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D180_NO_IAA3x5.avi_15.28.03/exp.npy')[0]
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D240_NO_IAA3x5.avi_17.28.52/exp.npy')[0]
    exp2 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D240_NO_IAA3x5.avi_17.28.04/exp.npy')[0]
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D300_NO_IAA3x5.avi_20.12.12/exp.npy')[0]
    exp2 = np.load('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D300_NO_IAA3x5.avi_20.11.34/exp.npy')[0]
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')


    day_summary_plots(plots_pairs, plot_legends, plot_titles)

if __name__ == "__main__":
    main()


