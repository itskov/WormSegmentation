import matplotlib.pyplot as plt
import pandas as pd

from Behavior.Visualizers.PairwiseAnalyses import PairWiseRoi_
from Behavior.Tools.Artifacts import Artifacts

import numpy as np
import seaborn as sns

def day_summary_plots(exp_pairs, titles, legends, paper=False, show=True, output_file=None):
    number_of_exps = len(exp_pairs)

    fig, axs = plt.subplots(2, int(np.ceil(number_of_exps / 2)), figsize=(16, 4))
    plt.subplots_adjust(wspace=0.3, top=0.93, hspace=0.51)

    if not paper:
        plt.style.use("dark_background")
        sns.set_context("talk")
    else:
        sns.set_context("paper")


    axs = np.atleast_2d(axs)

    df = pd.DataFrame({'Time' : [], 'ATR+' : [], 'ATR-' : [],'Type' : []})

    for i, exp_pair in enumerate(exp_pairs):

        # ROI plot
        plt.sca(axs[int(i >= 3)][int(i % 3)])
        first_exp_art = Artifacts(expLocation=exp_pair[0])
        second_exp_art = Artifacts(expLocation=exp_pair[1])

        PairWiseRoi_(['ATR+', 'ATR- (Ctrl)'], [first_exp_art.getArtifact('roi'), second_exp_art.getArtifact('roi')], showShow=False, paper=paper, show_count=False, freq=120)
        #plt.title(titles[i])
        #plt.legend(fontsize='small', title_fontsize='40')

        first_mean_speed = np.mean(first_exp_art.getArtifact('speed')['speed'])
        second_mean_speed = np.mean(second_exp_art.getArtifact('speed')['speed'])

        first_mean_proj = np.mean(first_exp_art.getArtifact('proj')['proj'])
        second_mean_proj = np.mean(second_exp_art.getArtifact('proj')['proj'])

        first_max_roi = np.mean(first_exp_art.getArtifact('roi')['arrivedFrac'])
        second_max_roi = np.mean(second_exp_art.getArtifact('roi')['arrivedFrac'])


        cur_df = pd.DataFrame({'Time' : i, 'ATR+' : [first_mean_speed, first_mean_proj, first_max_roi],
                               'ATR-' : [second_mean_speed, second_mean_proj, second_max_roi],
                               'Type' : ['Speed','Projection','Roi']})

        df = pd.concat((df, cur_df), ignore_index=True)





    #fig.tight_layout()
    #fig.savefig('/home/itskov/Dropbox/dayfigs.png')

    if show:
        plt.show()

    return df


def main():
    plots_pairs = []
    plot_legends = []
    plot_titles = []

    exp1 = ('/mnt/storageNASRe/tph1/Results/15-Jan-2020/TPH_1_ATR_TRAIN_120M_D0_NO_IAA3x5.avi_18.43.27')
    exp2 = ('/mnt/storageNASRe/tph1/Results/15-Jan-2020/TPH_1_NO_ATR_TRAIN_120M_D0_NO_IAA3x5.avi_18.42.44')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')


    exp1 = ('/mnt/storageNASRe/tph1/Results/15-Jan-2020/TPH_1_ATR_TRAIN_60M_D60_NO_IAA3x5.avi_14.24.06')
    exp2 = ('/mnt/storageNASRe/tph1/Results/15-Jan-2020/TPH_1_NO_ATR_TRAIN_60M_D60_NO_IAA3x5.avi_14.23.14')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')


    exp1 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D120_NO_IAA3x5.avi_12.57.03/')
    exp2 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D120_NO_IAA3x5.avi_12.56.12/')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D180_NO_IAA3x5.avi_15.29.04/')
    exp2 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D180_NO_IAA3x5.avi_15.28.03/')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D240_NO_IAA3x5.avi_17.28.52/')
    exp2 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D240_NO_IAA3x5.avi_17.28.04/')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D300_NO_IAA3x5.avi_20.12.12/')
    exp2 = ('/mnt/storageNASRe/tph1/Results/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D300_NO_IAA3x5.avi_20.11.34/')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')


    day_summary_plots(plots_pairs, plot_titles, plot_legends, paper=True)

if __name__ == "__main__":
    main()


