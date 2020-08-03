import matplotlib.pyplot as plt
import pandas as pd

from Behavior.Visualizers.PairwiseAnalyses import PairWiseRoi_
from Behavior.Tools.Artifacts import Artifacts

import numpy as np
import seaborn as sns

def day_summary_plots(exp_pairs, titles, legends, paper=False, show=True, output_file=None):
    number_of_exps = len(exp_pairs)

    fig, axs = plt.subplots(number_of_exps, 3, figsize=(10, 14))
    plt.subplots_adjust(wspace=0.36, left=0.09, bottom=0.11, right=0.98, top=0.96, hspace=0.36)

    if not paper:
        plt.style.use("dark_background")
        sns.set_context("talk")
    else:
        sns.set_context("paper")


    axs = np.atleast_2d(axs)

    df = pd.DataFrame({'Value' : [], 'Type' : [], 'Cond' :[],'Exp' : []})

    for i, exp_pair in enumerate(exp_pairs):

        first_exp_art = Artifacts(expLocation=exp_pair[0])
        second_exp_art = Artifacts(expLocation=exp_pair[1])

        plt.sca(axs[int(i)][0])
        PairWiseRoi_(['ATR+', 'ATR- (Ctrl)'], [first_exp_art.getArtifact('roi'), second_exp_art.getArtifact('roi')], showShow=False, paper=paper, show_count=False, freq=120)
        df = df.append({'Value': np.max(first_exp_art.getArtifact('roi')['arrivedFrac']),
                        'Type': 'Max Roi',
                         'Cond': 'ATR+',
                         'Exp': i}, ignore_index=True)

        df = df.append({'Value': np.max(second_exp_art.getArtifact('roi')['arrivedFrac']),
                        'Type': 'Max Roi',
                         'Cond': 'ATR-',
                         'Exp': i}, ignore_index=True)


        plt.sca(axs[int(i)][1])
        first_speed = (first_exp_art.getArtifact('speed')['speed'])
        second_speed = (second_exp_art.getArtifact('speed')['speed'])
        sns.kdeplot(first_speed, shade=True, label='ATR+')
        ax = sns.kdeplot(second_speed, shade=True, label='ATR- (Control)')
        plt.gca().grid(alpha=0.2)
        ax.set(xlabel="Speed [au / sec]", ylabel="Density")
        plt.locator_params(nbins=5)
        df_cur = pd.DataFrame({'Value': np.mean(first_speed),
                               'Type': 'Speed [au]',
                               'Cond': 'ATR+',
                               'Exp': i}, index=[0])

        df = pd.concat((df, df_cur), ignore_index=True)

        df_cur = pd.DataFrame({'Value': np.mean(second_speed),
                               'Type': 'Speed [au]',
                               'Cond': 'ATR-',
                               'Exp': i}, index=[0])

        df = pd.concat((df, df_cur), ignore_index=True)




        plt.sca(axs[int(i)][2])
        first_proj = (first_exp_art.getArtifact('proj')['proj'])
        second_proj = (second_exp_art.getArtifact('proj')['proj'])
        sns.kdeplot(first_proj, shade=True, label='ATR+')
        ax = sns.kdeplot(second_proj, shade=True, label='ATR- (Control)')
        plt.gca().grid(alpha=0.2)
        ax.set(xlabel="Projection", ylabel="Density")
        plt.locator_params(nbins=5)


        df_cur = pd.DataFrame({'Value': np.mean(first_proj),
                               'Type': 'Projection',
                               'Cond': 'ATR+',
                               'Exp': i}, index=[0])

        df = pd.concat((df, df_cur), ignore_index=True)

        df_cur = pd.DataFrame({'Value': np.mean(second_proj),
                               'Type': 'Projection',
                               'Cond': 'ATR-',
                               'Exp': i}, index=[0])

        df = pd.concat((df, df_cur), ignore_index=True)





    if show:
        plt.show()

    return df


def main():
    plots_pairs = []
    plot_legends = []
    plot_titles = []

    #exp1 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_ATR_75M.avi_11.28.43')
    #exp2 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_NO_ATR_75M.avi_11.27.26')
    #plots_pairs.append((exp1, exp2))
    #plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    #plot_titles.append('0m')


    exp1 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_ATR_75M.avi_12.49.27')
    exp2 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_NO_ATR_75M.avi_12.48.42')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')


    exp1 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_ATR_75M.avi_14.01.15')
    exp2 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_NO_ATR_75M.avi_14.00.19')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_ATR_75M.avi_15.20.05')
    exp2 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_NO_ATR_75M.avi_15.19.25')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')

    exp1 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_ATR_75M.avi_16.48.02')
    exp2 = ('/mnt/storageNASRe/tph1/Results/05-Jul-2020/TM1_NO_ATR_75M.avi_16.46.59')
    plots_pairs.append((exp1, exp2))
    plot_legends.append(('ATR+ (Experiment)', 'ATR- (Control)'))
    plot_titles.append('0m')


    df = day_summary_plots(plots_pairs, plot_titles, plot_legends, paper=True, show=True)
    print(df)

if __name__ == "__main__":
    main()


