from Behavior.Tools.Artifacts import Artifacts
from os import path

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


import seaborn as sns

def tracksSpeedsPerSpike(exp_files):
    sns.set()
    sns.set_context('paper')

    for i, exp_tuple in enumerate(exp_files):
        dir_name = path.dirname(exp_tuple[0])

        cur_artifacts = Artifacts(expLocation=dir_name)

        tracks_by_pulse = cur_artifacts.getArtifact('tracks_by_pulse')

        for key in tracks_by_pulse.keys():
            pulse_df = pd.DataFrame()

            current_tracks = tracks_by_pulse[key]
            spikes_speeds = np.array([t._tracksSpeeds for t in current_tracks])
            # Getting rid start/end time artifacts.
            spikes_speeds = spikes_speeds[:, 2:-2]

            df = pd.DataFrame({'Time': tuple(range(0, spikes_speeds.shape[1])) * 0.5 * spikes_speeds.shape[0],
                               'Speed': np.reshape(spikes_speeds, np.size(spikes_speeds))})


            fig = sns.lineplot(x='time',y='speed',data=df, ci=None)
            fig.set_axes_label('Speed [au]', 'Time [s]')
            pass



def tracksSpeedsForAllSpikes(exp_files):
    #sns.set()
    sns.set_context('talk')
    plt.style.use('dark_background')

    df = pd.DataFrame({'Frame': [], 'Speed': [], 'ActivationTime': []})

    for i, exp_tuple in enumerate(exp_files):
        dir_name = path.dirname(exp_tuple[0])

        cur_artifacts = Artifacts(expLocation=dir_name)

        tracks_for_all_pulses = cur_artifacts.getArtifact('tracks_for_all_spikes')


        spikes_speeds = np.array([t._tracksSpeeds for t in tracks_for_all_pulses])
        # Getting rid start/end time artifacts.
        spikes_speeds = spikes_speeds[:, 2:-2]
        norm_factor = np.reshape(np.mean(spikes_speeds[:, 0:10], axis=1), (spikes_speeds.shape[0],1))
        spikes_speeds = spikes_speeds - norm_factor

        current_df = pd.DataFrame({'Time': tuple(np.array(range(0, spikes_speeds.shape[1])) * 0.5) * spikes_speeds.shape[0],
                                   'Speed': np.reshape(spikes_speeds, np.size(spikes_speeds)),
                                   'ActivationTime': exp_tuple[1]})

        cp = sns.cubehelix_palette(8)
        ax = sns.lineplot(x='Time', y='Speed', data=current_df, ci=95, color=cp[5], linewidth=2)
        plt.gca().grid(alpha=0.2)
        ax.set(xlabel='Time [s]', ylabel='Speed [au]')
        plt.gca().grid(alpha=0.2)

        df = pd.concat((df, current_df))
        pass


    ax = sns.lineplot(x='Time', y='Speed', hue='ActivationTime', data=df, ci=None, linewidth=2, alpha=0.75)
    plt.gca().grid(alpha=0.2)
    ax.set(xlabel='Time [s]', ylabel='Speed [au]')
    plt.gca().grid(alpha=0.2)
    plt.show()
    pass






if __name__ == "__main__":
    '''exp_files = [('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_0.5S.avi_10.09.48/exp.npy', '0.5s'),
                 ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_1S.avi_10.54.39/exp.npy', '1s'),
                 ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_1.5S.avi_11.45.54/exp.npy', '1.5s'),
                 ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_2S.avi_12.39.07/exp.npy', '2s'),
                 ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_2.5S.avi_13.25.34/exp.npy', '2.5s'),
                 ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_3S.avi_14.14.43/exp.npy', '3s')]'''

    exp_files = [('/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_0.5S.avi_22.01.55/exp.npy','0.5s'),
                 #('/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_1S.avi_19.49.05/exp.npy','1s'),
                 #('/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_3S.avi_20.36.24/exp.npy','3s'),
                 ('/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_5S.avi_21.16.44/exp.npy', '5s')]

    tracksSpeedsForAllSpikes(exp_files)


