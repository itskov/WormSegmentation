from Behavior.General.TracksFilter import filterTracksForAnalyses
from Behavior.General.ExpDir import ExpDir
from Behavior.Tools.Artifacts import Artifacts
from scipy.signal import savgol_filter

from os import path

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

import cv2


def gatherTracksForSpikes(exp_file, show_plot=False):
    RANGE_BEFORE = 100
    RANGE_AFTER = 260
    MAX_FRAMES = 3500

    speed_up_times = []

    exp = np.load(exp_file)[0]

    # Filtering the tracks.
    exp._tracks = filterTracksForAnalyses(exp._tracks, minSteps=370, minDistance=150)

    # Get pulses in time
    dir_name = path.dirname(exp._videoFilename)
    #pulses_filename = path.join(dir_name, 'pulses.npy')
    #print(pulses_filename)
    artifacts = Artifacts(expLocation=dir_name)
    artifacts.checkForArtifactsDir()
    frame_intensities = artifacts.getArtifact('frame_intensities')
    if frame_intensities is not None:
        print('Found file.')
    else:
        frame_intensities = np.zeros((MAX_FRAMES,))

        for i in range(MAX_FRAMES - 1):
            _, cur_frame = exp._cap.read()
            frame_intensities[i] = np.mean(cur_frame)
            print('Went over frame: %d' % i)

        artifacts.addArtifact('frame_intensities', frame_intensities)

    # Looking for spike points
    spike_points = np.where(np.diff(frame_intensities) > 10)[0]
    spike_points = spike_points[1:]

    spike_ranges = []
    for spike_point in spike_points:
        spike_ranges.append(list(range(spike_point - RANGE_BEFORE, spike_point + RANGE_AFTER)))

    tracks_for_spikes = []
    tracks_for_single_spikes = {}

    df = pd.DataFrame({'frame': [], 'speed': []})
    # Going over the relevant tacks.
    for track in exp._tracks:
        current_df = pd.DataFrame({'frame': track._trackFrames, 'speed': track._tracksSpeeds})
        df = pd.concat((df, current_df))
        for i, spike_range in enumerate(spike_ranges):
            # DEBUG
            # if i != 3:
            #    continue
            # DEBUG
            if spike_range[0] in track._trackFrames and spike_range[-1] in track._trackFrames:
                new_track = track.trimTrack(spike_range[-1], spike_range[0])
                tracks_for_spikes.append(new_track)
                tracks_for_single_spikes[i] = tracks_for_single_spikes[i] + [new_track]\
                    if i in tracks_for_single_spikes.keys() else [new_track]

        pass


    # Pick spike light represenetitive
    spike_rep = frame_intensities[spike_ranges[1]]

    spikes_speeds = np.array([t._tracksSpeeds for t in tracks_for_spikes])
    spikes_angles = np.array([t.getAngles(exp._regionsOfInterest['endReg']['pos']) for t in tracks_for_spikes])
    spikes_reversals = np.array([t._tracksReversals for t in tracks_for_spikes])

    spikes_angles[:, 0] = spikes_angles[:, 1]
    spikes_angles[:, -2] = spikes_angles[:, -3]
    spikes_angles[:, -1] = spikes_angles[:, -3]

    mean_spikes_speeds = np.median(spikes_speeds, axis=0)
    mean_spikes_angles = np.median(spikes_angles, axis=0)

    mean_spikes_speeds = mean_spikes_speeds[2:-2]
    spike_rep = spike_rep[2:-2]

    # Gathering the speeding up time.
    '''smoothed_speeds = savgol_filter(mean_spikes_speeds, 11, 3)
    spike_decay_point = np.max(np.where(np.diff(spike_rep) < -5)) + 3
    speed_up_time = \
        np.min(np.where(smoothed_speeds[spike_decay_point:] >= np.mean(smoothed_speeds[0:RANGE_BEFORE])))

    speed_up_times.append(speed_up_time)'''


    artifacts.addArtifact('tracks_by_pulse', tracks_for_single_spikes)
    artifacts.addArtifact('tracks_for_all_spikes', tracks_for_spikes)


if __name__ == "__main__":
    exp_files = ['/mnt/storageNASRe/tph1/Results/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_0.5S.avi_10.09.48/exp.npy',
                '/mnt/storageNASRe/tph1/Results/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_1S.avi_10.54.39/exp.npy',
                '/mnt/storageNASRe/tph1/Results/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_1.5S.avi_11.45.54/exp.npy',
                '/mnt/storageNASRe/tph1/Results/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_2S.avi_12.39.07/exp.npy',
                '/mnt/storageNASRe/tph1/Results/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_2.5S.avi_13.25.34/exp.npy',
                '/mnt/storageNASRe/tph1/Results/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_3S.avi_14.14.43/exp.npy']

    '''exp_files = ['/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_0.5S.avi_22.01.55/exp.npy',
                 '/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_1S.avi_19.49.05/exp.npy',
                 '/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_3S.avi_20.36.24/exp.npy',
                 '/home/itskov/Temp/behav/12-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA]_5S.avi_21.16.44/exp.npy']'''

    '''exp_files = ['/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_0.5S.avi_12.22.02/exp.npy',
                 '/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_1S.avi_11.22.03/exp.npy',
                 '/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_1.5S.avi_13.15.35/exp.npy',
                 '/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_2S.avi_13.49.57/exp.npy']'''

    exp_files = ['/mnt/storageNASRe/tph1/Results/01-Mar-2020/TPH_1_ATR_ONLINE[IAA].avi_10.38.56/exp.npy',
                '/mnt/storageNASRe/tph1/Results/01-Mar-2020/TPH_1_ATR_ONLINE[IAA].avi_13.16.09/exp.npy',
                '/mnt/storageNASRe/tph1/Results/01-Mar-2020/TPH_1_NO_ATR_ONLINE[IAA].avi_11.41.30/exp.npy',
                '/mnt/storageNASRe/tph1/Results/01-Mar-2020/TPH_1_NO_ATR_ONLINE[IAA].avi_14.17.21/exp.npy']






    [gatherTracksForSpikes(exp_file, show_plot=False) for exp_file in exp_files]
    #pulse_statistics(exp_files[1], show_plot=False)
    pass
