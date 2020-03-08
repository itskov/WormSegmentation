from Behavior.General.TracksFilter import filterTracksForAnalyses
from os import path

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

import cv2

def main(exp_file):
    RANGE_BEFORE = 100
    RANGE_AFTER = 260
    MAX_FRAMES = 5000

    exp = np.load(exp_file)[0]

    # Filtering the tracks.
    exp._tracks = filterTracksForAnalyses(exp._tracks, minSteps=380, minDistance=200)

    # Get pulses in time
    dir_name = path.dirname(exp._videoFilename)
    pulses_filename = path.join(dir_name, 'pulses.npy')
    print(pulses_filename)
    if path.exists(pulses_filename):
        print('Found file.')
        frame_intensities = np.load(pulses_filename)
    else:
        frame_intensities = np.zeros((MAX_FRAMES,))

        for i in range(MAX_FRAMES - 1):
            _, cur_frame = exp._cap.read()
            frame_intensities[i] = np.mean(cur_frame)
            print('Went over frame: %d' % i)

        np.save(pulses_filename, frame_intensities)


    # Looking for spike points
    spike_points = np.where(np.diff(frame_intensities) > 10)[0]
    spike_points = spike_points[1:]

    spike_ranges = []
    for spike_point in spike_points:
        spike_ranges.append(list(range(spike_point - RANGE_BEFORE, spike_point + RANGE_AFTER)))


    tracks_for_spikes = []
    df = pd.DataFrame({'frame' : [], 'speed' : []})
    # Going over the relevant tacks.
    for track in exp._tracks:
        current_df = pd.DataFrame({'frame' : track._trackFrames, 'speed' : track._tracksSpeeds})
        df = pd.concat((df, current_df))
        for i, spike_range in enumerate(spike_ranges):
            #DEBUG
            #if i != 3:
            #    continue
            #DEBUG
            if spike_range[0] in track._trackFrames and spike_range[-1] in track._trackFrames:
                new_track = track.trimTrack(spike_range[-1], spike_range[0])
                tracks_for_spikes.append(new_track)


    spikes_speeds = np.array([t._tracksSpeeds for t in tracks_for_spikes])
    spikes_reversals = np.array([t._tracksReversals for t in tracks_for_spikes])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.mean(spikes_speeds, axis=0))
    ax2 = ax.twinx()
    ax2.plot(np.sum(spikes_reversals, axis=0) / np.prod(spikes_reversals.shape))

    plt.figure()
    sns.lineplot(x='frame', y='speed', data=df, ci=None)

    #plt.plot(frame_intensities)
    plt.show()


    pass

if __name__ == "__main__":
    exp_file = '/home/itskov/Temp/behav/01-Mar-2020/TPH_1_ATR_ONLINE[NO_IAA].avi_10.38.56/exp.npy'
    main(exp_file)
