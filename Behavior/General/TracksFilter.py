import pandas as pd


def filterTracksForAnalyses(tracks, minSteps=0, minDistance=0):
    newTracks = [track for track in tracks if track.getMaxDistTravelled() >= minDistance and
                 track._trackCords.shape[0] >= minSteps]

    return newTracks



import numpy as np
#DEBUG
from Behavior.Visualizers.OccupVisualizer import OccupVisualizer


def subset_track(t, positions):
    t._trackCords = t._trackCords[positions, :]
    t._trackFrames = t._trackFrames[positions]
    t._tracksSteps = t._tracksSteps[positions, :]
    t._tracksReversals = t._tracksReversals[positions]
    t._tracksSpeeds = t._tracksSpeeds[positions]

    return t

def tracks_to_df(exp):

    exp._tracks = filterTracksForAnalyses(exp._tracks, 50, 50)

    tracks = exp._tracks
    df = None

    for i, t in enumerate(tracks):
        current_track_df = pd.DataFrame(columns=["cordx", "cordy", "speed", "angle", "distance", "frame"])

        angles = t.getAngles(exp._regionsOfInterest['endReg']['pos'])
        distances = t.getDistances(exp._regionsOfInterest['endReg']['pos'])

        distances /= exp._scale

        current_track_df = pd.DataFrame({'frame': t._trackFrames,
                                         'cordx': t._trackCords[:, 1],
                                         'cordy': t._trackCords[:,0],
                                         'speed': t._tracksSpeeds,
                                         'angle': angles,
                                         'distance': distances})

        df = current_track_df if df is None else pd.concat([df, current_track_df])


    df = df.dropna()
    df.angle = df.angle.astype(np.float64)
    df['distance_label'] = pd.cut(df.distance, bins=30, labels=False)
    df['distance_label'] = 30 - df['distance_label']

    return df



def seiveTracks(exp):
    scale = exp._scale

    SMALL_PROP = 0.3
    BIG_PROP = 1 - SMALL_PROP
    FRAMES = 4500
    ANGLE = np.pi / 3

    small_radius = (SMALL_PROP) * exp._scale
    big_radius = (BIG_PROP) * exp._scale

    # Getting the tracks.
    tracks = exp._tracks
    good_tracks = np.array([])

    # horizontal line
    horiz_line = np.array(exp._regionsOfInterest['startReg']['pos']) - np.array(exp._regionsOfInterest['endReg']['pos'])
    horiz_line /= np.linalg.norm(horiz_line)

    # Perp line
    if horiz_line[0] != 0:
        perp_line = -np.ones((2,))
        perp_line[1] = horiz_line[0] / horiz_line[1]
    else:
        perp_line = [1, 0]

    perp_line /= -np.linalg.norm(perp_line)



    # New basis
    new_basis = np.array([perp_line, horiz_line]).T
    trans = np.linalg.inv(new_basis)
    new_end_point = np.matmul(trans, exp._regionsOfInterest['endReg']['pos'])

    newTracks = []

    for i, t in enumerate(tracks):
        # Filter in time.
        if t._trackCords.shape[0] < 50:
            continue

        if t.getMaxDistTravelled() < 75:
            continue

        # First, Trimming the track.
        t = t.trimTrack(FRAMES)

        # Check if there are any coordinates left.
        if t == None or t._trackCords.shape[0] == 0:
            continue

        # Sieving by position
        distances = np.linalg.norm(t._trackCords - exp._regionsOfInterest['endReg']['pos'], axis=1)
        t = subset_track(t, (distances > small_radius))

        if t._trackCords.shape[0] == 0:
            continue


        distances = np.linalg.norm(t._trackCords - exp._regionsOfInterest['endReg']['pos'], axis=1)
        t = subset_track(t, (distances < big_radius))

        if t._trackCords.shape[0] == 0:
            continue

        newCords = np.matmul(trans, t._trackCords.T)
        newCords = newCords.T


        y_boundaries = (newCords[:, 1] - np.array(new_end_point[1])) * np.tan(ANGLE)
        t = subset_track(t, np.abs(newCords[:, 0] - np.array(new_end_point[0])) < y_boundaries)
        if t._trackCords.shape[0] == 0:
            continue

        # Trim it again to create the metrics again.
        t = t.trimTrack(FRAMES)

        if t is not None:
            newTracks.append(t)


        print('Gone over track:%d' % (i,))


    from copy import deepcopy
    from Behavior.General.ExpDir import ExpDir
    from os import path
    import matplotlib.pyplot as plt
    import seaborn as sns

    exp._regionsOfInterest['endReg']['pos'] = new_end_point
    exp._tracks = None
    exp._cap = None
    new_exp = deepcopy(exp)
    new_exp._tracks = np.array(newTracks)
    new_exp.initialize(ExpDir(path.dirname(new_exp._videoFilename)))

    #df = tracks_to_df(new_exp)

    return new_exp


def get_dataframe_per_cond(folder, cond):
    from pathlib import Path
    from os import path

    df = None

    for i, filename in enumerate(Path(folder).rglob(cond + "*")):
        exp_filename = path.join(filename, 'exp.npy')
        print('%d. Going over: %s' % (i, exp_filename))
        if path.exists(exp_filename):
            print('Found experiment.')
            exp = np.load(exp_filename)[0]

            if np.any(np.isnan(np.array(exp._regionsOfInterest['endReg']['pos']))):
                print('Bad regions.')
                continue

            exp = seiveTracks(exp)
            current_df = tracks_to_df(exp)
            df = current_df if df is None else pd.concat([df, current_df])


    # DEBUG
    #import seaborn as sns
    #sns.lineplot(data=df, x='distance_label', y='angle')
    # DEBUG

    return df

def main():
    import seaborn as sns
    # exp = np.load('/mnt/storageNASRe/ChristianData/ChrisNewTracks/05-Dec-2018_Chris/STAP.avi_11.51.41/exp.npy')[0]
    dfMockStap = get_dataframe_per_cond('/mnt/storageNASRe/ChristianData/ChrisNewTracks', 'MOCK_LTAP')
    dfStap = get_dataframe_per_cond('/mnt/storageNASRe/ChristianData/ChrisNewTracks', 'LTAP')

    dfMockStap['cond'] = 'MOCK_LTAP'
    dfStap['cond'] = 'LTAP'
    df = pd.concat((dfMockStap, dfStap), ignore_index=True)
    sns.lineplot(data=df, x='distance_label', y='angle', hue='cond')

    pass


if __name__ == "__main__":
    main()

