import numpy as np


def subset_track(t, positions):
    t._trackCords = t._trackCords[positions, :]
    t._trackFrames = t._trackFrames[positions]
    t._tracksSteps = t._tracksSteps[positions, :]
    t._tracksReversals = t._tracksReversals[positions]
    return t


def seiveTracks(exp):
    scale = exp._scale

    SMALL_PROP = 0.2
    BIG_PROP = 1 - SMALL_PROP
    FRAMES = 4500

    small_radius = SMALL_PROP * 2
    big_radius = BIG_PROP * 2

    # Getting the tracks.
    tracks = exp._tracks
    good_tracks = np.array([])

    for i, t in enumerate(tracks):
        # Filter in time.
        interesting_poses = t._trackFrames < FRAMES

        t = subset_track(t, interesting_poses)

        # Check if there are any coordinates left.
        if t._trackCords.shape[0] == 0:
            continue

        # Sieving by position
        distances = np.linalg.norm(t._trackCords - exp._regionsOfInterest['endReg']['pos'], axis=1)
        t = subset_track(t, np.logical_not(distances < small_radius))
        t = subset_track(t, np.logical_not(distances > big_radius))

        if t._trackCords.shape[0] == 0:
            continue





if __name__ == "__main__":
    exp = np.load('/home/itskov/Temp/behav/12-Dec-2019/STAV_CHOICETEST.avi_13.51.29/exp.npy')[0]
    seiveTracks(exp)