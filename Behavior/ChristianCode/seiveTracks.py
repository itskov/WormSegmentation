import numpy as np
#DEBUG
from Behavior.Visualizers.OccupVisualizer import OccupVisualizer


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
    ANGLE = 120 * (np.pi / 180)

    small_radius = (SMALL_PROP * 2) * exp._scale
    big_radius = (BIG_PROP * 2) * exp._scale

    # Getting the tracks.
    tracks = exp._tracks
    good_tracks = np.array([])

    # horizontal line
    horiz_line = np.array(exp._regionsOfInterest['endReg']['pos']) - np.array(exp._regionsOfInterest['startReg']['pos'])
    horiz_line /= np.linalg.norm(horiz_line)

    # Perp line
    perp_line = np.ones((2,))
    perp_line[0] = -horiz_line[1] / horiz_line[0]
    perp_line /= np.linalg.norm(perp_line)

    # New basis
    new_basis = np.array([perp_line, horiz_line]).T

    newTracks = []

    for i, t in enumerate(tracks):
        # Filter in time.

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

        newCords = np.matmul(np.linalg.inv(new_basis), t._trackCords.T)

        t = subset_track(t, np.abs(newCords[0, :]) < 200)
        if t._trackCords.shape[0] == 0:
            continue

        newTracks.append(t)

        print('Gone over track:%d' % (i,))


    exp._tracks = newTracks
    oc = OccupVisualizer(exp)
    oc.execute()




if __name__ == "__main__":
    exp = np.load('/home/itskov/Temp/behav/08-Jan-2020/TPH_1_ATR_TRAIN_70M_NO_IAA3x5.avi_13.05.52/exp.npy')[0]
    seiveTracks(exp)