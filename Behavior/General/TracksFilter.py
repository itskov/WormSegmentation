
def filterTracksForAnalyses(tracks):
    MIN_DISTANCE = 150
    MIN_STEPS = 100

    newTracks = [track for track in tracks if track.getMaxDistTravelled() >= MIN_DISTANCE and
                 track._trackCords.shape[0] >= MIN_STEPS]

    return newTracks

