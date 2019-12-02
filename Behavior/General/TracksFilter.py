
def filterTracksForAnalyses(tracks, minSteps=0, minDistance=0):
    newTracks = [track for track in tracks if track.getMaxDistTravelled() >= minDistance and
                 track._trackCords.shape[0] >= minSteps]

    return newTracks

