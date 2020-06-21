from Behavior.General.TracksFilter import seiveTracks
from Behavior.Tools.Artifacts import Artifacts
from pathlib import Path
from os import path

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    folder = '/home/itskov/Temp/Chris'

    #folder = '/mnt/storageNASRe/ChristianData/ChrisNewTracks'

    conds = ['LTAP.','MOCK_LTAP', 'LTAV.','MOCK_LTAV', 'STAP.','MOCK_STAP', 'STAV.','MOCK_STAV', 'NAIVE']


    df = None
    for cond in conds:
        print('Cond: ' + cond)
        for i, filename in enumerate(Path(folder).rglob(cond + "*")):
            print('Working on: %s' % (filename,))

            currentArt = Artifacts(expLocation=filename)
            current_roi = currentArt.getArtifact('roi')

            if current_roi is None:
                continue

            if current_roi['wormCount'] < 20:
                print('Not enough worms: %d. Continuing.' % (current_roi['wormCount'],))
                continue


            TIME = 2160
            frames = range(1, len(current_roi['arrivedFrac']) + 1)
            arrived_frac = np.maximum(0, current_roi['arrivedFrac'][0:TIME])
            current_df = pd.DataFrame({'frame': range(1, TIME + 1), 'arrived_frac': arrived_frac, 'cond' : cond})

            df = current_df if df is None else pd.concat((df, current_df), ignore_index=True)


    sns.set()
    #sns.lineplot(x='frame',y='arrived_frac', hue='cond', data=df, estimator=np.median, ci=68, n_boot=1000)
    df.to_csv('/home/itskov/workspace/fraction_in_time.csv')


    #plt.ylim([-0.001, 0.15])
    #plt.xlabel('Frame [2hz]')
    #plt.ylabel('Arrived Fraction')
    #plt.show()

    pass

if __name__ == "__main__":
    main()
