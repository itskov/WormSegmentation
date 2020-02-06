from Behavior.General.TracksFilter import seiveTracks
from Behavior.Tools.Artifacts import Artifacts
from pathlib import Path
from os import path

import seaborn as sns
import pandas as pd
import numpy as np

def main():
    folder = '/home/itskov/Temp/Chris'

    conds = ['LTAV.','MOCK_LTAV.','LTAP.','MOCK_STAP.', 'NAIVE']


    df = None
    for cond in conds:
        print('Cond: ' + cond)
        for i, filename in enumerate(Path(folder).rglob(cond + "*")):
            print('Working on: %s' % (filename,))

            currentArt = Artifacts(expLocation=filename)
            current_roi = currentArt.getArtifact('roi')

            if current_roi is None:
                continue


            if current_roi['wormCount'] < 25:
                print('Not enough worms: %d. Continuing.' % (current_roi['wormCount'],))
                continue


            frames = range(1, len(current_roi['arrivedFrac']) + 1)
            current_df = pd.DataFrame({'frame': range(1, 1501), 'arrived_frac': current_roi['arrivedFrac'][0:1500], 'cond' : cond})

            #DEBUG
            #sns.lineplot(x='frame', y='arrived_frac', hue='cond', data=current_df, estimator=np.median)
            #DEBUG

            df = current_df if df is None else pd.concat((df,current_df), ignore_index=True)


    sns.lineplot(x='frame',y='arrived_frac', hue='cond', data=df, estimator=np.median, ci=None)
    #sns.plt.show()

    pass

if __name__ == "__main__":
    main()