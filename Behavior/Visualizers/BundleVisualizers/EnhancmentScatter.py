from pathlib import Path
from Behavior.Tools.Artifacts import Artifacts

import pandas as pd
import seaborn as sns
import sys
import json

import numpy as np

import matplotlib.pyplot as plt


def gatherBundles(dirname):
    bundlesDf = pd.DataFrame({'Strain': [], 'ExpType': [], 'MiniProj': [], 'files': []})


    for path in Path(dirname).rglob('*.json'):
        print(path)

        with open(path) as f:
            currentJson = json.load(f)

        bundlesDf = bundlesDf.append({'Strain': currentJson['Strain'],
                                      'ExpType': currentJson['ExpType'],
                                      'MiniProj': currentJson['MiniProj'],
                                      'files': currentJson['files']},
                                     ignore_index=True)


    return bundlesDf

def scatterEnhacnment(pairsDf, paper=False, figNames=None):
    MAX_TIME = 3750

    plotDfArrivedFrac = pd.DataFrame({'Strain': [], 'ATR+': [], 'ATR-': []})
    plotDfProjection = pd.DataFrame({'Strain': [], 'ATR+': [], 'ATR-': []})
    plotDfSpeed = pd.DataFrame({'Strain': [], 'ATR+': [], 'ATR-': []})

    linePlotDf = pd.DataFrame({'Strain': [], 'time': [], 'FrationArrived': [],'Exp': []})

    print(pairsDf.shape)
    for i in range(pairsDf.shape[0]):
        try:
            current_row = pairsDf.iloc[i]
            files = np.array(current_row['files'])

            noAtrInd = np.array([file.find('NO_ATR') >= 0 for file in files])

            if np.sum(noAtrInd) != 1:
                print('*** Error: Invalid pair ***')
            else:
                atrFile = files[np.logical_not(noAtrInd)][0]
                noAtrFile = files[noAtrInd][0]

                print(atrFile)
                print(noAtrFile)

                atrRoi = Artifacts(expLocation=atrFile).getArtifact('roi')
                noAtrRoi = Artifacts(expLocation=noAtrFile).getArtifact('roi')

                if atrRoi['wormCount'] < 30 or noAtrRoi['wormCount'] < 30:
                    print('Not enough worms. Continuing.')
                    continue

                plotDfArrivedFrac = plotDfArrivedFrac.append({'Strain': current_row['Strain'],
                                                              'ATR+': np.minimum(atrRoi['arrivedFrac'][MAX_TIME], 1),
                                                              'ATR-': np.minimum(noAtrRoi['arrivedFrac'][MAX_TIME], 1),
                                                              'ExpId' : i},
                                                              ignore_index=True)

                atrProj = Artifacts(expLocation=atrFile).getArtifact('proj')['proj']
                noAtrProj = Artifacts(expLocation=noAtrFile).getArtifact('proj')['proj']

                plotDfProjection = plotDfProjection.append({'Strain': current_row['Strain'],
                                                            'ATR+': np.mean(atrProj),
                                                            'ATR-': np.mean(noAtrProj),
                                                            'ExpId' : i}, ignore_index=True)

                atrSpeed = Artifacts(expLocation=atrFile).getArtifact('speed')['speed']
                noAtrSpeed = Artifacts(expLocation=noAtrFile).getArtifact('speed')['speed']

                plotDfSpeed = plotDfSpeed.append({'Strain': current_row['Strain'],
                                                  'ATR+': np.mean(atrSpeed),
                                                  'ATR-': np.mean(noAtrSpeed),
                                                  'ExpId' : i}, ignore_index=True)

                print("Arrival vals: ATR+:%f, ATR-:%f, Projection mean: ATR+:%f, ATR-:%f, Speed mean ATR+:%f, ATR-:%f" %
                      (atrRoi['arrivedFrac'][MAX_TIME],
                       noAtrRoi['arrivedFrac'][MAX_TIME],
                       np.mean(atrProj),
                       np.mean(noAtrProj),
                       np.mean(atrSpeed),
                       np.mean(noAtrSpeed)))

                print(atrRoi['arrivedFrac'][0:MAX_TIME].shape)
                print(np.array(list(range(MAX_TIME))).shape)

                currentLinePlot = pd.DataFrame({'Strain': current_row['Strain'],
                                                'time': np.array(list(range(MAX_TIME))) * 0.5,
                                                'FractionArrived': atrRoi['arrivedFrac'][0:MAX_TIME],
                                                'Exp': 'ATR'})

                linePlotDf = pd.concat((linePlotDf, currentLinePlot))

                currentLinePlot = pd.DataFrame({'Strain': current_row['Strain'],
                                                'time': np.array(list(range(MAX_TIME))) * 0.5,
                                                'FractionArrived': noAtrRoi['arrivedFrac'][0:MAX_TIME],
                                                'Exp': 'NO ATR'})

                linePlotDf = pd.concat((linePlotDf, currentLinePlot))
        except Exception as e:
            raise e


    #print(plotDf)
    #DEBUG
    #plotDfArrivedFrac.to_pickle('/home/itskov/Dropbox/tempBundle_arrival.pkl')
    #plotDfProjection.to_pickle('/home/itskov/Dropbox/tempBundle_proj.pkl')
    #plotDfSpeed.to_pickle('/home/itskov/Dropbox/tempBundle_speed.pkl')
    #linePlotDf.to_pickle('/home/itskov/Dropbox/tempBundle2.pkl')
    #DEBUG

    if paper == False:
        plt.style.use("dark_background")
        sns.set_context('talk')
        cp = sns.dark_palette("purple", 7)
    else:
        cp = sns.cubehelix_palette(8, start=.5, rot=-.75)

    if paper == False:
        ax = sns.scatterplot(x='ATR-', y='ATR+',  data=plotDfArrivedFrac, linewidth=0,  alpha=0.85, color=cp[6])
        ax.plot([0, 1.1], [0, 1.1], ":")
    else:
        ax = sns.scatterplot(x='ATR-', y='ATR+',  data=plotDfArrivedFrac, linewidth=0,  alpha=0.85, color=cp[6])
        ax.plot([0, 1.1], [0, 1.1], ":", color='k')


    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.gca().grid(alpha=0.2)
    plt.title('Arrival Fracion, n = %d' % (plotDfArrivedFrac.shape[0],), loc='left')
    if figNames is None:
        plt.show()
    else:
        plt.gcf().savefig(figNames[0], format='svg')


    # Plotting the projection plot.
    #plt.style.use("dark_background")
    #sns.set_context('talk')
    #cp = sns.dark_palette("purple", 7)


    ax = sns.scatterplot(x='ATR-', y='ATR+',  data=plotDfProjection, linewidth=0,  alpha=0.85, color=cp[6])

    xmin = np.min(plotDfProjection['ATR-'])
    xmax = np.max(plotDfProjection['ATR-'])
    ymin = np.min(plotDfProjection['ATR-'])
    ymax = np.max(plotDfProjection['ATR+'])

    if paper == False:
        ax.plot([0, 1], [0, 1], ":")
    else:
        ax.plot([0, 1], [0, 1], ":", color='k')

    plt.xlim(0, 0.35)
    plt.ylim(0, 0.35)
    plt.gca().grid(alpha=0.2)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=6)
    plt.title('Mean Projection, n = %d' % (plotDfArrivedFrac.shape[0],), loc='left')
    if figNames is None:
        plt.show()
    else:
        plt.gcf().savefig(figNames[1], format='svg')

    # Plotting the speed plot.
    #plt.style.use("dark_background")
    #sns.set_context('talk')
    #cp = sns.dark_palette("purple", 7)
    ax = sns.scatterplot(x='ATR-', y='ATR+',  data=plotDfSpeed, linewidth=0,  alpha=0.85, color=cp[6])

    xmin = np.min(plotDfSpeed['ATR-'])
    xmax = np.max(plotDfSpeed['ATR-'])
    ymin = np.min(plotDfSpeed['ATR+'])
    ymax = np.max(plotDfSpeed['ATR+'])

    if paper == False:
        ax.plot([0, 0.0018], [0, 0.0018], ":")
    else:
        ax.plot([0, 0.0018], [0, 0.0018], ":", color='k')

    plt.xlim(0.0008, 0.0018)
    plt.ylim(0.0008, 0.0018)
    plt.gca().grid(alpha=0.2)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=6)
    plt.title('Mean Speed, n = %d' % (plotDfSpeed.shape[0],), loc='left')
    if figNames is None:
        plt.show()
    else:
        plt.gcf().savefig(figNames[2], format='svg')




    #ax = sns.lineplot(x='time', y='FractionArrived', hue='Exp', data=linePlotDf, ci=68, estimator=np.median)
    #plt.gca().grid(alpha=0.2)
    #ax.set(xlabel='Time [s]')

    #plt.show()
    return (plotDfArrivedFrac, plotDfProjection, plotDfSpeed)




if __name__ == "__main__":
    from os import path
    from Behavior.Visualizers.ProjectionAnalyses import ProjectionAnalyses

    #bundleDfs = gatherBundles(sys.argv[1])
    bundleDfs = gatherBundles('/mnt/storageNASRe/tph1/Results/')

    # Go over the right bundles

    pairsDf = bundleDfs[bundleDfs['ExpType'] == 'Pair Comparison']
    scatterEnhacnment(pairsDf)


    '''for i in range(pairsDf.shape[0]):
        dirnames = pairsDf.iloc[i]['files']

        for dirname in dirnames:
            exp = np.load(path.join(str(dirname), 'exp.npy'))[0]
            art = Artifacts(expLocation=dirname)
            projectionAnalyses = ProjectionAnalyses(exp)
            projectionAnalyses.execute()
            art.addArtifact('proj', projectionAnalyses._results)
            print('%d. Added artifact' % (i,))'''








