import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


from Behavior.Tools.Artifacts import Artifacts


def func(t, m, a):
    return m*(1 - np.exp(-(1/a) * t))

def analyzeSpeedUps(exps, paper=False, fitLinear=True, show=True):

    delays = []
    speed_ups = []

    for expDir, delay in exps:
        artifacts = Artifacts(expLocation=expDir)
        frame_intensities = artifacts.getArtifact('frame_intensities')
        tracks_for_all_spikes = artifacts.getArtifact('tracks_for_all_spikes')

        speeds = [t._tracksSpeeds for t in tracks_for_all_spikes]

        signal = np.mean(speeds, axis=0)
        signal = signal[2:-2]
        min_decay_ind = np.argmin(signal)
        signal = signal[min_decay_ind:]
        signal = signal / signal[-1]

        xdata = np.array(range(signal.shape[0])) * 0.5

        popt, pcov = curve_fit(func, xdata, signal)

        plt.plot(xdata, signal)
        plt.plot(xdata, func(xdata, *popt), 'r-')
        print(popt)

        delays.append(delay)
        speed_ups.append(popt[1])

        pass




    regressor  = LinearRegression()
    delays = np.array(delays).reshape(-1, 1)
    speed_ups = np.array(speed_ups).reshape(-1, 1)
    regressor.fit(delays, speed_ups)
    predicted = regressor.predict(delays)

    linearDf = pd.DataFrame({'Activation Duration [s]': np.ravel(delays),
                             'Time Coefficient [s]': np.ravel(speed_ups),
                             'Predicted': np.ravel(predicted)})

    if not fitLinear:
        print(np.ravel(delays))
        print(np.ravel(speed_ups))
        popt, pcov = curve_fit(func, np.ravel(delays), np.ravel(speed_ups))


    plt.close()
    if not paper:
        sns.set_context('talk')
        plt.style.use("dark_background")
        cp = sns.dark_palette("purple", 7)

        if fitLinear:
            sns.lineplot(data=linearDf, x='Activation Duration [s]', y='Predicted', ci=None, color=cp[2])
        else:
            xvals = np.linspace(np.min(delays), np.max(delays), 1000)
            plt.plot(xvals, func(xvals, *popt), color=cp[2])

        sns.scatterplot(data=linearDf, x='Activation Duration [s]', y='Time Coefficient [s]', linewidth=0, alpha=0.98,
                        color=cp[6])

    else:
        sns.set_context('paper')

        if fitLinear:
            sns.lineplot(data=linearDf, x='Activation Duration [s]', y='Predicted', ci=None)
        else:
            xvals = np.linspace(np.min(delays), np.max(delays), 1000)
            plt.plot(xvals, func(xvals, *popt))

        sns.scatterplot(data=linearDf, x='Activation Duration [s]', y='Time Coefficient [s]', linewidth=0, alpha=0.98)

    plt.title('$R^2 = %.2f$' % (r2_score(speed_ups, predicted),), loc='left')
    plt.gca().grid(alpha=0.2)

    if show:
        plt.show()
    pass






if __name__ == "__main__":
    exps = [('/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_0.5S.avi_12.22.02', 0.5),
            ('/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_1S.avi_11.22.03', 1),
            ('/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_1.5S.avi_13.15.35', 1.5),
            ('/mnt/storageNASRe/tph1/Results/21-Jun-2020/TM5_ATR_ONLINE_2S.avi_13.49.57', 2)]


    analyzeSpeedUps(exps)
    pass
