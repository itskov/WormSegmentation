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

def analyzeSpeedUps(exps):

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


    sns.set_context('talk')
    plt.close()
    plt.style.use("dark_background")
    cp = sns.dark_palette("purple", 7)
    sns.lineplot(data=linearDf, x='Activation Duration [s]', y='Predicted', ci=None, color=cp[2])
    sns.scatterplot(data=linearDf, x='Activation Duration [s]', y='Time Coefficient [s]', linewidth=0, alpha=0.98, color=cp[6])
    plt.title('$R^2 = %.2f$' % (r2_score(speed_ups, predicted),), loc='left')
    plt.gca().grid(alpha=0.2)
    plt.show()
    pass





if __name__ == "__main__":
    exps = [('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_0.5S.avi_10.09.48', 0.5),
            ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_1S.avi_10.54.39', 1),
            ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_1.5S.avi_11.45.54', 1.5),
            ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_2S.avi_12.39.07', 2),
            ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_2.5S.avi_13.25.34', 2.5),
            ('/home/itskov/Temp/behav/19-Mar-2020/TPH_1_ATR_ONLINE[IAA]_3S.avi_14.14.43', 3)]


    analyzeSpeedUps(exps)
    pass
