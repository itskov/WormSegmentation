from Behavior.Tools.Artifacts import Artifacts
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

art_120a = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D120_NO_IAA3x5.avi_12.57.03/')
art_120na = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D120_NO_IAA3x5.avi_12.56.12/')

art_180a = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D180_NO_IAA3x5.avi_15.29.04/')
art_180na = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D180_NO_IAA3x5.avi_15.28.03/')

art_240a = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D240_NO_IAA3x5.avi_17.28.52/')
art_240na = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D240_NO_IAA3x5.avi_17.28.04/')

art_300a = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_ATR_TRAIN_65M_D300_NO_IAA3x5.avi_20.12.12/')
art_300na = Artifacts(expLocation='/home/itskov/Temp/behav/22-Jan-2020/TPH_1_NO_ATR_TRAIN_65M_D300_NO_IAA3x5.avi_20.11.34/')


art_120a_frac = art_120a.getArtifact('roi')['arrivedFrac'][4499]
art_120na_frac = art_120na.getArtifact('roi')['arrivedFrac'][4499]

art_180a_frac = art_180a.getArtifact('roi')['arrivedFrac'][4499]
art_180na_frac = art_180na.getArtifact('roi')['arrivedFrac'][4499]

art_240a_frac = art_240a.getArtifact('roi')['arrivedFrac'][4499]
art_240na_frac = art_240na.getArtifact('roi')['arrivedFrac'][4499]

art_300a_frac = art_300a.getArtifact('roi')['arrivedFrac'][4499]
art_300na_frac = art_300na.getArtifact('roi')['arrivedFrac'][4499]


df = pd.DataFrame({'Opto': ['ATR+', 'ATR-', 'ATR+', 'ATR-', 'ATR+', 'ATR-','ATR+','ATR-'],
                   'Delay Period': ['120m','120m','180m','180m','240m','240m','300m','300m'],
                   'Fraction Arrived': [art_120a_frac, art_120na_frac, art_180a_frac,
                       art_180na_frac, art_240a_frac, art_240na_frac,
                       art_300a_frac, art_300na_frac]})

plt.style.use("dark_background")
sns.set_context("talk")
col_pal = ["#3C65B7", "#00A99C"]
sns.set_palette(sns.color_palette(col_pal))
g = sns.factorplot('Delay Period', 'Fraction Arrived', 'Opto',
                   data=df, kind="bar", legend=True)

# Show plot
plt.show()
