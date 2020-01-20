from Behavior.Tools.Artifacts import Artifacts
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

art_0a = Artifacts(expLocation='/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_NO_IAA3x5.avi_11.17.19')
art_0na = Artifacts(expLocation='/home/itskov/Temp/behav/15-Jan-2020/TPH_1_NO_ATR_TRAIN_60M_NO_IAA3x5.avi_11.16.35')

art_60a = Artifacts(expLocation='/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_D60_NO_IAA3x5.avi_14.24.06')
art_60na = Artifacts(expLocation='/home/itskov/Temp/behav/15-Jan-2020/TPH_1_NO_ATR_TRAIN_60M_D60_NO_IAA3x5.avi_14.23.14')

art_120a = Artifacts(expLocation='/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_D120_NO_IAA3x5.avi_17.27.06')
art_120na = Artifacts(expLocation='/home/itskov/Temp/behav/15-Jan-2020/TPH_1_NO_ATR_TRAIN_60M_D120_NO_IAA3x5.avi_17.26.18')


art_0a_frac = art_0a.getArtifact('roi')['arrivedFrac'][4000]
art_0na_frac = art_0na.getArtifact('roi')['arrivedFrac'][4000]

art_60a_frac = art_60a.getArtifact('roi')['arrivedFrac'][4000]
art_60na_frac = art_60na.getArtifact('roi')['arrivedFrac'][4000]

art_120a_frac = art_120a.getArtifact('roi')['arrivedFrac'][4000]
art_120na_frac = art_120na.getArtifact('roi')['arrivedFrac'][4000]

df = pd.DataFrame({'Opto': ['ATR+', 'ATR-', 'ATR+', 'ATR-', 'ATR+', 'ATR-'],
                   'Delay Period': ['0m','0m','60m','60m','120m','120m'],
                   'Fraction Arrived': [art_0a_frac, art_0na_frac, art_60a_frac, art_60na_frac, art_120a_frac, art_120na_frac]})

plt.style.use("dark_background")
sns.set_context("talk")
col_pal = ["#3C65B7", "#00A99C"]
sns.set_palette(sns.color_palette(col_pal))
g = sns.factorplot('Delay Period', 'Fraction Arrived', 'Opto',
                   data=df, kind="bar", legend=True)

# Show plot
plt.show()