import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Change the linewidth
sns.set(style="ticks", rc={"lines.linewidth": 5})


plt.interactive(False)
data_frame = pd.read_csv('/home/itskov/Dropbox/workspace/fraction_in_time.csv')
print(data_frame)
g = sns.lineplot(x='frame', y='arrived_frac', hue='cond', data=data_frame, ci=None)

# Change the ticks (setting the ticks from 1 to 1000, jumping by 200)
g.set(xticks=range(1,1000,200))

plt.show()

