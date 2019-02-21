import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import natsort
import os

plt.rc('font', size=20)
plt.rcParams['lines.linewidth'] = 2.0
cmap = matplotlib.cm.get_cmap('magma')

fig, ax = plt.subplots(1, 1, facecolor='lavender', figsize=(8, 6))
fnames = os.listdir('logs/')
fnames = natsort.natsorted(fnames)
norm = len(fnames)
for i, fname in enumerate(fnames):
    col = cmap(i / norm)
    df = pd.read_csv('logs/' + fname, delim_whitespace=True)
    df.columns = ['r', 'Sigma']
    ax.plot(df.r, df.Sigma, color=col)
ax.set_xlabel(r'r')
ax.set_ylabel(r'$\Sigma$')
plt.tight_layout()
plt.savefig('surface_density_evolution.pdf', format='pdf',
            dpi=128, bbox_inches='tight', pad_inches=0.5)
plt.show()
