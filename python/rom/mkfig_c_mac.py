"""Fig. (SOD-ROM): mass-weighted MAC between SOD smooth modes 1-8 (rows, smoothest first) and LNMs 1-8 (columns).
   Rows: beam with light Rayleigh damping; beam + tip dashpot (c_d=0.2). Columns: training signal. Data: c3_mac.pkl."""
import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plotprefs as pp
pp.apply_preferences(sizes=(10,9,8))
D=pickle.load(open('c3_mac.pkl','rb'))
fig,ax=plt.subplots(2,3,figsize=(7.0,5.0))
for i,sname in enumerate(['cantilever beam','cantilever beam + tip dashpot']):
    for j,tr in enumerate(['random','free decay','swept sine']):
        d=D[(sname,tr)]; Mac=d['MAC'][:8,:8]; a=ax[i,j]          # 8 x 8: the data excite no LNM above 8
        im=a.imshow(Mac,cmap='Blues',vmin=0,vmax=1,origin='upper',extent=(0.5,Mac.shape[1]+0.5,Mac.shape[0]+0.5,0.5))
        a.set_xticks(range(1,9)); a.set_yticks(range(1,Mac.shape[0]+1)); a.tick_params(labelsize=6)
        a.set_title(f"{'beam' if i==0 else 'beam + dashpot'}, {tr}\n(max modal corr. {d['cmax']:.2f})",fontsize=7)
        if j==0: a.set_ylabel('SOD mode')
        if i==1: a.set_xlabel('LNM')
fig.colorbar(im,ax=ax,shrink=0.6,label='MAC')
pp.latexify(fig); fig.savefig('../../figures/fig_c_mac.pdf',bbox_inches='tight'); fig.savefig('/tmp/claude-0/fig_c_mac.png',dpi=120,bbox_inches='tight')
