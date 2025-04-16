import numpy as np                 ##this loads the module numpy as np
import pandas as pd                ##this loads the module pandas as pd
import matplotlib as mpl           ## this loads the moduule matplotlib as mpl
import matplotlib.pyplot as plt    ## this loads the moduule matplotlib.pyplot as plt

import scipy.io as sio
from scipy.signal import savgol_filter as sg  
from scipy.spatial.distance import pdist,squareform
import scipy.cluster.hierarchy as sch
from numpy.linalg import pinv,inv


## define some functions
def svdpcaf2(A,nlv):
    '''
    calculate the scores and loading vectors of a set of data
    usage:  s,L = svdpcaf(A,nlv)
    where A = the matrix of data A.shape = (nspec,npts)
    nlv = the number of loading vectors (integer)
    s = scores matrix s.shape = (nspec,nlv)
    L = is the loading vector matrix L.shape = (nlv, npts)
    requires 
    '''
    import numpy as np
    from numpy.linalg import svd
    U,SG, Vt = svd(A,full_matrices=True)
    S =  -1*(U@np.diag(SG))
    L = -1*Vt

    nspec,npts = A.shape
    F = (SG**2) / (nspec-1)
    #eig_val = sigma**2 / (A_.shape[0]-1)
    F = F / F.sum()

    return S[:,:nlv], L[:nlv,:],F

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="default")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_HM(ax,labels,d):
    '''
    This is a quick function to annotate a heat map.
    ax =  annotate_HM(ax,d)
    where ax is the axis to annotate
    d is the distance squareform array

    '''
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, '{:0.2f}'.format(d[i, j]),ha="center", va="center", color="w")
    return ax




#  load the data from a file
ifile = 'data/IR_which_marzipan_recipe.mat'
mdict = sio.loadmat(ifile)
print(mdict.keys())
Akn = mdict['Akn']
grp = mdict['grp']
xorg = mdict['xorg']
Au = mdict['Au']
labels= mdict['labels']

### make color array to color code groups (r,g,b)
colordict = {'A':(1,0,0), ## red
             'B':(0,0,1)  ##  blue
            }
colors = []
for g in grp:
    colors.append(colordict[g])

## pre-processing
## crop the data

rng = np.arange(542,929) ## fingerprint information rich region of spectrum
#rng = np.arange(len(Akn[0,:]))  ## whole spectrum

Ac = Akn[:,rng]   ## crop to fingerprint region
Auc = Au[:,rng]   ## crop unknown to the same region
xc = xorg[rng]     ## crop the x to the same region for plotting



## 2nd derivative (baseline de-emphasizing)
windowsize = 17      ## smoothing function size
porder = 3                        ##polynomial order must be larger than derivative
dorder = 2                    ## derivative order 0 = smooth, 1 = 1st derivative, 2 = 2nd derivative
Ad = sg(Ac,windowsize,polyorder=porder,deriv=dorder)     ## sg known data
Aud = sg(Auc,windowsize,polyorder=porder,deriv=dorder)   ## sg unknown data same way


## pair-wise Euclidean distance on known spectra,look for sets of spectra that are close in ED distance,
## these will be possible similar data
met = 'euclidean'
ed =squareform( pdist(Ad,metric=met) )

## calculate the linkages from the distances
Y = sch.linkage(Ad, method='average',metric='euclidean')

## PCA of known data
nspec, npts = Ad.shape
#calculate the scores
nlv = 6
s,L,F = svdpcaf2(Ad,nlv)
s = np.array(s)

## use the loading vectors for the known data to generate scores for the unknown data
Aud = np.matrix(Aud)  ## convert array to matrix
su = np.array(Aud*pinv(L))  ## convert to array for plotting purposes

## make a bunch of graphs
fig, axs = plt.subplots(3, 2, sharex=False, sharey=False ,gridspec_kw={'hspace': 0.5, 'wspace': 0.5},figsize=(18,18))
(ax1, ax2), (ax3, ax4), (ax5,ax6) = axs

## rawdata
crap = ax1.plot(xorg,Akn.T)
ax1.invert_xaxis()
ax1.set_xlabel('Wavenumber /cm$^{-1}$')
ax1.set_ylabel('Absorbance')
ax1.set_title('Raw data')


## cropped data
crap = ax2.plot(xc,Ac.T)
ax2.invert_xaxis()
ax2.set_xlabel('Wavenumber /cm$^{-1}$')
ax2.set_ylabel('Absorbance')
ax2.set_title('Cropped data')


## derivative spectra
crap = ax3.plot(xc,Ad.T)
ax3.invert_xaxis()
ax3.set_xlabel('Wavenumber /cm$^{-1}$')
ax3.set_ylabel('2$^{nd}$Derivative \n Absorbance')
ax3.set_title('Derivative Spectra')


## heatmap
im = heatmap(ed, labels, labels, ax=ax4, cmap="jet", cbarlabel="Euclidean distance")

## HCA
Z = sch.dendrogram(Y,labels=labels,color_threshold=0.01, show_leaf_counts=False, ax=ax5)

## scoreplot
## plot the scores
r = 0
c = 1

ax6.scatter(s[:,r],s[:,c],25,colors)  ## plot the known scores
ax6.scatter(su[:,r],su[:,c],25,'k')  ## plot the unknown scores
ax6.set_xlabel('Score 0')
ax6.set_ylabel('Score 1')

#add the text labels to the coordinates
for i in range(nspec):
    #ax.text(s[i,r],s[i,c],str(np.arange(nspec)[i]))
    #ax.text(s[i,r],s[i,c],labels[i][-1])
    ax6.text(s[i,r],s[i,c],grp[i][-1])               ## add grp label to points on scatter plot

ofile = 'figures/classificaiton_example.png'
fig.savefig(ofile,dpi=300)
# return ofile
