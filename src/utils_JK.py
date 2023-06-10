import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import koma
import seaborn as sns
sns.set_theme(context='paper',font_scale=1, style = 'ticks', color_codes='deep')

map=sns.color_palette("blend:b,orangered", as_cmap=True)

def MAC(v1,v2): 
    """Finding the MAC-number of vector v1 and v2"""
    teller=np.abs(v1.conj().T@v2)**2
    nevner=(v1.conj().T@v1)*(v2.conj().T@v2)
    return np.real(teller/nevner)

def rel_diff(a1,a2): 
    """Relative distance between a1 and a2"""
    return np.abs(a1-a2)/np.max(np.abs([a1,a2]))

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
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

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=-100000, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold =  im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    valfmt_0=("{x:.0f}")
    valfmt_0 = matplotlib.ticker.StrMethodFormatter(valfmt_0)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if data[i,j]==0: 

                text = im.axes.text(j, i, valfmt_0(data[i, j], None), **kw)
                texts.append(text)
            else:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                print(type(text))
                texts.append(text)

    return texts

def plot_result_heatmap(metric, m, d_track_threshold, title, metric_title, n_decimalpoints = '3', min = None, max = None):
    """
    metric: 7x7 3D matrix
    m: 1x7 2D matrix
    d_track_threshold: 1x7 2D matrix
    title: value of n

    Output: heatmap of the results as an image

    """
    fig, ax = plt.subplots()

    fig.set_size_inches(6,6)

    ax.set_title(title,fontsize = 30)
    ax.set_ylabel('$m$')
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel('$d_{threshold}$')

    im, cbar = heatmap(metric, m, d_track_threshold, ax=ax,
                    cmap=map, cbarlabel=metric_title, vmin = min, vmax = max, cbar_kw={'shrink': 0.65})

    texts = annotate_heatmap(im, valfmt="{x:."+n_decimalpoints+"f}")

    fig.tight_layout()

    return im, fig

def n_skipped_time_seg_clustering(freqs_out, phis_out, true_freqs_sorted, true_phi_sorted, lim = 0.01):
    """
    Function to calculate how many time the koma clustering fails to find an analytical mode over all time segments.

    Arguments
    ---------
    freqs_out: list of 1D arrays
        output frequencies over the time segments
    
    phis_out: list of 2D arrays
        output mode shapes over the time segments
    
    true_freqs_sorted: 2D array
        analytical frequencies over the time segments sorted by frequency on first day

    true_phi: 3D array
        analytical mode shapes over the time segments sorted by frequency on first day

    lim: float
        the lowest value of d = df + 1 - MAC to accept the koma clustering modes as equal to the analytical modes
    """

    n = np.zeros_like(true_freqs_sorted[0])

    for k, fk in enumerate(freqs_out):

        bool_list = np.full_like(n, False)

        for j, fkj in enumerate(fk):

            phikj = phis_out[k][j,:]

            df_list = np.array([])
            dmac_list = np.array([])
            d_list = np.array([])

            for i, truefki in enumerate(true_freqs_sorted[k]):
                df = rel_diff(fkj, truefki)
                dmac = 1 - MAC(phikj, true_phi_sorted[i,:,k])
                
                df_list = np.append(df_list, df)
                dmac_list = np.append(dmac_list, dmac)
                d_list = np.append(d_list, df + dmac)

            id_min = np.argmin(d_list)

            if d_list[id_min] < lim and not bool_list[id_min]: 
                bool_list[id_min] = True
            elif bool_list[id_min]:
                id_second_min = np.where(d_list == np.sort(d_list)[1])[0]
                if d_list[id_second_min] < lim: bool_list[id_second_min] = True

            
        for i, isadded in enumerate(bool_list):
            if not isadded: n[i] += 1

    return n

def rad2toHz(lambds):
    freqHz = []
    for lambd in lambds:
        freqHz.append(lambd**0.5/(2*np.pi))
    return freqHz

def radtoHz(ws):
    freqHz = []
    for w in ws:
        freqHz.append(w/(2*np.pi))
    return freqHz

def plot_physical_traces(mode_traces):
    fig, axs = plt.subplots(figsize = (7,5))

    for i, trace in enumerate(mode_traces):
        if not trace.isphysical: continue
        axs.scatter(trace.time_seg, trace.frequencies,
                    label = 'Mode trace {}'.format(i),s=0.3)
    
    axs.set_xlabel('Time segments')
    axs.set_ylabel('Frequency [Hz]')
    axs.legend()

    return axs

def n_skipped_time_seg_clustering2(freqs_out, phis_out, true_freqs_sorted, true_phi_sorted, df_lim = 0.1, dmac_lim = 0.1, printinfo = False):
    """
    Function to calculate how many time the koma clustering fails to find an analytical mode over all time segments.

    Arguments
    ---------
    freqs_out: list of 1D arrays
        output frequencies over the time segments
    
    phis_out: list of 2D arrays
        output mode shapes over the time segments
    
    true_freqs_sorted: 2D array
        analytical frequencies over the time segments sorted by frequency on first day

    true_phi_sorted: 3D array
        analytical mode shapes over the time segments sorted by frequency on first day

    df_lim: float
        the lowest value of df to accept the koma clustering modes as equal to the analytical modes

    dmac_lim: float
        the lowest value of 1 - mac to accept the koma clustering modes as equal to the analytical modes

    printinfo: bool
        option to print information about the mac number between the theoretical and clustered mode
    """

    n = np.zeros_like(true_freqs_sorted[0])

    dfs = np.zeros((true_freqs_sorted.shape[0], len(freqs_out)))
    macs = np.zeros_like(dfs)

    fs = np.zeros_like(dfs)

    for k, freqs_out_k in enumerate(freqs_out):
        
        bool_list = np.full_like(n, False)

        for j, freqs_out_kj in enumerate(freqs_out_k):

            phis_out_kj = phis_out[k][j,:]

            df_temp = np.array([])
            dmac_temp = np.array([]) 
            d_temp = np.array([])

            for i, true_mode_freq_i in enumerate(true_freqs_sorted[k]):
                true_phi_i = true_phi_sorted[i,:,k]

                df = rel_diff(true_mode_freq_i, freqs_out_kj)
                dmac = 1 - MAC(true_phi_i, phis_out_kj)

                df_temp = np.append(df_temp, df)
                dmac_temp = np.append(dmac_temp, dmac)
                d_temp = np.append(d_temp, df + dmac)

            id_min = np.argmin(d_temp)

            if df_temp[id_min] < df_lim and dmac_temp[id_min] < dmac_lim and not bool_list[id_min]:
                dfs[id_min, k] = df_temp[id_min]
                macs[id_min, k] = 1 - dmac_temp[id_min]
                fs[id_min, k] = freqs_out_kj 
                bool_list[id_min] = True
            #If two modes are closest to the same analytical mode in a time segment, add to second closest analytical mode
            elif bool_list[id_min]:
                id_second_min = np.where(d_temp == np.sort(d_temp)[1])[0]
                bool_list[id_second_min] = True
                dfs[id_second_min, k] = df_temp[id_min]
                macs[id_second_min, k] = 1 - dmac_temp[id_min]
                fs[id_second_min, k] = freqs_out_kj
            else:
                dfs[id_min, k] = df_temp[id_min]
                macs[id_min, k] = 1 - dmac_temp[id_min]
                fs[id_min, k] = freqs_out_kj 

            if printinfo:
                print('Time segment: {:.2f}'.format(k))
                print('-------------------------')
                print('The clustered mode is closest to analytical mode {}'.format(id_min))
                print('df: {:.2f}'.format(df_temp[id_min]))
                print('MAC: {:.2f}'.format(1-dmac_temp[id_min]))
                print('Analytical frequncy: {:.2f} Hz'.format(true_freqs_sorted[k][id_min]))
                print('Clustered frequency: {:.2f} Hz'.format(freqs_out_kj))
                print('Analytical mode shape:')
                print(true_phi_sorted[id_min,:,k])
                print('Clustered mode shape:')
                print(np.real(phis_out_kj))
                if df_temp[id_min] <= df_lim and dmac_temp[id_min] <= dmac_lim:
                    print('Accepted')
                else:
                    print('Declined')
                print('n:')
                print(n)
                print('\n \n \n \n \n')

        for i, isadded in enumerate(bool_list):
            if not isadded: 
                n[i] += 1
                dfs[i, k] = -1
                macs[i, k] = -1
                fs[i, k] = -1

    return n, fs, dfs, macs

def divide_modes(freqsHz, phis, freq_lim):
    '''
    Divide input modes based on a frequency limit

    Arguments
    ---------
    freqsHz: list of 1D arrays

    phis: list of 2D arrays

    freq_lim: float
        frequency in Hz for which to divide the modes
    '''

    freqsHz_below = []
    freqsHz_above = []

    phis_below = []
    phis_above = []

    for k, freqsHz_k in enumerate(freqsHz):
        phis_k = phis[k]

        id_all = np.arange(0, freqsHz_k.shape[0], 1)
        id_below = np.where(freqsHz_k < freq_lim)[0]
        id_above = np.setdiff1d(id_all, id_below)
        
        freqsHz_below.append(freqsHz_k[id_below])
        freqsHz_above.append(freqsHz_k[id_above])

        phis_below.append(phis_k[id_below])
        phis_above.append(phis_k[id_above])

    return freqsHz_above, freqsHz_below, phis_above, phis_below

def divide_modes2(freqsHz, phis, freq_lim):
    '''
    Divide input modes based on a frequency limit

    Arguments
    ---------
    freqsHz: 1D array

    phis: 2D array

    freq_lim: float
        frequency in Hz for which to divide the modes
    '''

    id_above = np.where(freqsHz > freq_lim)[0]
    id_below = np.where(freqsHz <= freq_lim)[0]

    freqsHz_above = freqsHz[id_above]
    freqsHz_below = freqsHz[id_below]
    phis_above = phis[id_above,:]
    phis_below = phis[id_below,:]

    return freqsHz_above, freqsHz_below, phis_above, phis_below

def distance_matrix(freqs, phis):
    '''
    Create the HDBSCAN clustering matrix
    '''
    #Prepare distance metric for clustering call
    n_modes = len(freqs)
    d_mat = np.zeros([n_modes, n_modes])

    #Calculate the distance between the input modes in a matrix format
    for k, fk in enumerate(freqs):
        phik = phis[k,:]
        for j, fj in enumerate(freqs[k:]):
            j+=k
            phij = phis[j,:]
            df_jk = rel_diff(fk, fj)
            mac_jk = MAC(phik, phij)
            d_jk = df_jk + 1 - mac_jk
            if d_jk < 1e-10:
                d_mat[k, j] = 1
            else:
                d_mat[k, j] = d_jk
                d_mat[j, k] = d_jk

    return d_mat

def distance_matrix_koma(freqs, phis):
    '''
    Create the HDBSCAN distance matrix using the KOMA package (faster)
    '''
    dmacs = np.abs(1 - koma.modal.xmacmat(phis.T))
    dfs = koma.clustering.crossdiff(freqs, relative=True, allow_negatives=False)

    return dfs + dmacs
