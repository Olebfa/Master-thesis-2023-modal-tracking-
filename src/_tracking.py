import numpy as np
import koma
import hdbscan
from utils_JK import MAC, rel_diff, distance_matrix_koma

def perform_tracking_He2022(frequenciesHz, phi):
    """Tracking algorithm according to paper 'Continuous Modal Identification and Tracking of a Long Span Suspension
    Bridge Using a Robust Mixed Clustering Method'
        
    Arguments
    ---------
    frequenciesHz : list with 1D arrays
        1st input: time segment
        2nd input: detected mode frequencies in Hz

        n detected modes
        frequenciesHz = [
            [f0_t0, f1_t0, ... , fn0_t0],
            [f0_t1, f1_t1, ... , fn1_t1],
            ...
            [f0_tend, f1_tend, ... , fnm_tend]
        ] 

    Phi : list with 2D arrays
        1st input: time segment
        2nd input: detected mode shape
        3rd input: mode inputs

        First day
        p-DOFS
        n0-detected modes
        Phi[0][:,:] = [
            [modeshape00_t0, modeshape01_t0, ... , modeshape0p_t0],
            [modeshape10_t0, modeshape11_t0, ... , modeshape1p_t0],
            ...
            [modeshapen00_t0, modeshapen01_t0, ... , modeshapen0p_t0]
        ] 

    Returns
    -------
    mode_traces:
        list of mode trace objects defines in this file

    """
    
    #Set modes from first day as reference modes
    ref_mode_frequenciesHz = frequenciesHz[0]
    ref_mode_modeshapes = phi[0]

    #Initialize mode traces
    mode_traces = []
    for k, ref_mode_frequencyHz in enumerate(ref_mode_frequenciesHz):
        mode_traces.append(ModeTrace(ref_frequency=ref_mode_frequencyHz, ref_mode_shape=ref_mode_modeshapes[k,:], init_time=0))
        mode_traces[-1].add_mode(ref_mode_frequencyHz, ref_mode_modeshapes[k,:], 0)

    #Iterate through the next time segments and track the modes
    for k, fk in enumerate(frequenciesHz):
        #Skip first time segment
        if k == 0: continue
        
        else:
            #Iterate through the modes at the current time segment
            for j, fkj in enumerate(fk):
                phikj = phi[k][j,:]

                #Initialize array for dj
                dj_temp = np.array([])

                for i, trace in enumerate(mode_traces):
                    macij = MAC(phikj, trace.ref_mode_shape)
                    dfij = rel_diff(fkj, trace.ref_frequency)
                    dj_temp = np.append(dj_temp, dfij + 1 - macij)

                #Identify which reference mode the current mode is closest to
                id_min = np.argmin(dj_temp)

                #Add to trace and set as new reference mode
                mode_traces[id_min].add_mode(fkj, phikj, k)
                mode_traces[id_min].set_ref_mode(fkj, phikj)
                mode_traces[id_min].add_diff(0, 0, djtrack = dj_temp[id_min])

    return mode_traces

def perform_tracking_magalhaes2008(frequenciesHz, phi, MAC_threshold = 0.8, df_threshold = 0.15):
    """Tracking algorithm according to paper 'Online Automatic Identification of the Modal Parameters of a long span
    arch bridge'
        
    Arguments
    ---------
    frequenciesHz : list with 1D arrays
        1st input: time segment
        2nd input: detected mode frequencies in Hz

        n detected modes
        frequenciesHz = [
            [f0_t0, f1_t0, ... , fn0_t0],
            [f0_t1, f1_t1, ... , fn1_t1],
            ...
            [f0_tend, f1_tend, ... , fnm_tend]
        ] 

    Phi : list with 2D arrays
        1st input: time segment
        2nd input: detected mode shape
        3rd input: mode inputs

        First day
        p-DOFS
        n0-detected modes
        Phi[0][:,:] = [
            [modeshape00_t0, modeshape01_t0, ... , modeshape0p_t0],
            [modeshape10_t0, modeshape11_t0, ... , modeshape1p_t0],
            ...
            [modeshapen00_t0, modeshapen01_t0, ... , modeshapen0p_t0]
        ] 

    Returns
    -------
    mode_traces:
        list of mode trace objects defines in this file

    """
    
    #Set modes from first day as reference modes
    ref_mode_frequenciesHz = frequenciesHz[0]
    ref_mode_modeshapes = phi[0]

    #Initialize mode traces
    mode_traces = []
    for k, ref_mode_frequencyHz in enumerate(ref_mode_frequenciesHz):
        mode_traces.append(ModeTrace(ref_frequency=ref_mode_frequencyHz, ref_mode_shape=ref_mode_modeshapes[k,:], init_time=0))
        mode_traces[-1].add_mode(ref_mode_frequencyHz, ref_mode_modeshapes[k,:], 0)
    
    #Iterate through the next time segments and track the modes
    for k, fk in enumerate(frequenciesHz):
        #Skip first time segment
        if k == 0: continue
        
        else:
            #Iterate through the modes at the current time segment
            for j, fkj in enumerate(fk):
                phikj = phi[k][j,:]

                #Initialize array for dfs and macs
                mac_temp = np.array([])
                df_temp = np.array([])

                for i, trace in enumerate(mode_traces):
                    macij = MAC(phikj, trace.ref_mode_shape)
                    dfij = rel_diff(fkj, trace.ref_frequency)
                    df_temp = np.append(df_temp, dfij)
                    mac_temp = np.append(mac_temp, macij)

                #Identify which reference mode the current mode is closest to
                id_max = np.argmax(mac_temp)

                #Add to trace and set as new reference mode if within criteria
                if df_temp[id_max] <= df_threshold and mac_temp[id_max] >= MAC_threshold:
                    mode_traces[id_max].add_mode(fkj, phikj, k)
                    mode_traces[id_max].set_ref_mode(fkj, phikj)

    return mode_traces

def gkern(theta, mu, std):
    """
    Gaussian kernel described in the relevant paper
    """
    return np.exp(-0.5*1/std**2*(theta-mu)**2)


def fsa_tilde(mu_pa, fsa1, sf):
    """
    Needed for trackingGaussian algorithm

    Arguments
    ---------
    mu_pa : 1d array
        all frequencies in the current time segment

    fsa1 : float
        fs from last time segment

    sf : float
        standard deviation of the gaussian kernel

    """
    kern = gkern(mu_pa, fsa1, sf)
    nom = np.sum(mu_pa*kern)
    denom = np.sum(kern)
    return nom/denom

def perform_tracking_favarelli(frequenciesHz, Nt = 200, N_lim = 250,  Bf = 0.4, epsilon = 0.7, sf = 0.01,*args,return_initial=False):
    """Tracking algorithm based on paper 'Machine Learning for Automatic Processing of Modal Analysis in Damage Detection
    of Bridges'

    Arguments
    ---------
    frequenciesHz : list with 1D arrays
        1st input: time segment
        2nd input: detected mode frequencies in Hz

        n detected modes
        frequenciesHz = [
            [f0_t0, f1_t0, ... , fn0_t0],
            [f0_t1, f1_t1, ... , fn1_t1],
            ...
            [f0_tend, f1_tend, ... , fnm_tend]
        ] 
    
    Nt : int
        Size of frequenciesHz to be used to estimate the initial frequencies

    N_lim : int
        Required amount of data points in a histogram column to be initialized as a starting frequency
    
    Bf : float
        Bandwith of bins in the histogram

    epsilon : float
        Controls the impact of the new observation. In the range of [0,1]

    sf : float
        Standard deviation of the Guassian kernel. Larger values of sf make the system more reactive to fast frequency changes during the tracking, but more sensitive to outliers due to the noisy measurements.

    plot_histogram : bool
        Specify as true if histogram is to be plotted

    Returns
    -------
    F_track : 2d array
        a sorted list of F containing the tracked frequencies over the time segments. The order is decided by the first time segment
    
    F_track = [
        [f0_t0], [f0_t1], ... , [f0_tend],
        [f1_t0], [f1_t1], ... , [f1_tend],
        ...
        [fn_t0], [fn_t1], ... , [fn_tend]
    ]
    
    """
    F_initial = frequenciesHz[:Nt]
    F_online = frequenciesHz[Nt:]

    #Initialize flattened lists of frequencies and shapes
    freqs_inital_flat = []

    #Add all the frequencies and shapes to the flattened lists
    for fk in F_initial:
        for fkj in fk:
            freqs_inital_flat.append(fkj)

    #Convert to numpy arrays (Easiest to append a 1D array to a python list and the convert to numpy)
    freqs_inital_flat = np.array(freqs_inital_flat)

    #Number of bins
    Nbins = int(np.ceil((freqs_inital_flat.max() - freqs_inital_flat.min())/Bf))

    #Make a histogram of the 
    hist, bin_edges = np.histogram(freqs_inital_flat, bins = Nbins, density = False)

    #Extract the initial frequencies
    id = np.array([i for i, v in enumerate(hist >= N_lim) if v]) #Find the highest peaks in the histogram

    #Find the bin edges for each of the detected initial frequency bins and reshape for easier for-loop
    found_bin_edges = np.array([])
    for i in id:
        found_bin_edges = np.append(found_bin_edges, np.array([bin_edges[i], bin_edges[i+1]]))
    found_bin_edges = found_bin_edges.reshape((int(found_bin_edges.shape[0]/2), 2))

    f0 = np.array([])
    


    for f_start, f_end in found_bin_edges:
        id_f = np.where(np.logical_and(freqs_inital_flat <= f_end, freqs_inital_flat >= f_start)) #Indices of frequencies in the bin
        f0 = np.append(f0, np.mean(freqs_inital_flat[id_f])) #Mean of the frequencies in the bin

    ###### OB ROT
    if return_initial:
        return f0

    #Initialize F_track with as many rows as identified frequencies and as many columns as the remaining time segments
    F_track = np.zeros([f0.shape[0], len(F_online)])
    F_track[:,0] = f0 #Copy initial frequencies

    for a, mu_pa in enumerate(F_online):
        if a == 0:
            continue
        else:
            for i in range(f0.shape[0]):
                fsa_tilde_i = fsa_tilde(mu_pa, F_track[i,a-1], sf)
                F_track[i,a] = (1-epsilon)*F_track[i,a-1] + epsilon*fsa_tilde_i
                
    return F_track

class ModeTrace():
    def __init__(self, ref_frequency, ref_mode_shape, init_time,  df_threshold = 0.1, dmac_threshold = 0.1):
        """
        A class for a mode trace over time. Currently uses mean of next modes for new reference mode or linear extrapolation.

        Arguments
        --------
        ref_frequency: float
            initial reference frequency
        
        ref_mode_shape: 1D array
            initial reference mode shape

        df_threshold: float
            initial maximum difference in frequency from the reference mode a candiate mode can have

        dmac_threshold: float
            initial maximum 1 - MAC from the reference mode a candidate mode can have
        """

        self.ref_frequency = ref_frequency
        self.ref_mode_shape = ref_mode_shape

        self.frequencies = []
        self.mode_shapes = []
        self.time_seg = []
        
        self.ref_mode_frequency_temp = []
        self.ref_mode_shape_temp = []
        self.ref_time_stamp_temp = []

        self.n_skipped_time_seg = init_time
        self.n_false_positives = 0

        self.dhf = []
        self.dhmac = []

        self.isphysical = True

        self.candidate_modes_freqs = []
        self.candidate_modes_shapes = []
        self.candidate_modes_ds = []
        self.candidate_modes_df = []
        self.candidate_modes_dmac = []

        self.df_threshold = df_threshold
        self.dmac_threshold = dmac_threshold
        self.df_threshold_list = [df_threshold]
        self.dmac_threshold_list = [dmac_threshold]

        self.djtrack_list = []


    def add_mode(self, f, phi, time_stamp):
        """
        Add a mode to the trace
        """
        self.frequencies.append(f)
        self.mode_shapes.append(phi)
        self.time_seg.append(time_stamp)

    def set_ref_mode(self, new_f_reference, new_mode_shape_reference):
        """
        Manually set a reference mode. Not used in current scheme
        """
        self.ref_frequency = new_f_reference
        self.ref_mode_shape = new_mode_shape_reference

    def add_mode_for_next_ref_mode(self, f_ref, mode_shape_ref, time_stamp_ref):
        """
        Add a mode to be used when calculating the next reference mode in the funtion "set_next_ref_mode"
        """
        self.ref_mode_frequency_temp.append(f_ref)
        self.ref_mode_shape_temp.append(mode_shape_ref)
        self.ref_time_stamp_temp.append(time_stamp_ref)

    def set_next_ref_mode(self, method = 'average'):
        """
        Calculate the next reference mode based on the added modes. Reset the temp list for modes to be used for calculation.
        """

        if len(self.ref_mode_frequency_temp) == 0 or len(self.ref_mode_frequency_temp) == 1:
            pass
        
        else:

            phi_temp_norm,_ = koma.modal.normalize_phi(np.array(self.ref_mode_shape_temp).T)
            phi_temp_norm = koma.modal.align_modes(phi_temp_norm)
            phi_temp_norm = phi_temp_norm.T    
            
            if method == 'average':

                self.ref_frequency = np.mean(self.ref_mode_frequency_temp)
                self.ref_mode_shape = np.mean(phi_temp_norm,axis = 0)

            elif method == 'linear_extrapolation':

                X = np.vstack([np.ones(len(self.ref_time_stamp_temp)), self.ref_time_stamp_temp]).T
                
                try:
                    bf = np.linalg.solve(X.T @ X, X.T @ self.ref_mode_frequency_temp)

                    self.ref_frequency = bf[0] + bf[1] *(X[-1, -1] + 1)

                    phi_next = np.zeros_like(self.ref_mode_shape)

                    for i in range(phi_next.shape[0]):
                        yi = phi_temp_norm[:,i]
                        bi = np.linalg.solve(X.T @ X, X.T @ yi)
                        phi_next[i] = bi[0] + bi[1] *(X[-1, -1] + 1)

                    self.ref_mode_shape = phi_next

                #Rare case of a division containing two modes at the same time segment. Use average
                except:
                    self.ref_frequency = np.mean(self.ref_mode_frequency_temp)
                    self.ref_mode_shape = np.mean(phi_temp_norm,axis = 0)


            self.ref_mode_frequency_temp = []
            self.ref_mode_shape_temp = []
            self.ref_time_stamp_temp = []

    def add_diff(self, dhf, dhmac, djtrack = 0):
        '''
        Add difference measure from the reference mode at the current time stamp
        '''
        self.dhf.append(dhf)
        self.dhmac.append(dhmac)
        self.djtrack_list.append(djtrack)

    def check_mode_trace(self, tcheck, k):
        '''
        Check if a mode trace most likely containts physical or mathematical modes, 
        based on the distance in time between the two last added modes. Return False if the distance is too large.
        '''
        #Don't call this function at the first time segment

        #Wait for the next check if only one mode has been added in the current 'checking interval',
        #could discover a mode at the end.
        if len(self.time_seg) == 1 and np.abs(self.time_seg[0] - k) < tcheck:
            return True
        
        elif len(self.time_seg) == 1:
            return False
        
        else:
            tdist = self.time_seg[-1] - self.time_seg[-2]
            if tdist >= tcheck:
                return False
            elif self.time_seg[-1] + tcheck < k:
                return False
            else:
                return True
        
    def set_physicality(self, bol):
        '''
        Set the physicality of the trace
        '''
        self.isphysical = bol

    def add_candidate_mode(self, freq, phi, d, df, dmac):
        '''
        Add a candidate mode
        '''
        self.candidate_modes_freqs.append(freq)
        self.candidate_modes_shapes.append(phi)
        self.candidate_modes_ds.append(d)
        self.candidate_modes_df.append(df)
        self.candidate_modes_dmac.append(dmac)

    def reset_candidate_modes(self):
        '''
        Reset the candidate modes list
        '''
        self.candidate_modes_freqs = []
        self.candidate_modes_shapes = []
        self.candidate_modes_ds = []
        self.candidate_modes_df = []
        self.candidate_modes_dmac = []

    def set_threshold(self, df_threshold_new, dmac_threshold_new):
        '''
        Set new thresholds
        '''
        self.df_threshold = df_threshold_new
        self.dmac_threshold = dmac_threshold_new
        self.df_threshold_list.append(df_threshold_new)
        self.dmac_threshold_list.append(dmac_threshold_new)

    def count_missed_time_segments(self, freqs_out, phis_out, true_mode_f, true_mode_phi, n = 0, lim = False, df_max = False, dmac_max = False):
        '''
        Postprocessing tool to count how many times the trace missed a mode in a segment, or chose a mode which does not
        belong in the trace to the trace.

        Two cases:
        1) Freqs out contains the analytical mode within the limit, but the trace did not find it
        2) Freqs out does not contain the analytical mode within the limit, but trace contains a mode at the current time segment

        Arguments
        ------------
        freqs_out: list of 1D arrays
            AOMA frequencies at each time segment
        phis_out: list of 2D arrays
            AOMA mode shapes at each time segment
        true_mode_f: 1D array
            True mode frequency at each time segment
        true_mode_phi: 2D array
            True mode shape at each time segment, stacked column-wise.
            true_mode_phi[:,k] = true mode shape at k-th day
        '''
        for k, fk in enumerate(freqs_out):

            #Skip time segment if there are no modes. Impossible for the trace to contain any modes
            if len(fk) == 0: continue

            df_list = np.array([])
            dmac_list = np.array([])
            d_list = np.array([])
            
            for j, fkj in enumerate(fk):
                phikj = phis_out[k][j,:]
                
                dfkj = rel_diff(fkj, true_mode_f[k])
                dmackj = 1 - MAC(phikj, true_mode_phi[:,k])

                df_list = np.append(df_list, dfkj)
                dmac_list = np.append(dmac_list, dmackj)
                d_list = np.append(d_list, dfkj + dmackj)

            if lim:

                id_min = np.argmin(d_list)

                id_trace = np.where(np.array(self.time_seg) == (k + n))[0]
                
                if d_list[id_min] < lim:

                    #If the AOMA mode is close enough to the analytical mode, but the trace does not contain any modes: missed time segment
                    if len(id_trace) == 0: 
                        self.n_skipped_time_seg += 1

                    #If the AOMA mode is close enough to the analytical mode, but the trace contains more than one mode: false positive
                    elif len(id_trace) == 2:
                        self.n_false_positives += 1

                    #If the AOMA mode is close enough to the analytical mode, but the trace does not contain the AOMA mode: false positive
                    elif MAC(np.array(self.mode_shapes)[id_trace][0], phis_out[k][id_min,:]) < 0.98:
                        self.n_false_positives += 1    
                else:
                    #The AOMA-step failed to discover the mode. If the trace contains a mode at the current time segment: false positive
                    if len(id_trace) != 0: 
                        self.n_false_positives += 1

            else:
                id_max = np.argmin(dmac_list)

                id_trace = np.where(np.array(self.time_seg) == k)[0]
                
                if df_list[id_max] < df_max and dmac_list[id_max] < dmac_max:

                    #If the AOMA mode is close enough to the analytical mode, but the trace does not contain any modes: missed time segment
                    if len(id_trace) == 0: 
                        self.n_skipped_time_seg += 1

                    #If the AOMA mode is close enough to the analytical mode, but the trace contains more than one modes: false positive
                    elif len(id_trace) == 2:
                        self.n_false_positives += 1

                    #If the AOMA mode is close enough to the analytical mode, but the trace does not contain the AOMA mode: false positive
                    elif MAC(np.array(self.mode_shapes)[id_trace][0], phis_out[k][id_max,:]) < 0.95:
                        self.n_false_positives += 1   

                else:
                    #The AOMA-step failed to discover the mode. If the trace contains a mode at the current time segment: false positive
                    if len(id_trace) != 0: 
                        self.n_false_positives += 1

def sort_traces(traces, init_true_freqs_sorted, init_true_phis_sorted):
    '''
    Sort the mode traces in the same orded as the sorted true modes.

    Only works when there are equal amount of traces as true traces
    '''

    traces_sorted = []

    for i, init_true_f in enumerate(init_true_freqs_sorted):
        init_true_phi = init_true_phis_sorted[i,:]

        d_temp = np.array([])
        for j, trace in enumerate(traces):
            if len(trace.frequencies) == 0:
                dij = rel_diff(trace.ref_frequency, init_true_f) + 1 - MAC(trace.ref_mode_shape, init_true_phi)
            else:
                dij = rel_diff(trace.frequencies[0], init_true_f) + 1 - MAC(trace.mode_shapes[0], init_true_phi)
            d_temp = np.append(d_temp, dij)

        traces_sorted.append(traces[np.argmin(d_temp)])

    return traces_sorted


def get_initial_modes(freqs_inital, phis_initial):
    """
    Calculate the initial reference modes by clustering the initial phase based only on frequency

    Arguments
    ----------
    freqs_initial: list of 1D arrays
        frequencies from the time segments in the initial phase

    phis_initial: list of 2D arrays
        mode shapes from the time segments in the initial phase. Row-wise modes in the 2D arrays

    Returns
    ---------
    mode_traces_initial: list
        list of mode objects from the class ModeTrace
    """
    
    #Initialize flattened lists of frequencies and shapes
    freqs_inital_flat = []
    phis_inital_flat = []

    #Add all the frequencies and shapes to the flattened lists
    for k, fk in enumerate(freqs_inital):
        phik = phis_initial[k]
        for j, fkj in enumerate(fk):
            phikj = phik[j,:]
            freqs_inital_flat.append(fkj)
            phis_inital_flat.append(phikj)

    #Convert to numpy arrays (Easiest to append a 1D array to a python list and the convert to numpy)
    freqs_inital_flat = np.array(freqs_inital_flat)
    phis_inital_flat = np.array(phis_inital_flat)

    #Calculate the distance between the input modes in a matrix format
    d_mat = distance_matrix_koma(freqs_inital_flat, phis_inital_flat)

    #Clustering call
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=5 ,min_cluster_size=10, alpha=1.0, gen_min_span_tree=False)
    clusterer.fit(d_mat)
 
    #Initialize list of initial mode traces
    mode_trace_initial = np.array([])

    #Initialize list for sorting based on frequency
    freq_temp = np.array([])

    #Calculate the mean mode shape and frequency and create mode trace instances for the detected modes
    for label in range(clusterer.labels_.max()+1):
            
        f0_temp = freqs_inital_flat[np.where(clusterer.labels_ == label)[0]]
        phi0_temp = phis_inital_flat[np.where(clusterer.labels_ == label)[0]]

        phi0_temp, _ = koma.modal.normalize_phi(phi0_temp.T)
        phi0_temp = koma.modal.align_modes(phi0_temp)
        phi0_temp = phi0_temp.T 

        mode_trace_i = ModeTrace(ref_frequency=np.mean(f0_temp), ref_mode_shape=np.mean(phi0_temp,axis = 0),init_time=0)
        mode_trace_initial = np.append(mode_trace_initial, mode_trace_i)

        freq_temp = np.append(freq_temp, np.mean(f0_temp))

    #Sort the traces based on frequency
    idsort = np.argsort(freq_temp)
    mode_trace_initial = mode_trace_initial[idsort]

    return mode_trace_initial

def get_initial_modes2(freqs_inital, phis_initial):
    """
    Calculate the initial reference modes by clustering the initial phase in two steps. New

    Arguments
    ----------
    freqs_initial: list of 1D arrays
        frequencies from the time segments in the initial phase

    phis_initial: list of 2D arrays
        mode shapes from the time segments in the initial phase. Row-wise modes in the 2D arrays

    Returns
    ---------
    mode_traces_initial: list
        list of mode objects from the class ModeTrace
    """
    
    #Initialize flattened lists of frequencies and shapes
    freqs_inital_flat = []
    phis_inital_flat = []

    #Add all the frequencies and shapes to the flattened lists
    for k, fk in enumerate(freqs_inital):
        phik = phis_initial[k]
        for j, fkj in enumerate(fk):
            phikj = phik[j,:]
            freqs_inital_flat.append(fkj)
            phis_inital_flat.append(phikj)

    #Convert to numpy arrays (Easiest to append a 1D array to a python list and the convert to numpy)
    freqs_inital_flat = np.array(freqs_inital_flat)
    phis_inital_flat = np.array(phis_inital_flat)

    #Get distance matrix for first clustering step
    d_mat1 = distance_matrix_koma(freqs_inital_flat, phis_inital_flat)

    #Clustering call
    clusterer1 = hdbscan.HDBSCAN(metric='precomputed', min_samples=5 ,min_cluster_size=10, alpha=1.0, gen_min_span_tree=False)
    clusterer1.fit(d_mat1)
 
    #Inizialize a list for the frequencies from the first clustering step
    f1 = np.array([])

    #Find f_split
    for label in range(clusterer1.labels_.max()+1):
        f1_temp = freqs_inital_flat[np.where(clusterer1.labels_ == label)[0]]
        f1 = np.append(f1, np.mean(f1_temp))

    f1_sorted = np.sort(f1)
    max_difference_index = np.argmax(np.diff(f1_sorted))
    f_split = (f1_sorted[max_difference_index + 1] + f1_sorted[max_difference_index])/2 

    #Extract divided modes
    id_div_above = np.where(freqs_inital_flat > f_split)[0]
    id_all = np.arange(freqs_inital_flat.shape[0])
    id_div_below = np.setdiff1d(id_all, id_div_above)
    
    freqs_inital_flat_above = freqs_inital_flat[id_div_above]
    freqs_inital_flat_below = freqs_inital_flat[id_div_below]

    phis_inital_flat_above = phis_inital_flat[id_div_above]
    phis_inital_flat_below = phis_inital_flat[id_div_below]

    #Create d_mats for step 2
    d_mat2_above = d_mat1[id_div_above][:,id_div_above]
    d_mat2_below = d_mat1[id_div_below][:,id_div_below]
    
    #Cluster call
    clusterer2_above = hdbscan.HDBSCAN(metric='euclidean', min_samples=5 ,min_cluster_size=10, alpha=1.0, gen_min_span_tree=False)
    clusterer2_above.fit(d_mat2_above)

    clusterer2_below = hdbscan.HDBSCAN(metric='euclidean', min_samples=5 ,min_cluster_size=10, alpha=1.0, gen_min_span_tree=False)
    clusterer2_below.fit(d_mat2_below)

    #Initialize list of initial mode traces
    mode_trace_initial = np.array([])

    #Initialize list for sorting based on frequency
    freq_temp = np.array([])

    #Calculate the mean mode shape and frequency and create mode trace instances for the detected modes for frequencies above split
    for label in range(clusterer2_above.labels_.max()+1):
            
        f0_temp1 = freqs_inital_flat_above[np.where(clusterer2_above.labels_ == label)[0]]
        phi0_temp1 = phis_inital_flat_above[np.where(clusterer2_above.labels_ == label)[0]]

        phi0_temp1, _ = koma.modal.normalize_phi(phi0_temp1.T)
        phi0_temp1 = koma.modal.align_modes(phi0_temp1)
        phi0_temp1 = phi0_temp1.T 

        mode_trace_i = ModeTrace(ref_frequency=np.mean(f0_temp1), ref_mode_shape=np.mean(phi0_temp1,axis = 0),init_time=0)
        mode_trace_initial = np.append(mode_trace_initial, mode_trace_i)

        freq_temp = np.append(freq_temp, np.mean(f0_temp1))

    #Calculate the mean mode shape and frequency and create mode trace instances for the detected modes for frequencies below split
    for label in range(clusterer2_below.labels_.max()+1):
            
        f0_temp2 = freqs_inital_flat_below[np.where(clusterer2_below.labels_ == label)[0]]
        phi0_temp2 = phis_inital_flat_below[np.where(clusterer2_below.labels_ == label)[0]]

        phi0_temp2, _ = koma.modal.normalize_phi(phi0_temp2.T)
        phi0_temp2 = koma.modal.align_modes(phi0_temp2)
        phi0_temp2 = phi0_temp2.T 

        mode_trace_i = ModeTrace(ref_frequency=np.mean(f0_temp2), ref_mode_shape=np.mean(phi0_temp2,axis = 0),init_time=0)
        mode_trace_initial = np.append(mode_trace_initial, mode_trace_i)

        freq_temp = np.append(freq_temp, np.mean(f0_temp2)) 

    #Sort the traces based on frequency
    idsort = np.argsort(freq_temp)
    mode_trace_initial = mode_trace_initial[idsort]

    return mode_trace_initial

def divide_segment(inlist, m):
    """
    Return a list of the divided segments of lenght m of the inlist
    """
    if len(inlist)%m == 0:
        divlist = np.arange(0, len(inlist)+1, m)
    else:
        divlist = np.arange(0, len(inlist), m)
        divlist = np.append(divlist, divlist[-1]+len(inlist)%m)
    return divlist

def perform_tracking(freqs, phis, n = 20, m = 5, d_track_threshold = 0.1, method = 'average', split = False):
    """
    Perform mode tracking on modes over time.

    Arguments
    ----------
    freqs: list of 1D arrays
        frequencies over time. Each array comes from seperate time segments
    
    phis: list of 2D arrays
        mode shapes over time

    n: int
        Amount of time segments from the inital phase to be used for calculating the initial reference modes

    m: int
        Amount of time segments ahead of each reference point to be used for calculation of the next reference modes

    d_track_treshold: float
        limit for highest value of dj_track to qualify a mode for a trace

    method: string
        either 'average' or 'linear_extrapolation'
    
    split: bool
        split the initial phase in two frequency bands based on algorithm 1 from the thesis
        
    Returns
    ---------
    mode_traces: list
        List of mode trace object defined according to the mode trace class in this file

    """
    freqs_inital = freqs[:n]
    phis_initial = phis[:n]

    freqs_online = freqs[n:]
    phis_online = phis[n:]

    if split:
        mode_traces = get_initial_modes2(freqs_inital, phis_initial)
    else:
        mode_traces = get_initial_modes(freqs_inital, phis_initial)
    
    online_div = divide_segment(freqs_online, m)

    for seg_i in range(online_div.shape[0]-1):

        for k, fk in enumerate(freqs_online[online_div[seg_i]:online_div[seg_i+1]]):

            for j, fkj in enumerate(fk):
                phikj = phis_online[online_div[seg_i]:online_div[seg_i+1]][k][j,:]

                dj_track_temp = np.array([])
                for mode_trace in mode_traces:
                    df = rel_diff(fkj, mode_trace.ref_frequency)
                    mac_kj = MAC(phikj, mode_trace.ref_mode_shape)
                    dj_track = df + 1 - mac_kj
                    dj_track_temp = np.append(dj_track_temp, dj_track)

                #Skip this if there are no AOMA modes in the current time segment
                if len(dj_track_temp) != 0:

                    id_mode = np.argmin(dj_track_temp)

                    #Add all modes that qualify for a trace as a candidate
                    if dj_track_temp[id_mode] <= d_track_threshold:
                        mode_traces[id_mode].add_candidate_mode(fkj, phikj, d = dj_track_temp[id_mode], df = 0, dmac = 0)

            for i, trace in enumerate(mode_traces):
                
                #Skip trace if no modes qualify for it
                if not trace.candidate_modes_ds: continue

                #Only add the closest candidate mode, discard the rest
                id_min = np.argmin(trace.candidate_modes_ds)
                mode_traces[i].add_mode(f = trace.candidate_modes_freqs[id_min], phi = trace.candidate_modes_shapes[id_min], time_stamp = online_div[seg_i] + k + n)
                mode_traces[i].add_mode_for_next_ref_mode(trace.candidate_modes_freqs[id_min], trace.candidate_modes_shapes[id_min], time_stamp_ref = k)
                mode_traces[i].reset_candidate_modes()

    
        for i in range(len(mode_traces)):
            mode_traces[i].set_next_ref_mode(method)

    return mode_traces

def perform_tracking_cabboi(freqsHz, phis, ref_mode_frequenciesHz, ref_mode_modeshapes, dfi_max = 0.1, dmaci_max = 0.1):
    """
    Perform mode tracking on modes over time using an updated threshold. 

    Arguments
    ----------
    freqsHz: list of 1D arrays
        frequencies over time in Hz. Each array comes from seperate time segments
    
    phis: list of 2D arrays
        mode shapes over time

    ref_mode_frequenciesHz: 1D array
        sorted mode frequencies deemed to be true after a day of monitoring

    ref_mode_modeshapes: 2D array
        sorted mode shapes deemed to be true after a day of monitoring

    dfi_max: float
        starting frequency difference threshold for each mode trace
    
    dmaci_max: float
        starting 1 - mac threshold for each mode trace
    
    Returns
    ---------
    mode_traces: list
        List of mode trace object defined according to the mode trace class in this file

    """

    #Initialize mode traces
    mode_traces = []
    for k, ref_mode_frequencyHz in enumerate(ref_mode_frequenciesHz):
        mode_traces.append(ModeTrace(ref_frequency=ref_mode_frequencyHz, ref_mode_shape=ref_mode_modeshapes[k,:], init_time=0, df_threshold=dfi_max, dmac_threshold=dmaci_max))

    #Iterate through the time segments and trace modes
    for h, fh in enumerate(freqsHz):
        for j, fhj in enumerate(fh):
            phihj = phis[h][j,:]
            for i, mode_trace_i in enumerate(mode_traces):

                dhf_j = rel_diff(mode_trace_i.ref_frequency, fhj)
                dhmac_j = 1 - MAC(mode_trace_i.ref_mode_shape, phihj)

                if dhf_j <= mode_trace_i.df_threshold and dhmac_j <= mode_trace_i.dmac_threshold:
                    mode_traces[i].add_mode(f = fhj, phi = phihj, time_stamp = h)
                    mode_traces[i].add_diff(dhf = dhf_j, dhmac = dhmac_j)


        #Update the thresholds after each time segment if there is added more than two modes
        if h > 1:
            for i in range(len(mode_traces)):
                if len(mode_traces[i].dhf) > 1:
                    df_max_temp = np.sqrt(np.std(mode_traces[i].dhf)*(1 + 1/(np.log(h))))
                    dmac_max_temp = np.sqrt(np.std(mode_traces[i].dhmac)*(1 + 1/(np.log(h))))

                    mode_traces[i].set_threshold(df_max_temp, dmac_max_temp)

    return mode_traces

def perform_tracking_continuous(freqsHz, phis, tcheck = 10, m = 5, df_max = 0.1, dmac_max = 0.1, update_ref = False, update_thresholds = False):
    '''
    Perform tracking of poles with the following attributes:
    1) Select all poles from the first day as the reference modes
    2) For the subsequent days select the closest poles to a trace, based on df + 1 - mac, 
    and check if df and mac are below the criteria.
    3) If the closest pole to a trace fullfills the criteria, select it as a candidate pole.
    4) For all candidate poles to a trace, add the closest based on df + 1 - mac, and start a new trace for the rest.
    5) If a pole fails to qualify for any of the traces, start a new trace for this pole.
    6) Label a trace as not physical (isphyscial = False) if no poles where added at an interval of at least tcheck.
    7) Update the reference modes if specified at the end of each interval of length m
    8) Update the thresholds if specified each time a new mode is added to a trace (and there are more than two modes in
    the trace from before)

    Arguments
    ----------
    freqsHz: list of 1D arrays
        frequencies over time in Hz. Each array comes from seperate time segments
    
    phis: list of 2D arrays
        mode shapes over time

    tcheck: int
        interval to check for physicality

    m: int
        interval to update reference modes

    dfi_max: float
        starting frequency difference threshold for each mode trace
    
    dmaci_max: float
        starting 1 - mac threshold for each mode trace

    update_ref: boolean
        if False, use the poles at the first time segment as the reference modes throughout the algorithm.
        if True, update the reference modes at an interval of m, by taking the average of the modes added 
        in the previous interval of length m

    update_thresholds: boolean
        if True, updates the thresholds of each trace according to cabboi 2016
    '''
    mode_traces = []

    freqsHz0 = freqsHz[0]
    phi0 = phis[0]


    #Initialize the traces with the poles of the first day
    for i, fi in enumerate(freqsHz0):
        mode_traces.append(ModeTrace(ref_frequency=fi, ref_mode_shape=phi0[i,:], init_time=0, df_threshold = df_max, dmac_threshold = dmac_max))
        mode_traces[-1].add_mode(f = fi, phi = phi0[i,:], time_stamp = 0)

    #Iterate through the time segments
    for k, fk in enumerate(freqsHz):
        
        if k == 0: continue

        for j, fkj in enumerate(fk):
            phikj = phis[k][j,:]

            #Initialize a boolean for the case of a pole not qualifying for a trace
            isadded = False

            #Use d = df + 1 - mac as a criteria to find the nearest mode to a trace
            d_temp = np.array([])

            #Store the dfs and macs to call after
            df_temp = np.array([])
            dmac_temp = np.array([])

            #Check the pole with each of the traces
            for i, mode_trace_i in enumerate(mode_traces):
                
                #Only consider physical traces
                if mode_trace_i.isphysical:
                    dfj = rel_diff(fkj, mode_trace_i.ref_frequency)
                    dmacj = 1 - MAC(phikj, mode_trace_i.ref_mode_shape)
                    
                    df_temp = np.append(df_temp, dfj)
                    dmac_temp = np.append(dmac_temp, dmacj)
                    d_temp = np.append(d_temp, dfj + dmacj)

                #Append a large number to keep correct indexing
                else:
                    df_temp = np.append(df_temp, 100)
                    dmac_temp = np.append(dmac_temp, 100)
                    d_temp = np.append(d_temp, 100)

            id_min = np.argmin(d_temp)

            #Add mode to candidate modes of a trace if df and mac are acceptable and the trace is still physical
            if df_temp[id_min] <= mode_traces[id_min].df_threshold and dmac_temp[id_min] <= mode_traces[id_min].dmac_threshold:
                
                isadded = True
                mode_traces[id_min].add_candidate_mode(freq = fkj, phi = phikj, d = d_temp[id_min], df = df_temp[id_min], dmac = dmac_temp[id_min])
            
            #If a mode does not qualify for a trace, start a new one
            if not isadded:
                mode_traces.append(ModeTrace(ref_frequency=fkj, ref_mode_shape=phikj,init_time=k, df_threshold = df_max, dmac_threshold = dmac_max))
                mode_traces[-1].add_mode(f = fkj, phi = phikj, time_stamp=k)

        new_traces = []

        #Go through the candidate modes and choose the closest mode to add to the trace. Start a new trace for the rest
        for i, trace in enumerate(mode_traces):
            
            #Skip trace if no modes qualify for it
            if not trace.candidate_modes_ds: continue

            id_min = np.argmin(trace.candidate_modes_ds)
            mode_traces[i].add_mode(f = trace.candidate_modes_freqs[id_min], phi = trace.candidate_modes_shapes[id_min], time_stamp = k)
            mode_traces[i].add_diff(dhf = trace.candidate_modes_df[id_min], dhmac = trace.candidate_modes_dmac[id_min])

            freqs_leftover = [f for id, f in enumerate(trace.candidate_modes_freqs) if id != id_min]
            phis_leftover = [phi for id, phi in enumerate(trace.candidate_modes_shapes) if id != id_min]

            #Skip creation of new trace if there is only one candidate mode at a time segment
            if not freqs_leftover: continue

            #Create new traces for the rest of the candidate poles
            for j, freq_leftover in enumerate(freqs_leftover):
                new_traces.append(ModeTrace(ref_frequency=freq_leftover, ref_mode_shape=phis_leftover[j], init_time=k, df_threshold = df_max, dmac_threshold = dmac_max))
                new_traces[-1].add_mode(f = freq_leftover, phi = phis_leftover[j], time_stamp = k)
        
        #Add the new traces to the end of the mode traces list if there are any to be added
        if new_traces: 
            for new_trace in new_traces: mode_traces.append(new_trace)

        #Reset the candidate list and check if a mode was added to each trace in the current segment
        for i in range(len(mode_traces)): 
            mode_traces[i].reset_candidate_modes()

        #Check for physicality each tcheck interval
        if k%tcheck == 0 and k > 0:
            for i in range(len(mode_traces)):
                if not mode_traces[i].check_mode_trace(tcheck, k):
                    mode_traces[i].set_physicality(False)
                
        #Update the reference modes if specified
        if update_ref and k%m == 0 and k > 0:
            for i in range(len(mode_traces)):
                idupd = np.where(np.array(mode_traces[i].time_seg) > k - m)[0]
                
                update_timeseg = np.array(mode_traces[i].time_seg)[idupd]
                update_freqs = np.array(mode_traces[i].frequencies)[idupd]
                update_shapes = np.array(mode_traces[i].mode_shapes)[idupd]

                for j, t in enumerate(update_timeseg):
                    mode_traces[i].add_mode_for_next_ref_mode(update_freqs[j], update_shapes[j], t)

                mode_traces[i].set_next_ref_mode(method = 'average')

        #Update thresholds after Cabboi if specified
        if k > 1 and update_thresholds:
            for i in range(len(mode_traces)):

                if len(mode_traces[i].dhf) > 1:

                    #Use rules from article to calculate the new thresholds
                    f_threshold_new_i = np.sqrt(np.std(mode_traces[i].dhf)*(1 + 1/(np.log(k))))
                    dmac_threshold_new_i =  np.sqrt(np.std(mode_traces[i].dhmac)*(1 + 1/(np.log(k))))

                    #Set the new thresholds to the traces
                    mode_traces[i].set_threshold(df_threshold_new = f_threshold_new_i, dmac_threshold_new = dmac_threshold_new_i)
    
    return mode_traces


class ModeTracesPostProcess():
    def __init__(self, sortedTraces, frequenciesHz, phis, true_f, true_phi, algorithm, df_threshold, dmac_threshold):
        """
        A class for post processing of a list of traces

        Arguments
        --------
        sortedTraces: sorted list of ModeTrace instances by frequency
            ModeTrace class defined in this file
        
        frequenciesHz: list of 1D arrays
            AOMA frequencies over all time segments

        phis: list of 2D arrays
            AOMA mode shapes over all time segments

        algorithm: string
            Applied tracking algorithm (Threshold value definition)
            Options: 'magalhaes2008', 'cabboi2016', 'he2022', 'proposed'

        df_threshold: float
            df

        dmac_threshold: float
            1 - MAC
        """
        self.n_skipped_time_seg = np.zeros(true_f.shape[1])
        self.n_false_positives = np.zeros_like(self.n_skipped_time_seg)
        self.n_modes = np.zeros_like(self.n_false_positives)

        self.sortedTraces = sortedTraces
        self.frequenciesHz = frequenciesHz
        self.phis = phis

        self.true_f = true_f
        self.true_phi = true_phi

        self.algorithm = algorithm

        self.df_threshold = df_threshold
        self.dmac_threshold = dmac_threshold
        self.dj_threshold = df_threshold + dmac_threshold

    def calculate_performance_measure(self, n = 0):
        if self.algorithm == 'magalhaes2008' or self.algorithm == 'cabboi2016':

            for j, trace in enumerate(self.sortedTraces):
                self.sortedTraces[j].count_missed_time_segments(self.frequenciesHz, self.phis, self.true_f[:,j], self.true_phi[j,:,:], df_max = self.df_threshold, dmac_max = self.dmac_threshold)
                self.n_skipped_time_seg[j] = trace.n_skipped_time_seg
                self.n_false_positives[j] = trace.n_false_positives
                self.n_modes[j] = len(trace.time_seg)

        elif self.algorithm == 'he2022':
            for j, trace in enumerate(self.sortedTraces):
                self.sortedTraces[j].count_missed_time_segments(self.frequenciesHz, self.phis, self.true_f[:,j], self.true_phi[j,:,:], lim = self.df_threshold + self.dmac_threshold)
                self.n_skipped_time_seg[j] = trace.n_skipped_time_seg
                self.n_false_positives[j] = trace.n_false_positives
                self.n_modes[j] = len(trace.time_seg)
        
        elif self.algorithm == 'proposed':
            #print('specify n')
            for j, trace in enumerate(self.sortedTraces):
                self.sortedTraces[j].count_missed_time_segments(self.frequenciesHz[n:],self.phis[n:], self.true_f[n:,j], self.true_phi[j,:,n:], lim = self.df_threshold + self.dmac_threshold, n = n)
                self.n_skipped_time_seg[j] = trace.n_skipped_time_seg
                self.n_false_positives[j] = trace.n_false_positives
                self.n_modes[j] = len(trace.time_seg)


    def latex_format(self, run_index):
        '''
        Write a row of latex formatted info to be placed in table

        Arguments
        --------
        run_index : int
        
        '''
        mode0mo = self.n_modes[0]
        mode1mo = self.n_modes[1]

        mode0f = self.n_false_positives[0]
        mode1f = self.n_false_positives[1]

        mode0m = self.n_skipped_time_seg[0]
        mode1m = self.n_skipped_time_seg[1]

        aoma0 = mode0mo - mode0f + mode0m
        aoma1 = mode1mo - mode1f + mode1m
        
        perc_hit0 = (mode0mo - mode0f)/aoma0*100
        perc_false0 = mode0f/mode0mo*100
        perc_hit1 = (mode1mo - mode1f)/aoma1*100
        perc_false1 = mode1f/mode1mo*100

        print(str(int(run_index))+' & '+ str(int(aoma0)) +' & \missed{'+  str(int(mode0m)) +
              '} & \mfalse{'+ str(int(mode0f))+'} & '+ str(int(mode0mo))+ ' & \percent{' + str(int(perc_hit0))+ 
                '}\% & \mbadpercent{'+str(int(perc_false0))+'}\% & '+ str(int(aoma1))+' & \missed{'+ str(int(mode1m)) +
                    '} & \mfalse{'+ str(int(mode1f))+'} & '+ str(int(mode1mo))+ ' & \percent{'+ str(int(perc_hit1))+ 
                        '}\% & \mbadpercent{'+str(int(perc_false1))+"}\% \\"+ "\\")
        
        return np.array([aoma0, mode0m, mode0f, mode0mo, perc_hit0, perc_false0, aoma1, mode1m, mode1f, mode1mo, perc_hit1, perc_false1])


    