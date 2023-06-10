import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from scipy import stats
import koma.oma
import strid
import koma.clustering
import pandas as pd
import warnings
from datetime import datetime
# import sys
from utils_OB import*
import gc

import psutil

try:
    import mat73  #install with pip install mat73
except: 
    print('Could not import mat 73. Intall width "pip install mat73" if class "import_file" is to be used.') 


class import_file(): 
    '''Class to import and copy wanted data to a new object and save it for further use. 
    Arguments: 
        folder_path: string
            path to find the folder whre the file import is located.
        file_name: string 
            name of the .mat file to import.'''
    def __init__(self,folder_path,file_name):
        self.folder_path=folder_path
        self.file_name=file_name
        self.path=os.path.join(folder_path,file_name)
        #importing the WHOLE file
        self.data = mat73.loadmat(self.path)

    def get_adjusted_data(self):
        '''
        Function to extract the adjusteted data from an
        imported file. 

        returns: 
            data: dict
                keys are sensor names. each key contains three time series.
                x, y and z accelerations 
        '''
        ad_data={}
        for component in self.data.get('recording').get('sensor'):
            key=component.get('sensor_name')
            if key[0]=='H': #skips the wind
                sens=component.get('component')
                data=sens.get('adjusted_data')
                out=np.array([data[0][0],data[1][0],data[2][0]])
                ad_data[key]=out
        return ad_data

    def get_raw_data(self):
        '''
        Function to extract the raw data from an
        imported file. 

        returns: 
            data: dict
                keys are sensor names. each key contains three time series.
                x, y and z accelerations 
        '''
        rw_data={}
        for component in self.data.get('recording').get('sensor'):
            key=component.get('sensor_name')
            if key[0]=='H': #skips the wind
                sens=component.get('component')
                data=sens.get('raw_data')
                out=np.array([data[0][0],data[1][0],data[2][0]])
                rw_data[key]=out
        return rw_data
    
    def get_component_atr(self,atr): 
        pass

    def get_A6_wind(self):
        ad_data={}
        for component in self.data.get('recording').get('sensor'):
            key=component.get('sensor_name')
            if key[0:2]=='A6': 
                sens=component.get('component')
                quality=sens.get('data_quality')
                data=sens.get('adjusted_data')

                out=np.array([data[0][0],data[1][0],data[2][0],data[3][0]])
                ad_data[key]=out
                ad_data['quality']=quality
        return ad_data


    def convert(self,):
        '''Function to create a new object from class
         'time_segment' and assign wanted values to it.  
         
         returns: object, type 'time_segment'
                This object can be further manipulated with functions 
                from its own class. 
         '''
        self.component_atr_to_transfer=[
        'component_no',
        'data_quality',
        'mean',
        ]
        new_ts=time_segment()
        data_dict={}
        for sensor in self.data.get('recording').get('sensor'):
            sensor_name=sensor.get('sensor_name').replace(' ','_')

            #--------------- transformation-------
            trans_matrix=sensor.get('transformation_matrix')

            
            if sensor_name[0]=='H': #skips the wind
                # transferring the data for each senor
                component=sensor.get('component') 
                adjusted_data=component.get('adjusted_data')
                for i,dir in enumerate(['_x','_y','_z']):
                    comp_name=sensor_name+dir
                    data_dict[comp_name]=np.array(
                        adjusted_data[i][0])
                    
                    new_ts.sensor_metadata[comp_name]={}
                    # transfering sensor metadata
                    for atr in self.component_atr_to_transfer:
                        new_ts.sensor_metadata[
                            comp_name][atr]=component.get(atr)[i][0]
                    new_ts.sensor_metadata[comp_name][
                        'adjusted_samplerate']=sensor.get(
                            'adjusted_samplerate')
        length=len(adjusted_data[0][0])
        # setting the metadata for the whole series: 
        new_ts.series_metadata['file_name']=self.file_name[:-4]
        
        dur=self.data.get(
            'recording').get('general').get('duration')
        new_ts.set_atr('ac_data',data_dict)
        new_ts.series_metadata['duration']=dur
        new_ts.set_atr('timeseries',np.linspace(0,dur,length))

        return new_ts

        


################################################                
        
def import_converted_ts(path,name): 
    '''Function to import a .pkl object of class 'time_segment'
    arguments: 
            path:  string 
                folder path
            name: string 
                filename. '''
    if name[-4:]=='.pkl':
        tot_path=path+name
    else: tot_path=path+name+'.pkl'
    with open(tot_path, "rb") as file:
        return pickle.load(file)


##########################################################

class time_segment():
    '''Class for creating a new object from 
    data extracted with "import_file" class. '''
    def __init__(self):
        self.sensor_metadata={}
        self.series_metadata={}

    def set_general_metadata(self,key,value): 
        self.series_metadata[key]=value
        
    def get_from_general_metadata(self,key):
        if key in self.series_metadata.keys():
            return self.series_metadata.get(key)
        else: return None

    def set_atr(self,attribute_name,sensor_data):
        setattr(self,attribute_name,sensor_data)

    ##===========================
    def do_covssi(self,i,orders,start,stop): 
        '''Function for performing cov-ssi on the loaded data.
        Start and stop are used to slice the timeseries. 
        Arguments: 
            i: int
                Maximum number of block rows
            orders: 1darray
                What modal orders to use
            start: float
                where in the time series to start.
                start index = int(len(t)*start) 
                start = 0 => start at t=0
                start = 1 => start at t=end
            stop: float 
                Where to stop. 
        '''
        data_dict=self.ac_data
        full_length= len(data_dict.get('H3_East_y'))
        start_index= int(start*full_length)
        stop_index=  int(stop*full_length)
        sliced_length= stop_index-start_index
        
        data_arr=np.zeros((len(data_dict),sliced_length))
        for i,key in enumerate(data_dict.keys()):
            series=data_dict.get(key)
            data_arr[i]=series[start_index:stop_index]

        # self.fs=len(self.timeseries)/(
                # self.timeseries[-1]-self.timeseries[0])
        self.fs=20        

        data_arr=np.nan_to_num(data_arr)
        self.lambd,self.phi=koma.oma.covssi(data_arr.T,self.fs
                                            ,i,orders,showinfo=False)

        
    # ============================== 
    def do_covssi_strid(self,j,orders):
        data_dict=self.ac_data
        data_arr=np.zeros((len(data_dict),len(data_dict.get(
            'H3_East_y'))))
        for i,key in enumerate(data_dict.keys()):
            series=data_dict.get(key)
            data_arr[i]=series

        # self.fs=len(self.timeseries)/(
                # self.timeseries[-1]-self.timeseries[0])
        self.fs=20        
        self.orders=orders
        data_arr=np.nan_to_num(data_arr)

        ssid = strid.CovarianceDrivenStochasticSID(data_arr, self.fs)

        self.modes = {}
        phi=[]
        lambds=[]
        for i, order in enumerate(orders):
            A, C, G, R0 = ssid.perform(order, j)
            lr, Q = np.linalg.eig(A)
            u = self.fs*np.log(lr)
            Phi = C.dot(Q)
            phi.append(Phi)
            lambds.append(u)
        self.lambd=lambds
        self.phi=phi
        self.orders=orders

        

    #==============================
    def find_stable_poles(
            self,s,stabcrit,valid_range,indicator,*args,orders=False):
        '''Function to determend wich poles to keep for clustering. 
        13.03.23: 'Orders moved to optional. use orders=orders for older
          uses and older time segment-objects.
         Arguments: 
            s: int
                Number of poles where stabcrit needs to be ok
            stabcrit: dict
                {'freq':0.05, 'damping': 0.2, 'mac': 0.2}
                realitve difference tolerance for poles. 
                small values --> conservative --> fewer poles
            valid_range: dict:
                ex: { 'freq': [0, 62.8],'damping': [-1,np.inf] }
            indicator: string
                'freq' or 'mac'

        '''
        if not orders: 
            lambd=np.array(self.lambd,dtype=object)
            step=int((len(lambd[-1])-len(lambd[0]))/(len(lambd)-1))
            os=np.arange(len(lambd[0]),len(lambd[-1])+step,step)
            self.orders=os


        self.lambd_stab, self.phi_stab, self.orders_stab, self.ix_stab =koma.oma.find_stable_poles(self.lambd, self.phi,self.orders, s, 
        stabcrit=stabcrit,valid_range=valid_range,
        indicator=indicator,return_both_conjugates=False)

    #======================
    def cluster(self,p_ts,mcs,ms,scaling):
        '''Method to cluster the stable poles 
        Arguments: 
        --------------
            p_ts: float
                probability of pole to belong to 
                cluster, based on estimated "probability" density function
            mcs: int
                minimum cluster size
            ms: int
                number of points in neighbourhood for point to be
                considered core point 
                (larger value => more conservative clustering)
            scaling: dict
                scaling={'mac':1, 'lambda_real':1, 'lambda_imag': 1}
        '''

        #ignore old syntax warning in KOMA: 
        warnings.simplefilter('ignore', SyntaxWarning)

        pole_clusterer = koma.clustering.PoleClusterer(self.lambd_stab,
         self.phi_stab, self.orders_stab,
        min_cluster_size=mcs, 
         min_samples=ms,scaling=scaling)



        args =pole_clusterer.postprocess(prob_threshold=p_ts, 
        normalize_and_maxreal=True)

        self.xi_auto, self.lambds_n_auto, self.phi_auto, self.order_auto, probs_auto, ixs_auto = koma.clustering.group_clusters(*args)
        ###### OBS self.lambds_n_auto is actually self.omega_n_auto######
        self.PHI_median=[]
        # self.phi_vars=[]
        # self.phi_medians=[]

        for phi in self.phi_auto:
            # self.PHI_median.append(np.real(np.median(phi,axis=1)))
            self.PHI_median.append((np.median(phi,axis=1)))

        
        self.lambds_median=[]
        # self.lamd_vars=[]
        # self.lambd_means=[]
        for l in self.lambds_n_auto:
            self.lambds_median.append(np.median(l))
            # self.lambd

    #========================================
    def inspect_phis(self,cluster_index, n_orders):
        phis=self.phis[cluster_index,:n_orders]

    # =======================================
    def create_confidence_intervals(self,*args,stat='meadian',confidence=95,
                                    included_clusters=-1):
        ''' Function to calculate the confidence interval of the 
        frequency and the eigenvectors of the clusters by bootstrapping. 
        Arguments: 
        -----------
            stat: string
                'median', 'mean' or 'mode'. The statistic to calculate. 
                Default is median
            confidence: float
                The confidence border of the confidence interval. 
            included_clusters: int
                index of the last cluster to include. -1 for all clusters

        Returns: 
             added attributes to the time_segment object: 
             Same shape as .PHI_median and .lambds_median
            new aattributes: 
                .bs_lambds: 1d array
                    One new estimated lambda for each cluster
                .bs_lambds_confidence: 2darray
                    For each lambd a 2x1 array with limits of the
                    given confidence interval.
                .bs_phi: 2darray
                    One vector for each cluster, where all components in the 
                    vector is estimated by the botstrapping of the chosen 
                    statistic. 
                .bs_phi_confidence: 3darray
                    Same shape as bs.phi on the two first levels, but 
                    each place holdes a 2x1 array with the limits of 
                    the confidence interval. 
        
        '''
        self.bs_lambds=[]
        self.bs_lambds_confidence=[]
        self.bs_phi =[]
        self.bs_phi_confidence=[]
        for cluster_ix in range(len(self.lambds_n_auto)): #iterate the clusters 
            # lambda:
            bs_object=stats.bootstrap((self.lambds_n_auto[cluster_ix],),
                                      np.median,n_resamples=999)


            # self.bs_lambds.append(stats.mode(
            #     bs_object.bootstrap_distribution)[0])
            self.bs_lambds.append(np.mean(
                                bs_object.bootstrap_distribution))


            self.bs_lambds_confidence.append(
                np.array(bs_object.confidence_interval[:]))
            #Phi:

        
    #=================================
    def save(self,path):
        '''save the object as .pkl-file at the dessignated 
        location. The name is picked from the object. 
        Arguments: 
            path: string'''
        tot_path=path+self.series_metadata['file_name']+'.pkl'
        with open(tot_path, 'wb') as fout:
            pickle.dump(self, fout)

    #==================================
    def stabdiag(self,*args,true_w=None,
    show_num=True,xmin=0,xmax=10,info={},
    figsize=(15,8),display=False,color='clusters',**kwargs):
        
        fig=PlotStabDiagFromTsObject(self,true_w=None,
        show_num=True,xmin=0,xmax=10,info={},
        figsize=(15,8),display=display,color='clusters',**kwargs)




#########################################
def macplot(ts,number_of_modes,*args,ref='median'):
    '''Function to inspect modeshapes of the poles in a number of clusters.
     Arguments:
            ts: time_segment object
                must have been clustered
             ref: string, optional
    returns:
            figure
              '''
    # matplotlib.use('tkagg')
    fig=plt.figure(figsize=(10,8))
    print(len(ts.phi_auto[0][8]))
    print(len(ts.phi_auto[0]))
    print(len(ts.order_auto))

    for i in range(0,number_of_modes):
        cluster=ts.phi_auto[i]
        order=ts.order_auto[i]
        print((cluster.shape))
        macs=np.zeros(len(cluster[0]))
        if ref== 'median':
            ref_mode=np.median(cluster,axis=1)
        for j in range(len(cluster[0])):
            phi=cluster[:,j]
            macs[j]=MAC(phi,ref_mode)
        if i !=0:
            axes=plt.subplot2grid((1,number_of_modes),
                              (0,i),rowspan=1,colspan=1,fig=fig,sharey=axes)
            axes.set_axis_off()
        else:
            axes=plt.subplot2grid((1,number_of_modes),
                              (0,i),rowspan=1,colspan=1,fig=fig)
            
        freq=np.median(np.abs(ts.lambds_n_auto[i]))
        axes.axvline(1,linestyle='--',color='black')
        axes.set_title(
            'f=\n'+str(round(freq,3))+'\nHz\nMode\n'+str(i+1),loc='right')
        axes.set_xlim(0.1,1.1)
        # axes.set_xscale('log')

        # 

        norm = matplotlib.colors.LogNorm(vmin=0.2, vmax=1)
        axes.scatter(macs,order,c=macs**10,cmap='brg',norm=norm)


    plt.show()
    return fig

#################################################
def quality_check(ts): 

    '''Function to check the quality of the imported series:
    arguments. checks that the quality stamp is not 'poor'
    and that all the components are where they should be. : 
        ts: object of class 'time_segment'''
    good_quality=True
    for key in ts.sensor_metadata.keys(): 
        string=ts.sensor_metadata.get(key).get('data_quality')
        if 'poor' in string:
            good_quality=False
            return good_quality
        
        test2=ts.sensor_metadata.get(key).get('component_no')
        if not 'x' in key and test2==1:
            good_quality=False
            return good_quality
        if not 'y' in key and test2==2:
            good_quality=False
            return good_quality
        if not 'z' in key and test2==3:
            good_quality=False
            return good_quality
    if not len(ts.sensor_metadata.keys())==60:
        good_quality=False
    return good_quality

###################################################

def convert_folder(path_in,*args,path_out='',sorting_crit='-'):
    '''Function to convert a whole folder of .mat -files to 
    "time_segment" objects and save them as .pkl files
    arguments: 
        path_in: string 
            path of the folder with the .mat files. 
            can contain other folders, but not other file types
        
        *args: 
        path_out: string, optional
            Where to save the .pkl files. Creates a subfolder called 
            'converted_ts' in the input folder by default. 
        sorting_crit: string 
            substring of filename demanded to convert file
            eg. '2015-05' for converting all files from may 2015
            name format: HB141M-YYYY-MM-DD_HH-MM-SS.mat  '''
    if path_out == '':
        path_out=path_in+'/converted_ts/'
    try:
        os.mkdir(path_out)
    except FileExistsError:
        pass
    errors=[]
    dir = os.listdir(path_in)
    dir_out=os.listdir(path_out)
    for entry in dir:
        try:
            file_path=os.path.join(path_in,entry)
            if os.path.isfile(file_path) and sorting_crit in file_path:
                if not entry[:-3]+'pkl' in dir_out:
                    imported=import_file(path_in,entry)
                    new_ts=imported.convert()
                    if quality_check(new_ts):
                        new_ts.save(path_out)

        except:
            errors.append(entry)
            continue
    print('Errors: ',errors)              

#######################################################
def get_ts_folder(path_in): 
    '''Function for importing all .pkl files from a folder and 
    store the objects in a list. '''
    time_series=[]
    dir = os.listdir(path_in)
    for entry in dir:
        file_path=os.path.join(path_in,entry)
        if os.path.isfile(file_path):
            ts=import_converted_ts(path_in,entry)
            time_series.append(ts)
    return time_series

#######################################################

def cluster_folder(path_in,s,orders,stabcrit,
valid_range,indicator,prob_threshold,min_cluster_size,
min_samples,scaling,*args,path_out='',overwrite=False,plot=False,figsize=(8,10)):
    if path_out == '':
        path_out=path_in+'/clustered/'
    try:
        os.mkdir(path_out)
    except FileExistsError:
        if overwrite:
            pass
        else: 
            print('Try new output path, or set overwrite = True')
            return

    dir = os.listdir(path_in)
    for entry in dir:
        file_path=os.path.join(path_in,entry)
        if os.path.isfile(file_path):

            ts=import_converted_ts(path_in,entry)
            
            ts.find_stable_poles(
                    orders,s,stabcrit,valid_range,indicator)
            ts.cluster(prob_threshold,
                     min_cluster_size,min_samples,scaling)
            ts.save(path_out)
            

            
#######################################################

def import_folder_for_tracking(path_in):
    '''Funtion that takes all the "time_segment" objects from 
    a folder and import them, end creates a list of lambds and a 
    list of the corresponding phis.
    
    Arguments: 
        path_in: string
            Path to the dessignated folder
    Returns:
        lambds: list of 1darrays
        phis: list of 2darrays
        names: list of strings with file names
        '''

    lambds=[]
    names=[]
    phis=[]
    dir = os.listdir(path_in)
    dir.sort()
    for entry in dir:
        file_path=os.path.join(path_in,entry)
        if os.path.isfile(file_path):
            ts=import_converted_ts(path_in,entry)
            #print(np.where(np.sqrt(np.array(ts.lambds_median))/(2*np.pi) < 0.500))
            id = np.where(np.sqrt(np.array(ts.lambds_median))/(2*np.pi) < 0.500)[0]
            lambds.append(np.array(ts.lambds_median)[id])
            phis.append(np.array(ts.PHI_median)[id])
            names.append(entry[:-4])
    return lambds,phis,names



#############################################################
def adjust_means(ts): 
    for key in ts.ac_data.keys():
        mean=ts.sensor_metadata.get(key).get('mean')
        ts.ac_data[key]=ts.ac_data.get(key)-mean
    return ts

#######################################################

def plot_psd(simple_ts_object, data_key_list, plot_cutoff_percentage):
    """
    Uses instance of the class "time_segment" after the method "convert()" is used in "import file"
    Method to plot the spectra and cross spectra of the specified keys in the input.

    Arguments
    ------------
    data_key_list: list of strings
        specify sensors

    plot_cutoff_percentage: float
        specify how much of the psd to plot 
    """

    Ndivisions = 10
    Nwindow = np.ceil(len(simple_ts_object.timeseries)/Ndivisions)
    Nfft_pow2 = 2**(np.ceil(np.log2(Nwindow)))

    dt = simple_ts_object.timeseries[1] - simple_ts_object.timeseries[0]

    fig, axs = plt.subplots(len(data_key_list), len(data_key_list), figsize=(15,10), constrained_layout=True)

    if len(data_key_list) == 1:

        f, S_Hz = signal.welch(simple_ts_object.ac_data.get(data_key_list[0]), fs = 1/dt, window = 'hann', nperseg = Nwindow, noverlap=None, nfft = Nfft_pow2, return_onesided=True)
        axs.plot(f, np.real(S_Hz))
        axs.plot(f, np.imag(S_Hz))
        axs.set_ylabel(data_key_list[0])
        axs.set_xlim([0,plot_cutoff_percentage*f[-1]])

    else:
        for i, k1 in enumerate(data_key_list):
            for j, k2 in enumerate(data_key_list):
                f, S_Hz = signal.csd(simple_ts_object.ac_data.get(k1), simple_ts_object.ac_data.get(k2), fs = 1/dt, window = 'hann', nperseg = Nwindow, noverlap=None, nfft = Nfft_pow2, return_onesided=True)
                
                axs[i,j].plot(f, np.real(S_Hz))
                axs[i,j].plot(f, np.imag(S_Hz))
                axs[i,j].set_ylabel(k1+" and "+k2)
                axs[i,j].set_xlim([0,plot_cutoff_percentage*f[-1]])

#################################################

def PlotStabDiagFromTsObject(

    segment,*args,true_w=True,
    show_num=True,xmin=0,xmax=10,info={},
    figsize=(15,8),display=False,color='clusters',**kwargs):
    '''Function to plot stabdiagram from a time_segment object. 
    Arguments: 
            segment: object
        *args: 
            true_w: bool:
                plot prev dixcovered freqs or not 
            show_num: bool
                wheter or not to show the numbering of the clusters in the plot
            xmin: float
                initial lower limit for the xaxis
            xmax: float 
                initial upper limit for the xaxis
            info: dict
                info to plot in text box in plot
            figsize: tuple
                (w,h)
            display: bool 
                wheter or not to show the plot, or just return 
                the figure. 
            color: string
                How to colorcode the poles. 
                'cluster' -> each cluster get one color
                'mac' -> colors show mac to cluster mode
                'damping_abs' -> colorcode damping on a absolute scale
                'damping rel' -> difference from median damping in cluster
        
    '''
    lambd=segment.lambd
    # print((lambd))
    lambd_stab=segment.lambd_stab
    sorted_lambds=segment.lambds_n_auto
    sorted_orders=segment.order_auto
    stab_orders=segment.orders_stab
    phi_auto=segment.phi_auto
    xi_auto=segment.xi_auto
    if color=='mac':
        fig=plot_stab_from_KOMA(sorted_lambds=sorted_lambds,
                            sorted_orders=sorted_orders,lambd_stab=lambd_stab,
                            stab_orders=stab_orders, lambd=lambd,
                            true_w=true_w,show_num=show_num,xmin=xmin,
                            xmax=xmax,info=info,display=display
                            ,figsize=figsize
                            ,phi_auto=phi_auto

                            )
    elif color=='damping_abs':
        fig=plot_stab_from_KOMA(sorted_lambds=sorted_lambds,
                            sorted_orders=sorted_orders,lambd_stab=lambd_stab,
                            stab_orders=stab_orders, lambd=lambd,
                            true_w=true_w,show_num=show_num,xmin=xmin,
                            xmax=xmax,info=info,display=display
                            ,figsize=figsize
                            ,xi_auto_abs=xi_auto
                            )
    elif color=='damping_rel':
        fig=plot_stab_from_KOMA(sorted_lambds=sorted_lambds,
                            sorted_orders=sorted_orders,lambd_stab=lambd_stab,
                            stab_orders=stab_orders, lambd=lambd,
                            true_w=true_w,show_num=show_num,xmin=xmin,
                            xmax=xmax,info=info,display=display
                            ,figsize=figsize
                            ,xi_auto_rel=xi_auto

                            )
    else: 
        fig=plot_stab_from_KOMA(sorted_lambds=sorted_lambds,
                            sorted_orders=sorted_orders,lambd_stab=lambd_stab,
                            stab_orders=stab_orders, lambd=lambd,
                            true_w=true_w,show_num=show_num,xmin=xmin,
                            xmax=xmax,info=info,display=display
                            ,figsize=figsize

                            )
    

    return fig

################################## 
def ts_via_cov_ssi_df(path,orders,depth):
    ac_channels=['H1 East_x', 'H1 East_y', 'H1 East_z', 'H1 Vest_x',
     'H1 Vest_y',
       'H1 Vest_z', 'H2 Vest_x', 'H2 Vest_y', 'H2 Vest_z', 'H3 East_x',
       'H3 East_y', 'H3 East_z', 'H3 Vest_x', 'H3 Vest_y', 'H3 Vest_z',
       'H4 East_x', 'H4 East_y', 'H4 East_z', 'H4 Vest_x', 'H4 Vest_y',
       'H4 Vest_z', 'H5 East_x', 'H5 East_y', 'H5 East_z', 'H5 Vest_x',
       'H5 Vest_y', 'H5 Vest_z', 'H6 East_x', 'H6 East_y', 'H6 East_z',
       'H6 Vest_x', 'H6 Vest_y', 'H6 Vest_z', 'H7 East_x', 'H7 East_y',
       'H7 East_z', 'H7 Vest_x', 'H7 Vest_y', 'H7 Vest_z', 'H8 East_x',
       'H8 East_y', 'H8 East_z', 'H9 East_x', 'H9 East_y', 'H9 East_z',
       'H9 Vest_x', 'H9 Vest_y', 'H9 Vest_z', 'H10 East_x', 'H10 East_y',
       'H10 East_z', 'H10 Vest_x', 'H10 Vest_y', 'H10 Vest_z', 'H11 East_x',
       'H11 East_y', 'H11 East_z', 'H11 Vest_x', 'H11 Vest_y', 'H11 Vest_z',]
    

     # log='\n\n ######################  New test ######################\n'
    name=os.path.split(path)[-1]
    log='\n\n ######################  New test ######################\n'
    name=os.path.split(path)[-1]


    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open('start_log.txt','a') as log_file:
       log_file.write(dt_string+': '+str(name)+' initiated\n')
       log_file.close()

    try: 
        df=pd.read_pickle(path)
        # print(df.head())
        data_arr=df[ac_channels].to_numpy(copy=True).T 
        data_arr=np.nan_to_num(data_arr,nan=0.0)
    except: 
        raise ValueError('Funka ikke det her ')
        return ''
    

    ts=df['timeseries'].to_numpy()
    fs=len(ts)/(ts[-1]-ts[0])    


    log+='\n'+str(name)+':\n'
    log+='Data_arr: '+str(np.shape(data_arr))
    log+='\nTimeseries: '+str(np.shape(ts))
    if np.isfinite(data_arr).all():
        log+='\nOnly finite values found'
    else: 
        log+='\nInfinite values found'


    # total_memory=psutil.virtual_memory()[0]
    # used_memory=psutil.virtual_memory()[3]
    # free_memory=psutil.virtual_memory()[4]

    # log+='\nMemory status: \n'
    # log+='   Used: '+str(round(used_memory/10**9,3))+'Gb'
    # log+='\n   Free:: '+str(round(free_memory/10**9,3))+'Gb'
    # log+='\n   Percentage: '+str(round((
    #     used_memory/total_memory) * 100, 2))'

    ssid = strid.CovarianceDrivenStochasticSID(data_arr,fs)
    # print('Ssid object created')
    # log+='\n\nssid created and preformed for orders: \n'

    
    modes = {}
    phi=[]
    lambds=[]
    for i, order in enumerate(orders):

        A, C, G, R0 = ssid.perform(order, depth)
        log=str(order)+' '

        if not np.isfinite(A).all():
            A,C,G,R0=(0,0,0,0)
            ssid._svd_block_toeplitz.cache_clear()
            ssid=None
        
            with open('log_file.txt','a') as log_file:
                log_file.write(log)
                log_file.close()
          
            raise ValueError('Infinite values found in A')
        
        lr, Q = np.linalg.eig(A)
        u = fs*np.log(lr)
        Phi = C.dot(Q)
        phi.append(Phi)
        lambds.append(u) 
    ssid._svd_block_toeplitz.cache_clear()           
    del(ssid)
    gc.collect()
    lambd=lambds
    phi=phi

    # name=os.path.split(path)[-1]
    ts=time_segment()
    ts.lambd=lambd
    ts.phi=phi
    name=os.path.split(path)[-1]
    ts.series_metadata['file_name'] = name[:-4]
    
    log+='\n Done \n\n\n'
    with open('log_file.txt','a') as log_file:
        log_file.write(log)
        log_file.close()
    return ts

#####################################################################

def PlotPolesFromSegments(segments,*args,sort='date',max_freq=0.5,bs=False):
    '''Arguments: 
    --------------
        segments: list 
            list of time segments objects, sorted?''' 
            
    matplotlib.use('tkagg')

    fig,axes = plt.subplots(1,1)

    for i,seg in enumerate(segments):
        ix=np.where(np.array(seg.lambds_median)< max_freq*2*np.pi)
        om =axes.scatter (np.ones(len(np.array(seg.lambds_median)[ix]))*i,
                      np.abs(np.array(seg.lambds_median)[
                      ix])/2/np.pi,s=4,color='b',label='Observed medians')
        # axes.plot(seg.bs_lambds_confidence[:],np.hstack()
        
        
    if bs: #eldknsldkn
        for i,seg in enumerate(segments):
            ix=np.where(np.abs(np.array(seg.bs_lambds))< max_freq*2*np.pi)
            bsm =axes.scatter (np.ones(len(np.array(seg.bs_lambds)[ix]))*i,
                        np.abs(np.array(seg.bs_lambds)[ix])/2/np.pi,
                        s=4,color='r',
                        label='Bootstrapped medians')
            # print(seg.series_metadata['file_name'])
        
        for i,seg in enumerate(segments):
            for j in range(0,len(seg.bs_lambds)):
                if np.abs(seg.lambds_median[j]) > max_freq*2*np.pi:
                    continue
                else: 
                    just=0
                    x=[i,i]
                    
                    y=seg.bs_lambds_confidence[j]
                    conf_length=(y[1]-y[0])/2/np.pi
                    if conf_length < 0.01:
                        y_plot=[y[0]/2/np.pi-just*conf_length,
                                y[1]/2/np.pi+just*conf_length]
                        ci,=axes.plot(
                            x,y_plot,linewidth=.5,color='black',zorder=-1,
                                label='95% confidence interval') 
                    else: 
                        y_plot=[y[0]/2/np.pi-conf_length,
                                y[0]/2/np.pi+conf_length]
                        ci2, =axes.plot(
                            x,y_plot,linewidth=.5,color='grey',zorder=-1,
                                label='Large 95% confidence intervals') 


    true_w=np.array([0.052,0.105,0.119,0.142,0.183,0.206,0.212,
                                0.23,0.276,0.318,0.333,0.374,0.401,0.418])
    for w in true_w:
        a=axes.axhline(y=w,linewidth=1,linestyle='--',label='Prev. discovered freqs')
    if bs:    
        axes.legend(handles=[om,bsm,ci,ci2,a],loc='lower right')
    else: 
        axes.legend(handles=[om,a],loc='lower right')

    axes.set_ylim(0,max_freq)
    plt.show()
####################################################################
def PlotSegmentsFromFolder(folder_path,*args,max_freq=0.5,sort='date',bs=False):
    '''Plotting all time segments from one folder of time seegments
    
    Arguments: 
        folder_path: string 
            path of the folder width the segments
        max_freq: float, optional
            Highest frequenzy in Hz to include
        sort: string, optional
            how to sort the segments
            '''
    files=os.listdir(folder_path)
    segments=[]
    for name in files:
        segments.append(import_converted_ts(folder_path,name))
    PlotPolesFromSegments(segments,sort=sort,max_freq=max_freq,bs=bs)
    
#######################################################################
