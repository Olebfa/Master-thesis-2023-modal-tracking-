import sys 
path='../src'
if path not in sys.path:
    sys.path.insert(1,path)
from preposessing import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor

input_folder='../../../5hz_resampled_int_trans_covssi/'
output_folder='../../../5hz_resampled_int_trans_clustered_3/'

run_names=[]
for name in os.listdir(input_folder):
    if name not in os.listdir(output_folder):
        run_names.append(name)


def cluster(name): 
    
    input_folder='../../../5hz_resampled_int_trans_covssi/'
    output_folder='../../../5hz_resampled_int_trans_clustered_3/'
    valid_range= { 'freq': [0.025*2*np.pi, 6.28],'damping': [-0.02,np.inf] }

    indicator='freq'
    s=5
    stabcrit = {'freq':0.6, 'damping': 0.8, 'mac': 0.8}
    prob_threshold = 0.95   #probability of pole to belong to 
    # cluster, based on estimated "probability" density function

    min_cluster_size=40
    min_samples=30
    scaling={'mac':1, 'lambda_real':1, 'lambda_imag': 1}
    
    ts=import_converted_ts(input_folder,name)
    ts.find_stable_poles(
            s,stabcrit,valid_range,indicator)
    ts.cluster(prob_threshold,min_cluster_size,min_samples,scaling)
    # ts.create_confidence_intervals()
    ts.save(output_folder)
    
if __name__ =='__main__':
    executor =ProcessPoolExecutor(max_workers=15)
    for result in executor.map(cluster, run_names,chunksize=1):
        re=result

# ###################################
# _3: 
    # indicator='freq'
    # s=5
    # stabcrit = {'freq':0.6, 'damping': 0.8, 'mac': 0.8}
    # prob_threshold = 0.95   #probability of pole to belong to 
    # # cluster, based on estimated "probability" density function

    # min_cluster_size=40
    # min_samples=30
    # scaling={'mac':1, 'lambda_real':1, 'lambda_imag': 1}
    
    
    ########################################
#_2:
    # indicator='freq'
    # s=5
    # stabcrit = {'freq':0.6, 'damping': 0.8, 'mac': 0.8}
    # prob_threshold = 0.5   #probability of pole to belong to 
    # # cluster, based on estimated "probability" density function

    # min_cluster_size=35
    # min_samples=35
    # scaling={'mac':1, 'lambda_real':1, 'lambda_imag': 1}