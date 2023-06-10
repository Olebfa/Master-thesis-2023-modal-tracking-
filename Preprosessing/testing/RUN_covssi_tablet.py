# import pandas as pd 
import numpy as np 
# import matplotlib.pyplot as plt 
import os 
import sys 
import time
path='../src/'
if path not in sys.path: 
    sys.path.insert(1,path)

path='../../src/'
if path not in sys.path: 
    sys.path.insert(1,path)
    
path='C:/Users/olebfa\OneDrive - NTNU/00Master - Onedrive/OB ymse/modal-tracking-thesis-spring-2023/src/'
if path not in sys.path: 
    sys.path.insert(1,path)


# from utils_OB import *
from preposessing import *

import traceback
import psutil 
import gc
from concurrent.futures import ProcessPoolExecutor
# import math

with open('log_file.txt','w') as file:
   file.write('')
   file.close()

with open('start_log.txt','w') as file:
   file.write('')
   file.close()

folder_path='../../../5hz_resampled_int_tans/'

names=os.listdir(folder_path)

names.sort()

# test_names=names[0:20]

t0=time.time()
def add_name(name):
    with open('test_file.txt','a') as file:
        file.write(str(name)+'\n')
    file.close()
    # time.sleep(0.1)
    return name
    

    
def do_cov_ssi_multi(file_p):
    try:
        if file_p not in os.listdir('../../../5hz_resampled_int_trans_covssi/'):    
            path=folder_path+file_p    
            orders=np.arange(50,200,2)
            depth=120
            ts=ts_via_cov_ssi_df(path,orders,depth)
            ts.save('../../../5hz_resampled_int_trans_covssi/')
            gc.collect()

    except Exception as a:
        print('Failed '+file_p)

        crash_log='\n\n###############################################\nCrash log for '+file_p+': \n'
        crash_log+=(str(file_p)+' feil: '+str(a)+'\n\n')
        
        total_memory=psutil.virtual_memory()[0]
        used_memory=psutil.virtual_memory()[3]
        free_memory=psutil.virtual_memory()[4]
    
        crash_log+=('\nMemory status: \n')
        crash_log+=('   Used: '+str(round(used_memory/10**9,3))+'Gb')
        crash_log+=('\n   Free:: '+str(round(free_memory/10**9,3))+'Gb')
        crash_log+=('\n   Percentage: '+str(round((
            used_memory/total_memory) * 100, 2)))
        with open('log_file.txt','a') as file:
           file.write(crash_log+'\n\n')
           traceback.print_exc(file=file)
        file.close()
        gc.collect()
    return '1'


if __name__ =='__main__':
    executor =ProcessPoolExecutor(max_workers=1)
    for result in executor.map(do_cov_ssi_multi, names,chunksize=1):
    	re=result
     
# def main():
#     with ProcessPoolExecutor(max_workers=1) as executor: #max_workers= max
#         futures=executor.map(do_cov_ssi_multi, names,chunksize=1)
#         print('futures are created')
#         for i,future in enumerate(futures):
#             executor.submit(do_cov_ssi_multi,future)

# if __name__ == '__main__':
#     main()
