
import koma.clustering
import koma.oma
import numpy as np
import matplotlib.pyplot as plt 
from utils_OB import plot_stab_from_KOMA

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def clustering(lambds, phis,orders,s,stabcrit,valid_range,indicator,
               prob_threshold,scaling,min_cluster_size,min_samples,
               *args,true_w=None,plot=False):
    '''
    Arguments: 
        lambds: 2d list of 1d arrays
            eg: phis[0][1][3]= first time series
                            2. modal order
                            eigenvalue for mode 4
        phis: 2d list of 2d arrays
            eg: phis[0][1][3,4]= 
                            first time series
                            2. modal order
                            4. mode shape 
                            5. DOF 
    Returns: 
        all_grouped_phis: list of 2ddarray
            phis for each mode in each time segment
            [1][2,3]: segment 1
                       Mode 2
                       DOF 3
    
        all_grouped_lambds: list og 1darray
            [1][2] first time segment, second mode

                
    '''
    all_grouped_phis=[]
    all_grouped_lambds=[]

    #################
    #Remove:
    phis_auto=[]
    lambds_n_autos=[]
    ###########
    for segment in range(len(lambds)): #iterate the time series
        lambd_i=lambds[segment]
        phi_i=phis[segment]


        #finding stable poles
        lambd_stab, phi_stab, orders_stab, ix_stab =koma.oma.find_stable_poles(
            lambd_i, phi_i, orders, s, 
        stabcrit=stabcrit,valid_range=valid_range,
        indicator=indicator,return_both_conjugates=False)


        #clustering the stable poles 
        pole_clusterer = koma.clustering.PoleClusterer(lambd_stab,
         phi_stab, orders_stab, min_cluster_size=min_cluster_size, 
         min_samples=min_samples,scaling=scaling)


        # sorting the clusters in a new format
        args =pole_clusterer.postprocess(prob_threshold=prob_threshold, 
        normalize_and_maxreal=True)

        # creating new arrays with clustered data
        xi_auto, lambds_n_auto, phi_auto, order_auto, probs_auto, ixs_auto = koma.clustering.group_clusters(*args)

        # REMOVE AGAIN:
        phis_auto.append(phi_auto)
        lambds_n_autos.append(lambds_n_auto)
    return phis_auto,lambds_n_autos
        ####################
    #     omega_n_auto=np.array(lambds_n_auto,dtype=object)**0.5
    #     # merging each cluster to one mode
    #     avg_phis=np.array(
    #         [np.median(np.array(phi),axis=1)for phi in phi_auto])
        
    #     avg_lambds=np.array([np.median(lambd)for lambd in lambds_n_auto])

        
    #     all_grouped_phis.append(avg_phis)
    #     all_grouped_lambds.append(avg_lambds)
        

    # if plot:
        
    #     fig=plot_stab_from_KOMA(sorted_lambds=lambds_n_auto,
    #                                     sorted_orders=order_auto,
    #     lambd_stab=lambd_stab,stab_orders=orders_stab,
    #         lambd=np.array(lambd_i,dtype=object),
    #         all_orders=(orders),
    #         true_w=true_w[segment],figsize=(8,6))
    #     plt.title('Plot of the cluster from the last time segment')
    #     plt.show(block=True)

                
    # return all_grouped_phis,all_grouped_lambds