import numpy as np
import scipy.linalg as sp
import scipy.signal
import matplotlib.pyplot as plt
import random
import matplotlib
import os

import matplotlib.style as mplstyle
mplstyle.use('fast')


def MAC(v1,v2): 
    """Finding the MAC-number of vector v1 and v2
    input: 
        v1: 1D array, 
            complex, mode shape 1 
        v2: 1D array, 
            complex, mode shape 1. equal length as v1
    Output: 
        MAC-number: float
                    type: real"""
    teller=np.abs(v1.conj().T@v2)**2
    nevner=(v1.conj().T@v1)*(v2.conj().T@v2)
    return np.real(teller/nevner)


def rel_diff(a1,a2): 
    """Relative distance between a1 and a2
    input: 
        a1: 1D array, 
        a2: 1D array, 
    Output: 
        Relative difference: float
    """
    return np.abs(a1-a2)/np.max(np.abs([a1,a2])) 



class load_series(): 
    def __init__(self,t): 
        '''Function for obtaining a response for a given load- and time series
        
        Arguments: 
            t: 1darray
                Linear array contaning the time steps corresponding to the loads
                t :  [ t1,  t2,  ...   , tn ]'''
        self.F=np.zeros_like(t)
        self.t=t
    
    def add_white_noise(self,amplitude):
        '''Add white noise to the load
        Arguments: 
            Amplitude: float
                avrage amplitude of the noise'''
        self.F +=np.random.normal(size=(np.shape(self.F)))* amplitude


    def create_load (self,white_noise_amp,harmonic,harmonic_part,wn,w0):
        '''
        Arguments: 
            white_noise_amp: float
                the relative amplitude of the white noise 

            harmonic: string or None
                'random' crates a random freq. in the interval 
                [w_0 , w_n] w 
                are natural frqs of the frame.
                'decreasing' creates a decreasing frequenzy from
                 [w_n , w_0]

            harmonic_part: float
                the harmonic amplitude in relation to the white noise. 0.5
                gives 50% of the amplitude. 
            
            wn: float
                natural frequency of the most rapid ocselating mode
            w0: float 
                natural frequency of the slowest ocselating mode
                '''
        if harmonic=='decreasing':
            self.F += scipy.signal.chirp(self.t, wn/2/np.pi, self.t[-1]
            , w0/2/np.pi,          
            method='linear')*harmonic_part*white_noise_amp
            self.add_white_noise(white_noise_amp)

        elif harmonic == 'random': 
            freq=random.choice(np.linspace(w0,wn,100))/2/np.pi
            self.F += scipy.signal.chirp(self.t, freq, self.t[-1],    
             freq,method='linear')*harmonic_part*white_noise_amp      
            self.add_white_noise(white_noise_amp)
        else:
            self.add_white_noise(white_noise_amp)




def KK(kx,ky): 
    ''' Arguments: 
            kx: 1darray
                total stiffness for each floor i x-direction. 
                First entry is the lower column 
            ky: 1darray
                total stiffness of each floor in y-direction. 
                Same length as kx
            m: 1darray
                The masses of the floors. one entry for each floor. 
                First entry is the mass of the first floor. 
        Returns:
            KK: 2darray
                stiffness matrix for the full system. size=2*len(kx)
                | [Kx],[0]  |
                |  [0],[Ky] |
    '''
    n=len(kx)
    KKx=np.zeros((n,n))
    KKy=np.zeros((n,n))
    for i in range(len(kx)):
        if i==0:    # bottom floor
            KKx[i,i]+=kx[i]+kx[i+1]
            KKx[i,i+1]-=kx[i+1]
            KKx[i+1,i]-=kx[i+1]
            KKy[i,i]+=ky[i]+ky[i+1]
            KKy[i,i+1]-=ky[i+1]
            KKy[i+1,i]-=ky[i+1]

        elif i==len(kx)-1:  # top floor
            KKx[i,i]+=kx[i]
            KKy[i,i]+=ky[i]

        else:       #other floors
            KKx[i,i]+=kx[i]+kx[i+1]
            KKx[i,i+1]-=kx[i+1]
            KKx[i+1,i]-=kx[i+1]
            KKy[i,i]+=ky[i]+ky[i+1]
            KKy[i,i+1]-=ky[i+1]
            KKy[i+1,i]-=ky[i+1]
    return np.block([[KKx, np.zeros_like(KKx)],
                    [np.zeros_like(KKy), KKy]])



def plot_stab_from_KOMA(*args,sorted_lambds=False,sorted_orders=np.zeros(1),lambd_stab=False,stab_orders=np.zeros(1),lambd=False,all_orders=np.zeros(1),
                        true_w=True,show_num=True,xmin=0,xmax=0.5,info={},display=True,
**kwargs):
    """Returns a figure with the clusters color coded  
    and with lines in between the poles in one cluster
    input: 
    Arguments: 
        sorted_lambds: 2darray
            first axis correspons to each order, second is freqs
            in that order
        sorted_orders: 1d array
            orders tht corresponds to sorted_lambds
        *args: 
            lambd: 2darray
                same type as sorted_lambds. Eigenvalues or 
                frequencies 
            all_orders: 1d array
                orders that correspons to lambd.
            true_w: bool
                True or False
            figsize: (int,int)
            plot_negatives: bool
    
    """
    # import time
    # t0=time.time()
    # print(time.time()-t0)
    #warning for missing data:
    # if not( len(sorted_orders)>1 or len(stab_orders)>1 or
    #         len(all_orders)>1): 
    #     print('No data specified')
    #     return None
    
    # decides if plot is to be saved or displayed interactivly:
    if display:
        matplotlib.use('tkagg')  
    else:
        matplotlib.use('module://matplotlib_inline.backend_inline')

    #create figure and plot scatter

    if 'figsize' in kwargs.keys():
        fig_s=kwargs['figsize']
    else:
        fig_s=(9,9)
    #checks if color should be according to macs
    mac_colors=False
    if 'phi_auto' in kwargs.keys():
        phi_auto=kwargs['phi_auto']
        mac_colors=True

    damping_abs_colors=False
    if 'xi_auto_abs' in kwargs.keys():
        xi_auto=kwargs['xi_auto_abs']
        damping_abs_colors=True

    damping_rel_colors=False
    if 'xi_auto_rel' in kwargs.keys():
        xi_auto=kwargs['xi_auto_rel']
        damping_rel_colors=True

                #creating figure
    figur_h=plt.figure(figsize=fig_s,dpi=100)
    axes=figur_h.add_subplot(1,1,1)
    s_dot=fig_s[1]*2

    #The analythical freqs: 

    if true_w:
        
        true_w=np.array([0.052,0.105,0.119,0.142,0.183,0.206,0.212,
                                0.276,0.318,0.331,0.374,0.401])
        
        
        true_cable_modes=np.array([0.23,0.418])
        lab=0
        for w in true_w:
            if lab==0:
                axes.axvline(w,linestyle='--',
                             label='Known freqs',zorder=-1)
            else: 
                axes.axvline(w,linestyle='--',zorder=-1)

            lab=1
        lab=0
        for w in true_cable_modes:
            if lab==0:
                axes.axvline(w,linestyle='--',color='red',alpha=0.7,
                             label='Known freqs with cable',zorder=-1)
            else: 
                axes.axvline(w,linestyle='--',color='red',alpha=0.7,zorder=-1)

            lab=1
    # plotting all the discarded poles: 

    # print('start ',time.time()-t0)
    if lambd != None:
        lambd=np.array(lambd,dtype=object)
        step=int((len(lambd[-1])-len(lambd[0]))/(len(lambd)-1))
        orders=np.arange(len(lambd[0]),len(lambd[-1])+step,step)
        os=np.repeat(orders,orders)
        col='grey'
        axes.scatter(np.hstack(np.abs(lambd))/(2*np.pi),
                     os,color=col,marker='x',s=s_dot*0.5,alpha=0.5)
    # print('lambd ferdig ',time.time()-t0)

    #plotting the poles that were deemed stable: 
    if len(stab_orders)>1:
        col='green'
        axes.scatter((np.abs(lambd_stab))/(2*np.pi),
        stab_orders,color=col,
                marker='.',s=s_dot*1.3)
    # print('stab ferdig ',time.time()-t0)

    #plotting the poles to keep, and the lines between them
    min=1000
    count=0
    if len(sorted_orders)>1:
        for i in sorted_orders:
            for j in i:
                if j<min:
                    min=j
        # Make an arrays of colors. 

        if mac_colors:

            hsv = matplotlib.cm.get_cmap('hsv', 1024)
            newcolors = hsv(np.linspace(-0.1, 0.21, 20))
            green = np.array([0.1, 0.8, .1, 1])
            newcolors[-1:, :] = green
            newcmp = matplotlib.colors.ListedColormap(newcolors)

        if damping_abs_colors or damping_rel_colors: 
            hsv = matplotlib.cm.get_cmap('hsv', 1024)
            newcolors = hsv(np.hstack((
                np.linspace(0, 0.15, 100),np.linspace(0.25,5,100))))
            black = np.array([0, 0, 0, 1])
            i=111
            purp=newcolors[i]
            newcolors[i:,:]=purp
            newcolors[90:100, :] = black

            newcmp = matplotlib.colors.ListedColormap(newcolors)


        else: 
            count=0
            colors=[]
            num=len(sorted_lambds)+8
            root=int(num**(1/3))
            root_rest=int(num/root**2)+1
            colors=[]
            for r in range(root):
                for g in range(root):
                    for b in range(root_rest): 
                        colors.append([r/root,g/root,b/root_rest])
            random.shuffle(colors)

        #plotting: 

        just=4
        for i,order_i in enumerate(sorted_orders):
            freq_i=sorted_lambds[i]
            
            size=s_dot*1.2
            if mac_colors:
                cluster=phi_auto[i]
                ref_mode=np.median(cluster,axis=1)
                macs=np.zeros(len(cluster[0]))
                for j in range(len(cluster[0])):
                    phi=cluster[:,j]
                    macs[j]=MAC(phi,ref_mode)
                
                order_arr=np.ones_like(freq_i)*order_i

                axes.plot(np.abs(freq_i)/(2*np.pi),order_arr,
                color='black',linewidth=size/25,zorder=-1)

                scat=axes.scatter(np.abs(freq_i)/(2*np.pi),order_i,marker='s',
                c=macs,cmap=newcmp,s=size,vmin=0,vmax=1)
                if i==0:
                    figur_h.colorbar(scat,ax=axes,location='top',
                                    shrink=0.5,pad=0,aspect=40
                                    ,label='MAC relative to each '+ 
                                    'clusters median mode shape')

            elif damping_abs_colors:
                cluster=xi_auto[i]
                # ref_mode=np.median(cluster)
                # macs=np.zeros(len(cluster[0]))
                # for j in range(len(cluster[0])):
                #     phi=cluster[:,j]
                #     macs[j]=MAC(phi,ref_mode)
                
                order_arr=np.ones_like(freq_i)*order_i

                axes.plot(np.abs(freq_i)/(2*np.pi),order_arr,
                color='black',linewidth=size/25,zorder=-1)

                scat=axes.scatter(np.abs(freq_i)/(2*np.pi),order_i,marker='s',
                c=cluster,cmap=newcmp,s=size,vmin=-1,vmax=1)
                
                #----------------
                #annotation of absolute damping
                for k,freq in enumerate(freq_i):
                    xi=round(cluster[k],3)
                    if xi<0.01:
                        xi_str=str(xi)
                        axes.annotate(xi_str,(np.abs(freq)/(2*np.pi),order_arr[k]),textcoords=('offset pixels'),xytext=(6,-3),fontsize=size/3)
                        



                
                #--------------

                if i==0:
                    figur_h.colorbar(scat,ax=axes,location='top',
                                    shrink=0.5,pad=0,aspect=40
                                    ,label='Absolute damping value in'+ 
                                    'pole')

            else: 
                col=colors[i]
                order_arr=np.ones_like(freq_i)*order_i
                axes.scatter(np.abs(freq_i)/(2*np.pi),order_i,marker='s',
                color=col,s=size)
                axes.plot(np.abs(freq_i)/(2*np.pi),order_arr,
                color=col,linewidth=size/25)

            # creating numberboxes to number the clusters
            if show_num:
                count+=1
                props = dict(boxstyle='round', facecolor='wheat',
                 alpha=0.8)
                try:
                    if np.abs(freq_i[0]-sorted_lambds[i-1][0])<0.05:
                        just+=3.5
                    else:
                        just=4
                except IndexError:
                    pass

                axes.plot([freq_i[0]/(2*np.pi),
                freq_i[0]/(2*np.pi)],
                [order_i[0],min-just]
                ,color='grey',linestyle='--',
                linewidth=size/80)
                axes.text(freq_i[0]/(2*np.pi),
                min-just,str(count),fontsize=8,bbox=props)
            else: 
                count=i

    # print('cluster ferdig ',time.time()-t0)
    #adding the infobox if there are any info to display 
    keys=info.keys()
    if len(keys)>0:
        info['\n'+r"$\bf{Total\ number\ of\ clusters: }$"]=count
        string=r"$\bf{Parameters\ for\ stable\ poles\ and\ clustering:}$"
        for key in keys: 
            string+='\n'+str(key)+': '+str(info.get(key))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes.text(0.7, 0.01, string,
                        transform=axes.transAxes,
                         fontsize=9,ha='left',
                        verticalalignment='bottom', bbox=props,
                        clip_on=True)
        axes.legend(loc=(0.5,0.01))
    else:
        axes.legend(loc='upper left')
        pass
    
    #fixing the limits
    axes.set_xlim(xmin,xmax)
    axes.set_ylabel('Modal order')
    axes.set_xlabel('Frequencies [Hz]')
    # plt.tight_layout() 
    if display:
        plt.subplots_adjust(left=0.1, bottom=0.1,
                right=0.95, top=0.95, wspace=0, hspace=0)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    # print('ferdig ',time.time()-t0)
    return figur_h
 
########################################
def transform(modes_dict):
    '''Funtion that converts a dictionary of STRIDs mode 
    objects to lambd, phi and orders list, like used in KOMA'''
    lambd=[]
    phi=[]
    orders=list((modes_dict.keys()))

    for key in modes_dict.keys():
        num_poles=len(modes_dict[key])
        len_eig=len(modes_dict[key][0].v)
        lambd_i=np.empty(num_poles,complex)
        phi_i=np.empty((num_poles,len_eig),complex)

        for i,pole in enumerate(modes_dict[key]):
            phi_i[i]=pole.v
            lambd_i[i]=-pole.w*pole.xi+pole.wd*1j  #eigenvalue
        lambd.append(lambd_i)
        phi.append(phi_i.T)

    return lambd,phi,orders
###################################################
def remove_neg(modes):  
      #removes all modes where  the imaginary part of the 
      #eigenvalue is close to zero, or negative. 
      # correspons to remove modes with damped periode above 20 years.
      # must be done to be able to calculate xi= eig/ abs(eig) not equal to 1. 
    for key in modes: 
        temp=[]
        for i,m in enumerate(modes[key]):
            if np.sqrt(1-m.xi**2) !=0 and m.wd >0:
                temp.append(m)
        if len(temp)>0:
            modes[key]=temp
        else:   # no point in keeping the modal order if we dond use any of the poles
            modes.pop(key)
    return modes

