import numpy as np 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(1, "../")
from _shear_frame import ShearFrame_3D
from utils_OB import *
import koma.oma, koma.plot
import random
from scipy.signal import detrend, welch, resample


def dynamic_stiffness_matrix(kx,ky,shapex,shapey): 
    '''Function for changing the stiffness matrix over time
    Arguments: 
        kx: 1darray: 
            stiffness for each floor in x-direction, first entry 
            correspons to the lower floor. Length n correspons to 
            the numer of floors i system.
        ky: 1darray
            Same as kx, but i y-direction
        shapex: 1darray or ndarray
            array of values to be multiplied with kx. if 1darray
            all floors have the same changes. 
            If ndarray each of the entries correspons to the development
            in each floor. 
        shapey: 1darray or ndarray
            Same shape as shapex

    Return: 
        Kxs: array
            array containing a kx vector for each entry in shape matrix
        Kys: array
            same as for Kxs
    '''
    # generating full shapes if they are 1d
    if len(np.shape(shapex))==1:
        n=len(kx)
        shapex=np.array([shapex]*n)
    if len(np.shape(shapey))==1:
        n=len(ky)
        shapey=np.array([shapey]*n)
     
    kxs=np.zeros_like(shapex)
    kys=np.zeros_like(shapex)
    for i in range(len(shapex[0])):
        kxs[:,i]=shapex[:,i]*kx
        #print(kxs[:,i])
    for i in range(len(shapey[0])):
        kys[:,i]=shapey[:,i]*ky
    return kxs.T,kys.T



def generate_dynamic_series(kx, ky,shapex,shapey,M,t,*args,white_noise_amp=1,
next_segment='new',next_DOF='new',harmonic='decreasing',harmonic_part=1,
plot=False): 
    '''
    Arguments: 
        kx: 1darray: 
            stiffness for each floor in x-direction, first entry 
            correspons to the lower floor. Length n correspons to 
            the numer of floors i system.
        ky: 1darray
            Same as kx, but i y-direction
        shapex: 1darray or ndarray
            array of values to be multiplied with kx. if 1darray
            all floors have the same changes. 
            If ndarray each of the entries correspons to the development
            in each floor. 
        shapey: 1darray or ndarray
            Same shape as shapex
        M: 1darray
            mass for the floors. first entry is the mass of 
            the first floor.

    *args:
        If none of [next_segment,next_DOF,harmonic,harmonic_part] is 
        specified the same white noise loading are used for all
        DOFs, for all the time series.  

        next_segment: 'new' or'same'
                whether or not to creata a new set of loads 
                for each time segment

        next_DOF: 'new','same' or 'random'
                whether or not to create a new set of loading for each
                DOF. 'same' uses the same load series, new creates a new
                series with same parameters, randaom creates random params.
                as well.
        white_noise_amp: float
                the relative amplitude of the white noise 

        harmonic: string or None
            'random' crates a random freq. in the interval [0.5w_0,2w_n] w 
            are natural frqs of the frame.
            'decreasing' creates a decreasing frequenzy from [w_n , w_0]

        harmonic_part: float
            the harmonic amplitude in relation to the white noise. 0.5
            gives 50% of the amplitude. 

            

        plot: bolean
            Wether or not to plot the diagram

    Return:
        true_w: 2darray
            First axis are the modes, first x-dir, then y-dir. 
            Second axis are the frequencies for each of the time series
        true_phi: 3darray
            First axis are the modes, first x-dir, then y-dir.
            Second axis are the mode shape inputs.
            Third axis are the mode shapes of each of the time series
        
        responses: 2d list of 2darrays
            list[i]=[ddu,du,u] response from time series i
            list[i][2]= u for time series i
            each with following shape: 
            u=[ [u1(t1),..., u1(tn)],
                [u2(t1),..., u2(tn)],
                .,
                [um(t1),..., um(tn)]
               ]
            where m = number of DOFS (m/2 number of floors)
                  n = length of each time series

            eg: responses[0][2][3,100]= first time series
                                        3. type (displacement,u)
                                        DOF number 4
                                        at timestep 101
                
        t: 1darray
            the time series used in each step. same for all steps. 
            
    '''
    
    m=len(kx)     # number of DOFs in each direction
    kxs,kys=dynamic_stiffness_matrix(kx,ky,shapex,shapey)
    n=len(kxs) #numer of time series to create
    frame=ShearFrame_3D()
    true_w=np.zeros((n,2*m))
    true_phi = np.zeros((2*m,2*m,n))
    # t=np.linspace(0,3000,10000)
    responses=[]



    for i, kx_i in enumerate(kxs): #iterate the time segments
        ky_i=kys[i]
        frame.set_mass_and_stiffness(kx_i,ky_i,M)
        frame.set_damping()
        true_w[i]=np.hstack((frame.true_wx,frame.true_wy))**0.5
        phix_temp = np.hstack((frame.eigenvecs_x, np.zeros_like(frame.eigenvecs_x)))
        phiy_temp = np.hstack((np.zeros_like(frame.eigenvecs_y), frame.eigenvecs_y))
        true_phi[:,:,i] = np.vstack((phix_temp, phiy_temp))

        if next_segment=='same' and i==0 :
            F=np.zeros((2*m,len(t)))
            if next_DOF == 'same':
                load=load_series(t)
                load.create_load(white_noise_amp,harmonic,harmonic_part,
                np.amax(true_w),np.amin(true_w))
                F=np.array([load.F]*2*m)

            elif next_DOF =='new': 
                for i in range(2*m):
                    load=load_series(t)
                    load.create_load(white_noise_amp,harmonic,harmonic_part,
                    np.amax(true_w),np.amin(true_w))
                    F[i]=load.F

            elif next_DOF=='random':
                for i in range(2*m):    
                    harm=random.choice(['random','decreasing',None])
                    harm_amp=random.randrange(1,50,1)/10
                    wna=random.random()*10
                    load=load_series(t)
                    load.create_load(wna,harm,harm_amp, 
                    np.amax(true_w),np.amin(true_w))
                    F[i]=load.F

        if next_segment=='new':
            F=np.zeros((2*m,len(t)))
            if next_DOF == 'same':
                load=load_series(t)
                load.create_load(white_noise_amp,harmonic,harmonic_part,
                np.amax(true_w),np.amin(true_w))
                F=np.array([load.F]*2*m)

            if next_DOF =='new':
                for i in range(2*m):
                    load=load_series(t)

                    load.create_load(white_noise_amp,
                    harmonic,harmonic_part,
                    np.amax(true_w),np.amin(true_w))

                    F[i]=load.F
            elif next_DOF=='random':
                for i in range(2*m):    
                    harm=random.choice(['random','decreasing',None])
                    harm_amp=random.randrange(1,50,1)/10
                    wna=random.random()*10
                    load=load_series(t)
                    load.create_load(wna,harm,harm_amp, 
                    np.amax(true_w),np.amin(true_w))
                    F[i]=load.F

        response_i=frame.simulate_response(t,F)
        responses.append(response_i)
            

    if plot: 
        t_plot=range(len(true_w))
        for i,ti in enumerate(t_plot):
            plt.scatter(np.ones(len(true_w[i]))*ti,true_w[i],c='black',s=5)
        plt.ylabel(r'$\omega/s$')
        plt.show()
        plt.show(frame.display_system(F,response_i[2],t,display_axis=True))
    return true_w, true_phi,responses


def create_input_COVssi(responses,t,orders):
    '''
    Arguments:  
        responses: 2d list of 2darrays
            list[i]=[ddu,du,u] response from time series i
            list[i][2]= u for time series i
            each with following shape: 
            u=[ [u1(t1),..., u1(tn)],
                [u2(t1),..., u2(tn)],
                .,
                [um(t1),..., um(tn)]
               ]
            where m = number of DOFS (m/2 number of floors)
                  n = length of each time series

            eg: responses[0][2][3,100]= first time series
                                        3. type (displacement,u)
                                        DOF number 4
                                        at timestep 101
                
        t: 1darray
            the time series used in each step. same for all steps. 
    Returns: 
        phis: 2d list of 2d arrays
            eg: phis[0][1][3,4]= 
                            first time series
                            2. modal order
                            4. mode shape 
                            5. DOF 
                            
        lambds: 2d list of 1d arrays
            eg: phis[0][1][3]= first time series
                            2. modal order
                            eigenvalue for mode 4 
                       

    '''
    depth=30
    phis=[]
    lambds=[]
    fs=len(t)/(t[-1]-t[0])
    for time_serie in responses:
        response_i=time_serie[0] #displacement
        lambd,phi=koma.oma.covssi(response_i.T,fs,depth,orders,
        algorithm='standard',showinfo=False)
        lambds.append(lambd)
        phis.append(phi)
    return lambds,phis



def w(lam):
    return np.abs(lam)**0.5


def create_output_guassian(lambds):
    ys=[]
    for series in lambds:
        serie_i=np.array([])
        for order in series:
            freqs=w(order)
            serie_i=np.append(serie_i,freqs)
        ys.append(serie_i)
    return np.array(ys)




def add_noise_response(responses,noise_fac): 
    '''Function that adds noise to each measured series indevidually. 
    The noise are generated as the standard deviation of each indevidullay
    series, and multiplied with the noise_fac, and then added to 
    the measured series with a random factor from a standard 
    normal distribution. 
    Arguments: 
        responses: 2d list of 2darray
            list[i]=[ddu,du,u] response from time series i
            list[i][2]= u for time series i
            each with following shape: 
            u=[ [u1(t1),..., u1(tn)],
                [u2(t1),..., u2(tn)],
                .,
                [um(t1),..., um(tn)]
               ]
            where m = number of DOFS (m/2 number of floors)
                  n = length of each time series

            eg: responses[0][2][3,100]= first time series
                                        3. type (displacement,u)
                                        DOF number 4
                                        at timestep 101 
        noise_fac: float
            Factor to adjust size of added noise. 
    Returns: 
        response_noised: 2d list of 2d arrays
            Same shape as input, with noise. 
            '''
    responses_noised=[]
    for time_series in responses:
        response_noised=[]
        for response in time_series: 
            noise=np.std(response,axis=1)*noise_fac
            noise_stacked=np.vstack([noise]*np.shape(response)[1])
            noised=response + noise_stacked.T*np.random.randn(
                np.shape(response)[0],np.shape(response)[1])
            response_noised.append(noised)
        responses_noised.append(response_noised)
    return responses_noised