import numpy as np
import scipy.linalg as sp
import scipy.signal
import matplotlib.pyplot as plt
from utils_OB import KK




class ShearFrame_3D():
    def __init__(self):
        '''Class for constructing a 3D shear frame whith indevidual stiffness for 
        each floor, in both directions. Each floor have two DOFs located 
        perpendicular to each other in the slab center. '''


        pass

    def set_mass_and_stiffness(self,kx,ky,m):
        '''Arguments: 
        kx: 1darray
            total stiffness for each floor i x-direction. 
            First entry is the lower column 
        ky: 1darray
            total stiffness of each floor in y-direction. 
            Same length as kx
        m: 1darray
            The masses of the floors. one entry for each floor. 
            First entry is the mass of the first floor. '''

        self.kx=kx
        self.ky=ky
        self.m=m
    
        self.KK=KK(self.kx,self.ky)
        n=len(self.kx)
        self.KKx=self.KK[:n,:n]
        self.KKy=self.KK[n:,n:]

        MM=np.zeros((len(self.m),len(self.m)))
        for i in range(len(self.m)): 
            MM[i,i]=self.m[i]
        self.MMxy=MM
        self.MM=np.block([[MM, np.zeros_like(MM)],
                          [np.zeros_like(MM), MM]])
        
        #solves eigenvalue problems
        lambd_x,phi_x =sp.eig(self.KKx,self.MMxy)   
        lambd_y,phi_y =sp.eig(self.KKy,self.MMxy)   
        # self.true_wx=np.sqrt(np.real(lambd_x[np.argsort(lambd_x)]))
        # self.true_wy=np.sqrt(np.real(lambd_y[np.argsort(lambd_y)]))

        self.true_wx=(np.abs(lambd_x[np.argsort(lambd_x)]))
        self.true_wy=(np.abs(lambd_y[np.argsort(lambd_y)]))

        self.eigenvecs_x=np.array([phi_x[:,i]for i in np.argsort(lambd_x)])/np.max(phi_x)
        self.eigenvecs_y=np.array([phi_y[:,j]for j in np.argsort(lambd_y)])/np.max(phi_y)

    def set_damping(self,*args,cx=None,freqs_x=None,cy=None,freqs_y=None,Cx=None,Cy=None): 
        '''Function for creating a damping matrix for the system. By default Rayleigh
        damping with 5% damping specified in first and second mode in
        each direction is assumed.

        If more than two damping ratios are specified, the
        Rayleigh damping coefficients are determined by least squre fitting

        Run after set_mass_and_stiffness
        
        Arguments:
            cx:  1darray, optional
                Array of the desired damping ratios c/c_cr in x-direction
            freqs_x: 1darray, optional
                Array of the frequencies where the damping ratios are spicified
            cy:  1darray, optional
                Array of the desired damping ratios c/c_cr i y-direction
            freqs_y: 1darray, optional
                Array of the frequencies where the damping ratios are spicified
            Cx: ndarray, optional
                Complete damping matrix for the system in x direction, if desired. If not
                specified Rayleigh damping is assumed.
            Cy: ndarray, optional
                Complete damping matrix for the system in y direction, if desired. If not
                specified Rayleigh damping is assumed.'''

        if Cx: 
            self.CCx=Cx
        else: 
            # defining the rayleigh damping matrix:
            if not cx: 
                cx=np.array([0.05,0.05])
                freqs_x=self.true_wx[0:2]
            A=0.5* np.array([[1/wn,wn]for wn in freqs_x])
            ax,bx = np.linalg.lstsq(A, cx, rcond=None)[0]
            self.CCx=ax*self.MMxy+bx*self.KKx
            self.damping_ratios_x=0.5*(ax/self.true_wx+bx*self.true_wx)
        if Cy: 
            self.CCy=Cy
        else: 
            # defining the rayleigh damping matrix:
            if not cy: 
                cy=np.array([0.05,0.05])
                freqs_y=self.true_wy[0:2]
            A= 0.5* np.array([[1/wn,wn]for wn in freqs_y])
            ay,by = np.linalg.lstsq(A, cy, rcond=None)[0]
            self.CCy=ay*self.MMxy+by*self.KKy
            self.damping_ratios_y=0.5*(ay/self.true_wy+by*self.true_wy)
                
        self.CC=np.block([[self.CCx, np.zeros_like(self.CCx)],
                          [np.zeros_like(self.CCy), self.CCy]]) 
        
        
    @property
    def state_matrix(self):
        "CT State space  matrix (B)"
        M, C, K = self.MM, self.CC, self.KK
        Z = np.zeros_like(M)
        I = np.eye(M.shape[0])
        A11 = -np.linalg.solve(M, C)
        A12 = -np.linalg.solve(M, K)
        A = np.r_[np.c_[A11, A12],
                  np.c_[I, Z]]
        return A

    @property
    def input_influence_matrix(self):
        "CT State space input influence matrix (B)"
        n=len(self.KK)
        return np.r_[np.linalg.solve(self.MM, np.eye(n)),
                     np.zeros((n, n))]

    def get_state_space_matrices(self):
        "Continous time state space matrices A, B, C, D"
        A = self.state_matrix
        B = self.input_influence_matrix
        n = len(self.KK)
        O = np.zeros((n, n))
        I = np.eye(n)
        C = np.r_[A[:n, :],
                  np.c_[I, O],
                  np.c_[O, I]]
        D = np.r_[np.linalg.solve(self.MM, I),
                  O,
                  O]
        return A, B, C, D


    def simulate_response(self,t,F,*args,u0=None,du0=None):
        '''Function for obtaining a response for a given load- and time series. State space solution from Strid.
        
        Arguments: 
            F: 2darray
                Array containing the load for each dof. 
            t: 1darray
                Linear array contaning the time steps corresponding to the loads

                Shapes:
                t :  [ t1,  t2,  ...   , tn ]
            
                F :[ 
                    [ p1(t1), ... ,  p1(tn)]  <-- dof1, first floor x-dir:
                    .
                    [ pm(t1), ... ,  pm(tn)]  <-- dofm, top floor x-dir
                    [ pm+1(t1), ... ,pm+1(tn)]<-- dofm+1, first floor y-dir
                    .
                    [ pn(t1), ... ,  pn(tn)]  <-- dofn top floor y-dir    
                    ]  
                    

        Returns:
            ddu,du,u: 2darrays
                    All with same shape as F

        '''
        n = len(self.KK)
        u0 = np.zeros(n) if u0 is None else u0
        du0 = np.zeros(n) if du0 is None else du0
        x0 = np.r_[du0, u0]

        sys = scipy.signal.StateSpace(*self.get_state_space_matrices())
        U = np.zeros((t.size, n)) if F is None else F.T

        _, y, _ = scipy.signal.lsim(sys, U, t, X0=x0)
        y = y.T
        A = y[:n, :]
        V = y[n:2*n, :]
        D = y[2*n:, :]
        return A, V, D


    def display_system(self,F,U,t,*args,display_axis=False): 


        '''Display the shear frame, the loading and the response.
        Arguments: 
            F: 2darray
                loading, same input as '.simulate_response'
            U: 2darray
                the desired response, displacement, velocity or 
                acceleration. 
            t: 1darray
                timeseries of the loading
            display_axis: Boolean
                Whether or not to display the axis in the plots.
        
        '''
        n_story=len(self.KK)//2
        figure=plt.figure(figsize=(15,n_story*3*2+2*3))

        ax=plt.subplot2grid((n_story*4+5,3),(0,0),1,1,fig=figure)
        ax.set_axis_off()
        ax.text(0.25,0.25,'Loading',fontsize=40)

        ax=plt.subplot2grid((n_story*4+5,3),(0,1),1,1,fig=figure)
        ax.set_axis_off()
        ax.text(0.25,0.25,'X-direction',fontsize=40,)     
        
        ax=plt.subplot2grid((n_story*4+5,3),(0,2),1,1,fig=figure)
        ax.set_axis_off()
        ax.text(0.25,0.25,'Response',fontsize=40)       

        # last - plots:
        Fmax=np.amax(F)
        Fmin=np.amin(F)
        for i,ui in enumerate(F[:n_story]):
            ax=plt.subplot2grid((n_story*4+5,3),
            (2*(n_story-i)-1,0),2,1,fig=figure)
            ax.plot(t,ui)
            ax.set_xlim(t[0],t[-1])
            ax.set_ylim(Fmin,Fmax)
            ax.spines['bottom'].set_position(('data',0))
            ax.spines['top'].set_position(('data',0))
            ax.spines['right'].set_position(('data',0))
            if not display_axis:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
        
        #Response
        Umax=np.amax(U)
        Umin=np.amin(U)
        for i,ui in enumerate(U[:n_story]):
            ax=plt.subplot2grid((n_story*4+5,3),
            (2*(n_story-i)-1,2),2,1,fig=figure)
            ax.plot(t,ui)
            ax.set_xlim(t[0],t[-1])
            ax.set_ylim(Umin,Umax)
            ax.spines['bottom'].set_position(('data',0))
            ax.spines['top'].set_position(('data',0))
            ax.spines['right'].set_position(('data',0))
            if not display_axis:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        # Frame:
        for i in range(n_story):
            ax=plt.subplot2grid((n_story*4+5,3),(2*(n_story-i),1),2,1,fig=figure)
            ax.set_xlim(-0.5,1.5)
            ax.set_ylim(0,1)
            ax.set_axis_off()
            x=([0,0])
            y1=([0,1])
            y2=([1,1])
            if i==0:
                ax.plot(([-0.1,0.1]),x,linewidth=10,color='black')
                ax.plot(([0.9,1.1]),x,linewidth=10,color='black')
                ax.plot(([-0.5,1.5]),x,linewidth=5,color='black')
            ax.plot(x,y1,linewidth=10,color='black')
            ax.plot(y1,y2,linewidth=40,color='black')
            ax.plot(y2,y1,linewidth=10,color='black')
            ax.text(0.2,0.8,'M = '+str(round(self.m[i],2)),fontsize=25)  
            ax.text(0.08,0.4,'K = '+str(round(self.kx[i],2)),fontsize=25)  
            plt.scatter([-0.3,1.3],[0.96,0.96],marker=9,s=150,c='black')
            plt.scatter([-0.3,1.3],[0.96,0.96],marker=0,s=200,c='black')

        #Seperating lines: 
        ax=plt.subplot2grid((n_story*4+5,3),(2*(n_story)+1,0),1,1,)
        ax.plot(([-0.5,1.5]),x,linewidth=5,color='black')
        ax.set_xlim((-0.5,1.5))
        ax.set_ylim((0,1))
        ax.set_axis_off()

        ax=plt.subplot2grid((n_story*4+5,3),(2*(n_story)+1,2),1,1,)
        ax.plot(([-0.5,1.5]),x,linewidth=5,color='black')
        ax.set_xlim((-0.5,1.5))
        ax.set_ylim((0,1))
        ax.set_axis_off()

        ##### Y-direction:
        # 
        a=2*n_story+2 
        ax=plt.subplot2grid((n_story*4+5,3),(a,0),1,1,fig=figure)
        
        ax.set_axis_off()
        ax.text(0.25,0.25,'Loading',fontsize=40)

        ax=plt.subplot2grid((n_story*4+5,3),(a,1),1,1,fig=figure)
        ax.set_axis_off()
        ax.text(0.25,0.25,'Y-direction',fontsize=40,)     
        
        ax=plt.subplot2grid((n_story*4+5,3),(a,2),1,1,fig=figure)
        ax.set_axis_off()
        ax.text(0.25,0.25,'Response',fontsize=40)       

        # last - plots:
        

        Fmax=np.amax(F)
        Fmin=np.amin(F)
        # a+=1
        for i,ui in enumerate(F[n_story:]):
            ax=plt.subplot2grid((n_story*4+5,3),(a+2*(n_story-i)-1,0),2,1,fig=figure)
            ax.plot(t,ui)
            ax.set_xlim(t[0],t[-1])
            ax.set_ylim(Fmin,Fmax)
            ax.spines['bottom'].set_position(('data',0))
            ax.spines['top'].set_position(('data',0))
            ax.spines['right'].set_position(('data',0))
            if not display_axis:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
        
        #Response
        Umax=np.amax(U)
        Umin=np.amin(U)
        for i,ui in enumerate(U[n_story:]):
            ax=plt.subplot2grid((n_story*4+5,3),(a+2*(n_story-i)-1,2),2,1,fig=figure)
            ax.plot(t,ui)
            ax.set_xlim(t[0],t[-1])
            ax.set_ylim(Umin,Umax)
            ax.spines['bottom'].set_position(('data',0))
            ax.spines['top'].set_position(('data',0))
            ax.spines['right'].set_position(('data',0))
            if not display_axis:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        # Frame:
        for i in range(n_story):
            ax=plt.subplot2grid((n_story*4+5,3),(a+2*(n_story-i),1),2,1,fig=figure)
            ax.set_xlim(-0.5,1.5)
            ax.set_ylim(0,1)
            ax.set_axis_off()
            x=([0,0])
            y1=([0,1])
            y2=([1,1])
            if i==0:
                ax.plot(([-0.1,0.1]),x,linewidth=10,color='black')
                ax.plot(([0.9,1.1]),x,linewidth=10,color='black')
            ax.plot(x,y1,linewidth=10,color='black')
            ax.plot(y1,y2,linewidth=40,color='black')
            ax.plot(y2,y1,linewidth=10,color='black')
            ax.text(0.2,0.8,'M = '+str(round(self.m[i],2)),fontsize=25)  
            ax.text(0.08,0.4,'K = '+str(round(self.ky[i],2)),fontsize=25)  
            plt.scatter([-0.3,1.3],[0.96,0.96],marker=9,s=150,c='black')
            plt.scatter([-0.3,1.3],[0.96,0.96],marker=0,s=200,c='black')


        plt.tight_layout(h_pad=-1)
        
        return figure