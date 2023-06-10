import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import axes3d 
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import time
import matplotlib
# matplotlib.use('tkAgg')
import warnings
import matplotlib.colors as mc
import matplotlib.cm as cm
import pickle
import matplotlib.image as im

from matplotlib.widgets import Slider, Button,TextBox

warnings.filterwarnings('ignore','',np.ComplexWarning)

def display_modes_from_Ts(Ts):
    phi=Ts.PHI_median
    # phi=Ts.phi_auto[0].T
    plot=mode_plot(phi)
    plot.show()

class mode_plot():
    ''' Class for pLotting an ineractive plot of a mode. 

    Arguments: 

        phi: 1darray or 2darray: 
            
            len: 3*len DOFs_sensors
            Arr([dx1,dy1,dz1,dx2,dy2,dz2.....])
                =[H1E_x,H1E_y,H1E_z,H1W_x,...,H11W_z]
        
    Additional info on how PHI is bulit up: 
            [H1E_x, H1E_y,  H1E_z,  H1W_x,  H1W_y,  H1W_z,  
            H2W_x,  H2W_y,  H2W_z, 
            H3E_x,  H3E_y,  H3E_z,  H3W_x,  H3W_y,  H3W_z,
            H4E_x,  H4E_y,  H4E_z,  H4W_x,  H4W_y,  H4W_z,  
            H5E_x,  H5E_y,  H5E_z,  H5W_x,  H5W_y,  H5W_z, 
            H6E_x,  H6E_y,  H6E_z,  H6W_x,  H6W_y,  H6W_z,  
            H7E_x,  H7E_y,  H7E_z,  H7W_x,  H7W_y,  H7W_z,
            H8E_x,  H8E_y,  H8E_z,  
            H9E_x,  H9E_y,  H9E_z,  H9W_x,  H9W_y,  H9W_z, 
            H10E_x, H10E_y, H10E_z,H10W_x, H10W_y, H10W_z, 
            H11E_x, H11E_y, H11E_z,H11W_x, H11W_y, H11W_z]

            H1-H9:   bridge deck
            H10-H11: Towers

            E= east side
            W= west side

            slicing axamples: 
            #slicing:

            bridge_west_side_x=phi[:16*3:3][[1,2,4,6,8,10,12,15]] #len=8
            bridge_east_side_x=phi[:16*3:3][[0,3,5,7,9,11,13,14]] #len=8

            bridge_west_side_y=phi[1:16*3:3][[1,2,4,6,8,10,12,15]] #len=8
            bridge_east_side_y=phi[1:16*3:3][[0,3,5,7,9,11,13,14]] #len=8

            bridge_west_side_z=phi[2:16*3:3][[1,2,4,6,8,10,12,15]] #len=8
            bridge_east_side_z=phi[2:16*3:3][[0,3,5,7,9,11,13,14]] #len=8

            tower_10=phi[16*3:18*3]
            tower_11=phi[18*3:20*3]
            '''


    def __init__(self,PHI,*args,bridge_axis=None,figsize=(10,10)):
        #changing matplotlib backens to make interactive plot.
        matplotlib.use('tkagg')
        self.lw=3
        
        if type(PHI)==list:
            PHI=np.array(PHI)
        if len(PHI.shape)>1:
            phi=PHI[0]
        else: 
            phi=PHI
        self.PHI=PHI
        
        self.current_mode=0
        # setting up the bridges initial configuration
        H1E=np.array([ 480.0, 6.33,-8.38])  # 0
        H1W=np.array([ 480.0,-6.64,-8.38])
        H2W=np.array([ 360.0,-6.64,-6.64]) 
        H3E=np.array([ 240.0, 6.33,-4.45])
        H3W=np.array([ 240.0,-6.64,-4.45]) 
        H4E=np.array([ 120.0, 6.33,-2.48]) 
        H4W=np.array([ 120.0,-6.64,-2.48]) 
        H5E=np.array([-7.0  , 6.33, -0.4]) 
        H5W=np.array([-7.0  ,-6.64, -0.4]) 
        H6E=np.array([-120.0, 6.33,-2.25]) 
        H6W=np.array([-120.0,-6.64,-2.25]) 
        H7E=np.array([-240.0, 6.33,-4.22]) 
        H7W=np.array([-240.0,-6.64,-4.22]) 
        H8E=np.array([-360.0, 6.33,-6.18]) 
        H9E=np.array([-480.0, 6.33,-8.15]) 
        H9W=np.array([-480.0,-6.64,-8.15]) # 15

        H10E=np.array([655  ,  4.5, 120.5]) 
        H10W=np.array([655  , -4.5, 120.5]) 
        H11E=np.array([-655 ,  4.5, 120.5]) 
        H11W=np.array([-655,  -4.5, 120.5]) 

        DOFs_sensors=np.array([
            H1E,H1W,H2W,H3E,H3W,H4E,H4W,H5E,H5W,H6E,H6W,H7E,
            H7W,H8E,H9E,H9W,H10E,H10W,H11E,H11W])


        #moves the first tower to the beginning of the list 
        DOFs_sensors=DOFs_sensors[[16,17,0,1,2,3,4,5,6,7,8,
                                9,10,11,12,13,14,15,18,19]]

        # [H10E,H10W,H1E,H1W,H2W,H3E,H3W,H4E,H4W,H5E,H5W,H6E,
        #   H6W,H7E,H7W,H8E,H9E,H9W,H11E,H11W]

        self.DOFs_sensors=DOFs_sensors

        #interpolate for the two missing sensors in the bridge deck. 
        H2E_int= np.array([(DOFs_sensors[2,0]+DOFs_sensors[5,0])/2,
                  DOFs_sensors[4,1]+12.97,
                  (DOFs_sensors[2,2]+DOFs_sensors[5,2]+
                   2*DOFs_sensors[4,2])/4])
        
        H8W_int= np.array([(DOFs_sensors[14,0]+DOFs_sensors[17,0])/2,
            DOFs_sensors[15,1]-12.97,
            (DOFs_sensors[14,2]+DOFs_sensors[17,2]+
            5*DOFs_sensors[15,2])/7])
        # H8W_int = (DOFs_sensors[14]+DOFs_sensors[17])/2

        #Defining the structures base points 
        self.base_H10E=np.array([655  ,  7,   -60]) 
        self.base_H10W=np.array([655  , -7.3, -60]) 
        self.base_H11E=np.array([-655 ,  7,   -60]) 
        self.base_H11W=np.array([-655,  -7.3, -60]) 

        #dummy pinots between base an top of tower to connect bridge deck to
        dummy_H10E =self.dummy_point(DOFs_sensors[0],self.base_H10E,-10.6)  
        dummy_H10W=self.dummy_point(DOFs_sensors[1],self.base_H10W,-10.6)
        dummy_H11E=self.dummy_point(DOFs_sensors[-2],self.base_H11E,-10.6)
        dummy_H11W=self.dummy_point(DOFs_sensors[-1],self.base_H11W,-10.6) 

        #adding interpolated points to DOFs 
        self.DOFs=np.insert(DOFs_sensors,4,H2E_int,axis=0)
        self.DOFs=np.insert(self.DOFs,17,H8W_int,axis=0)
  
        #adding base and dummy point for tower 1:
        self.DOFs=np.insert(self.DOFs,2,self.base_H10E,axis=0)
        self.DOFs=np.insert(self.DOFs,3,self.base_H10W,axis=0)
        self.DOFs=np.insert(self.DOFs,4,dummy_H10E,axis=0)
        self.DOFs=np.insert(self.DOFs,5,dummy_H10W,axis=0)
        
        #adding base and dummy point for tower 2        
        self.DOFs=np.insert(self.DOFs,24,dummy_H11E,axis=0)
        self.DOFs=np.insert(self.DOFs,25,dummy_H11W,axis=0)
        self.DOFs=np.insert(self.DOFs,26,self.base_H11E,axis=0)
        self.DOFs=np.insert(self.DOFs,27,self.base_H11W,axis=0)
        
        #setting up the plot
        self.x0=self.DOFs[:,0]
        self.y0=self.DOFs[:,1]
        self.z0=self.DOFs[:,2]
        self.phi=self.sort_phi(phi)
        self.fig = plt.figure(figsize=figsize)
        self.ax=self.fig.add_subplot(111,projection='3d')
        if bridge_axis != None:
            self.ax=bridge_axis

        self.ax.set_ylim(-100,100)
        self.ax.set_xlim(-700,700)
        self.ax.set_zlim(-20,10)
        self.ax.set_box_aspect([100,35,3])
        self.ax.set_axis_off()
        plt.subplots_adjust(left=-0.4, bottom=-0.4,
        right=1.4, top=1.4, wspace=0, hspace=0)

        # plotting initial configuration 

        self.dots = self.ax.plot(self.x0,self.y0,
        self.z0,marker='.',
        markersize=self.lw*4,linestyle='None')[0]

        self.dots.set_markevery([0,1,6,7,9,10,11,12,13,14,15,16,17,18,19,20
        ,22,23,28,29])

        self.dots.set_markerfacecolor('peru')
        self.dots.set_markeredgecolor('peru')

        X=np.array([self.x0[::2],self.x0[1::2]])
        Y=np.array([self.y0[::2],self.y0[1::2]])
        Z=np.array([self.z0[::2],self.z0[1::2]])

        points = np.array([X,Y,Z]).T.reshape(-1, 1, 3)
        segments_xz =np.concatenate([points[:-2], points[2:]], axis=1)
        segments_y =np.concatenate([points[:-1], points[1:]], axis=1)
        segments=np.vstack((segments_xz,segments_y[::2]))

        self.lines=Line3DCollection(segments, linewidths=self.lw)
        self.ax.add_collection(self.lines)                    

        ints=np.array([H2E_int,H8W_int,dummy_H10E,dummy_H10W
        ,dummy_H11E,dummy_H11W])
        self.dots_int=self.ax.plot(ints[:,0],ints[:,1],ints[:,2],
        marker='.',markersize=self.lw*4,linestyle='None',
        alpha=1,color='grey')[0]

        ##############################################
        #plotting undeformed configuration to stay in bg
        self.dots_perm = self.ax.plot(self.x0,self.y0,self.z0,
        marker='.',linestyle='None',color='grey',alpha=0.2)[0]

        b_ix=[26,27,2,3]
        self.dots_perm2 = self.ax.plot(self.x0[b_ix],self.y0[b_ix],
        self.z0[b_ix],marker='s',linestyle='None',
        color='black',alpha=0.5)[0]

        X=np.array([self.x0[::2],self.x0[1::2]])
        Y=np.array([self.y0[::2],self.y0[1::2]])
        Z=np.array([self.z0[::2],self.z0[1::2]])

        # self.lines_perm=self.ax.plot_wireframe(X,Y,Z,
        # rstride=1,cstride=1,color='grey',alpha=0.2)
        #-----
        self.lines_perm=Line3DCollection(segments,color='grey',alpha=0.25,
                                         linewidth=self.lw*0.6)
        self.ax.add_collection(self.lines_perm)    
        #---

        #setting up the sliders

        axfreq = self.fig.add_axes([0.05, 0.05, 0.03, 0.6])
        self.fac_slider = Slider(
        ax=axfreq,
        label='Amplification\n factor',
        valmin=0,
        valmax=110,
        valinit=30,
        orientation="vertical")

        self.axfreq = self.fig.add_axes([0.1, 0.05, 0.03, 0.6])
        self.t_slider = Slider(
        ax=self.axfreq,
        label='Phase',
        valmin=-1,
        valmax=1,
        valinit=1,
        orientation="vertical")

        #setting up the button: 
        self.axplay = self.fig.add_axes([0.05, 0.72, 0.02, 0.02])
        self.button_play = Button(self.axplay, 'Play')
        self.button_play.on_clicked(self.play)
        


        #setting up the mode selector feild: 
        if len(PHI.shape)>1:
            self.axbox = self.fig.add_axes([0.05, 0.78, 0.02, 0.02])
            self.text_box = TextBox(self.axbox,
                                     "", textalignment="center")
            self.axbox.text(0.5,1.35,
                             'Mode number:',
                               horizontalalignment='center',
                            verticalalignment='center')
            
            self.text_box.on_submit(self.change_mode)
            #setting up the button: 
            self.axprev = self.fig.add_axes([0.04, 0.75, 0.02, 0.02])
            self.button_prev = Button(self.axprev, 'Prev')
            self.button_prev.on_clicked(self.prev_mode)
            
            #setting up the button: 
            self.axnext = self.fig.add_axes([0.06, 0.75, 0.02, 0.02])
            self.button_next = Button(self.axnext, 'Next')
            self.button_next.on_clicked(self.next_mode)
            
            
        self.fac_slider.on_changed(self.update_DOFs)
        self.t_slider.on_changed(self.update_DOFs)

        self.calculate_colors()
        self.update_DOFs('event')
        self.update_DOFs('event')


            
    def calculate_colors(self):
        perm_lines=self.lines_perm._segments3d
        live_lines=self.lines._segments3d
        
        color_factors=np.array([])
        for i in range(len(live_lines)):
            p_line=perm_lines[i]
            l_line=live_lines[i]
            ddd=np.array([])
            for j in range(len(p_line)):
                dx=p_line[j][0]-l_line[j][0]
                dy=p_line[j][1]-l_line[j][1]
                dz=p_line[j][2]-l_line[j][2]
                ddd=np.append(ddd,np.sqrt(dx**2+dy**2+dz**2))
            line_defs=(ddd[:-1]+ddd[1:])/2
            color_factors=np.append(color_factors,line_defs,)
        norms=mc.Normalize()
        normalized=norms.__call__(color_factors)*np.abs(self.t_slider.val)
        colors=cm.viridis(normalized)
        self.colors=colors
    
    def update_DOFs(self,event):
        

        fac=self.fac_slider.val*self.t_slider.val
        DOFs_i=np.zeros_like(self.DOFs_sensors[:])
        DOFs_i[:,0]=self.DOFs_sensors[:,0]+self.phi[0::3]*fac
        DOFs_i[:,1]=self.DOFs_sensors[:,1]+self.phi[1::3]*fac
        DOFs_i[:,2]=self.DOFs_sensors[:,2]+self.phi[2::3]*fac

        # print(fac)
        #interpolate for the two missing sensors in the bridge deck. 
        # H2E_int_i= (DOFs_i[2]+DOFs_i[5])/2
        # H8W_int_i = (DOFs_i[14]+DOFs_i[17])/2

        H2E_int_i= np.array([(DOFs_i[2,0]+DOFs_i[5,0])/2,
            DOFs_i[4,1]+12.97,
            (DOFs_i[2,2]+DOFs_i[5,2]+
            5*DOFs_i[4,2])/7])
        
        H8W_int_i= np.array([(DOFs_i[14,0]+DOFs_i[17,0])/2,
            DOFs_i[15,1]-12.97,
            (DOFs_i[14,2]+DOFs_i[17,2]+
            5*DOFs_i[15,2])/7])
 
        #dummy pinots between base an top of tower to connect bridge deck to
        dummy_H10E =self.dummy_point(DOFs_i[0],self.base_H10E,-10.6)  
        dummy_H10W=self.dummy_point(DOFs_i[1],self.base_H10W,-10.6)
        dummy_H11E=self.dummy_point(DOFs_i[-2],self.base_H11E,-10.6)
        dummy_H11W=self.dummy_point(DOFs_i[-1],self.base_H11W,-10.6) 

        #adding interpolated points to DOFs 
        DOFs_i=np.insert(DOFs_i,4,H2E_int_i,axis=0)
        DOFs_i=np.insert(DOFs_i,17,H8W_int_i,axis=0)
  
        #adding base and dummy point for tower 1:
        DOFs_i=np.insert(DOFs_i,2,self.base_H10E,axis=0)
        DOFs_i=np.insert(DOFs_i,3,self.base_H10W,axis=0)
        DOFs_i=np.insert(DOFs_i,4,dummy_H10E,axis=0)
        DOFs_i=np.insert(DOFs_i,5,dummy_H10W,axis=0)
        
        #adding base and dummy point for tower 2        
        DOFs_i=np.insert(DOFs_i,24,dummy_H11E,axis=0)
        DOFs_i=np.insert(DOFs_i,25,dummy_H11W,axis=0)
        DOFs_i=np.insert(DOFs_i,26,self.base_H11E,axis=0)
        DOFs_i=np.insert(DOFs_i,27,self.base_H11W,axis=0)

        ints=np.array([H2E_int_i,H8W_int_i])

        # updating the sensor DOFs
        self.dots.set_data_3d(DOFs_i[:,0],DOFs_i[:,1],DOFs_i[:,2])

        ints=np.array([H2E_int_i,H8W_int_i,dummy_H10E,dummy_H10W
        ,dummy_H11E,dummy_H11W])
        self.dots_int.set_data_3d(ints[:,0],ints[:,1],ints[:,2])

        #updating wireframe
        X=np.array([DOFs_i[0::2,0],DOFs_i[1::2,0]])
        Y=np.array([DOFs_i[0::2,1],DOFs_i[1::2,1]])
        Z=np.array([DOFs_i[0::2,2],DOFs_i[1::2,2]])
        
        self.lines.remove()

        points = np.array(
            [DOFs_i[:,0],DOFs_i[:,1],DOFs_i[:,2]]).T.reshape(-1, 1, 3)
        # points = np.array([X,Y,Z]).T.reshape(-1, 1, 3)
        segments_xz =np.concatenate([points[:-2], points[2:]], axis=1)
        segments_y =np.concatenate([points[:-1], points[1:]], axis=1)
        self.segments=np.vstack((segments_xz,segments_y[::2]))
        # self.lines.set_segments(segments)
        self.calculate_colors()
        self.lines=Line3DCollection(self.segments,
                                     linewidths=self.lw,colors=self.colors)
        self.ax.add_collection(self.lines)  


        #------
        
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        # time.sleep(0.1)

    def dummy_point(self,st,sb,z): 
        h_fac=(z-sb[2])/(st[2]-sb[2])  #from bot / tot heigth
        x=sb[0]+(st[0]-sb[0])*h_fac
        y=sb[1]+(st[1]-sb[1])*h_fac
        return np.array([x,y,z])
        
    def show(self):
        self.axfreq.legend([self.dots,self.dots_int,self.dots_perm2],
        ['Sensor placements','Interpolated points','Supports'],
        loc='center', bbox_to_anchor=(3.5, 0.05))
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show(block=True)   
    
    def sort_phi(self,phi):
        return phi[[48,49,50,51,52,53,0,1,2,3,4,5,6,7,8,9,10,
                        11,12,13,14,15,16,17,18,19,20,21,22,
                        23,24,25,26,27,28,29,30,31,32,33,34,35,
                        36,37,38,39,40,41,42,43,44,45,46,47,54,
                        55,56,57,58,59]]
    def play(self,event):
        t=0
        freq=2
        fps=18
        while t< 2:
            t0=time.time()
            val=np.sin(t*2*np.pi*freq)
            self.fig.canvas.flush_events()
            self.t_slider.set_val(val)
            t+=1/fps
            if 1/fps > (time.time()-t0):
                time.sleep(1/fps-(time.time()-t0))
                
    def next_mode(self,event):
        self.change_mode(self.current_mode+1)

    def prev_mode(self,event):
        self.change_mode(self.current_mode-1)
        
    def change_mode(self,i):
        try:
            self.current_mode=i
            m_i=int(i)
            if m_i>len(self.PHI):
                pass
            phi_i=self.PHI[m_i]
            self.phi=self.sort_phi(phi_i)
            self.t_slider.set_val(self.t_slider.val)
            self.t_slider.set_val(self.t_slider.val)

        except ValueError:
            pass
        except IndexError:
            pass
        
    def save_bridge_axis(self,path):
        with open(path, 'wb') as fout:
            pickle.dump(self, fout)
    
    def return_bridge_ax(self):
        return self.ax
    def get_line_collection(self):
        return self.lines


    def save_snapshot(self,elev,azim,roll):
        self.t_slider.set_val(1)
        for ax in self.fig.axes:
            if ax != self.ax:
                ax.set_visible(False)
        self.ax.view_init(elev=elev, azim=azim, roll=roll)
        
        self.ax.set_box_aspect(aspect=(100,35,3),zoom=0.60)
        self.fig.savefig(fname='test_view.png',dpi=400)

        for ax in self.fig.axes:
            if ax != self.ax:
                ax.set_visible(True)
            

    def report_figure(self,n,freq,damp,n_ids,*args,col=None):
        self.rep_fig=plt.figure(figsize=(7,2)
                                )

        ax_xz=plt.subplot2grid((2,6),loc=(0,0),rowspan=1,colspan=4,
                                fig=self.rep_fig)
        self.lw=5
        self.save_snapshot(0,-90,0)
        image=plt.imread('test_view.png')
        h=image.shape[1]
        h1=int(h*0.4)
        image=image[h1:-1*h1,:,:]
        # ax_xz.set_visible(False)
        ax_xz.set_axis_off()
        ax_xz.imshow(image)
        ax_xz.set_title('Side view:',loc='left',fontsize='small')


        ax_xy=plt.subplot2grid((2,6),loc=(1,0),rowspan=1,colspan=4,
                                fig=self.rep_fig)
        self.lw=5
        self.save_snapshot(-90,0,90)
        image=plt.imread('test_view.png')
        h=image.shape[1]
        h1=int(h*0.4)
        image=image[h1:-1*h1,:,:]

        # ax_xz.set_visible(False)
        ax_xy.set_axis_off()
        ax_xy.imshow(image)
        ax_xy.set_title('Top view:',loc='left',fontsize='small')

        ax_pers=plt.subplot2grid((2,6),loc=(0,4),rowspan=2,colspan=2,
                                fig=self.rep_fig)
        self.lw=3
        self.save_snapshot(10,-20,0)
        image=plt.imread('test_view.png')
        h=image.shape[1]
        fac=int(h*0.25)

        image=image[fac:-1*fac,fac:-1*fac,:]
        # ax_xz.set_visible(False)
        ax_pers.set_axis_off()
        ax_pers.imshow(image)
        if col != None:
            line=ax_pers.plot([1,1],[1,1],linewidth=3,c=col,label='Color in traceplot')
            self.rep_fig.legend(handles=line,loc='lower right')
        self.rep_fig.subplots_adjust(left=0, bottom=-0.1, right=1, top=0.9,
                         wspace=-0.1, hspace=-0.1)
        p_freq=f'{freq:.2}'
        p_damp=f'{damp:.2}'
        
        

        ids=',  traced in '+str(n_ids)+' segments'
        self.rep_fig.suptitle(r'Mode '+str(n)+r', f = '+str(p_freq)+r' Hz   $\zeta = $'+p_damp+ids,size='medium')
        name='Modeplot_'+str(n)+'.png'
        self.rep_fig.savefig(name,dpi=400)
        plt.close()
        # plt.savefig(fname='test_view_zoom.png',dpi=200)


    def gan_report_figures_for_freqs(self,freqs,damps,*args,phi_indices=None):
        if phi_indices == None:
            phi_indices = range(len(freqs))

        for mode_ix,freq,damp in zip(phi_indices,freqs,damps):

            self.change_mode(mode_ix)
            self.report_figure(mode_ix+1,freq,damp) 

