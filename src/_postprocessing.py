import numpy as np 
import os
import sys 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from preposessing import *
from scipy import signal
from mode_visulizer import *
from sklearn.linear_model import LinearRegression

import seaborn as sns
sns.set_theme(context='paper', style = 'ticks', color_codes='deep')

import warnings


    

class Postprocessor():
    def __init__(self):
        self.trace=None
        self.file_names=[]
        self.dates=[]
        self.vertical=[]
        self.horisontal=[]
        self.frequencies=[]
        self.mode_shapes=[]
        self.temp=[]
        self.angle=[]
        self.ullensvang_temp_data=None        
        self.ullensvang_temps=[]
#-------------------
    def set_dates(self):
        dates=[]
        for file_name in self.file_names:
            date=file_name[7:17]
            time=file_name[18:23].replace('-',':')

            dates.append(date+'T'+time) #YYYY-MM-DD_hh-mm
        self.dates=np.asarray(dates,dtype='datetime64[m]')
#----------------------
    def assign_trace(self,trace,names):
        '''
        Arguments: 
            trace: Trace object
                a single traces stamped as physical
            names: list
                list of the file names for the imported time segments
                
                '''
        self.trace=trace
        self.frequencies=trace.frequencies
        self.mode_shapes=trace.mode_shapes
        self.ixs=self.trace.time_seg

        for ix in self.ixs:
            self.file_names.append(names[ix])

#-----------------------
    def import_temp_fromUllensvang(self,filepath):
        data=pd.read_csv(filepath,sep=';')
        data['Tid(norsk normaltid)']=pd.to_datetime(data['Tid(norsk normaltid)'],format='%d.%m.%Y %H:%M')
        data.set_index(['Tid(norsk normaltid)'],inplace=True)
        self.ullensvang_temp_data=data[data['Navn']=='Ullensvang Forsøksgard']

#-----------------------
    def get_ullensvang_temp_from_ts(self,file_name):
        date=file_name[7:20] #YYYY-MM-DD_hh
        date_obj=pd.to_datetime(date,format='%Y-%m-%d_%H')
        df=self.ullensvang_temp_data
        idx = df.iloc[df.index.get_indexer((date_obj,), method='nearest')]
        return idx['Lufttemperatur'][0]


#-----------------------

    def assign_wheater_from_A6(self,A6_path):
        for file in self.file_names:
            wheater=import_converted_ts(A6_path,file)
            h=np.nanmean(wheater[0]) 
            v=np.nanmean(wheater[1]) 
            a=np.nanmean(wheater[2]) 
            T=np.nanmean(wheater[3])
            self.horisontal.append(h) 
            self.vertical.append(v)
            self.angle.append(a)
            self.temp.append(T)
#-----------------------

    def assign_weather_from_ts(self,ts,name): 
        temp=[]
        for sensor in ['A1_vertical','A2_vertical','A3_vertical',
               'A4_vertical', 'A6_vertical','A7_vertical',
               'A8_vertical']:
            try:
                series=ts.__getattr__(sensor)
            except:
                continue
            mean=series[:2060].mean(skipna=True)
            temp.append(mean)
        if all(np.isnan(temp)):
            self.vertical.append(np.nan)
        else:
            self.vertical.append(np.nanmean(temp))
        #-----------
        temp=[]
        for sensor in [ 'A1_horizontal','A2_horizontal',
                 'A3_horizontal', 'A4_horizontal', 
            'A6_horizontal','A7_horizontal','A8_horizontal',]:
            try:
                series=ts.__getattr__(sensor)
            except:
                continue
            mean=series[:2060].mean(skipna=True)
            temp.append(mean)
        if all(np.isnan(temp)):
            # print('nan')
            self.horisontal.append(np.nan)
        else:
            
            self.horisontal.append(np.nanmean(temp))
        #-----------
        temp=[]
        for sensor in ['A1_angel','A2_angel','A3_angel',
                  'A4_angel','A6_angel','A7_angel',
                    'A8_angel']:
            try:
                series=ts.__getattr__(sensor)
            except:
                continue
            mean=series[:2060].mean(skipna=True)
            temp.append(mean)
       
        if all(np.isnan(temp)):
            self.angle.append(np.nan)
        else:
            self.angle.append(np.nanmean(temp))
        #-----------
        temp=[]
        for sensor in ['A1_temperature','A2_temperature','A3_temperature',
                 'A4_temperature','A6_temperature','A7_temperature',
                 'A8_temperature']:
            try:
                series=ts.__getattr__(sensor)
            except:
                continue
            mean=series[:2060].mean(skipna=True)
            temp.append(mean)
        if all(np.isnan(temp)):
            self.temp.append(np.nan)
        else:
            self.temp.append(np.nanmean(temp))
        #-----------------
        self.ullensvang_temps.append(
            self.get_ullensvang_temp_from_ts(name))

#-----------------------
        
    def get_weather_report(self):
        print('Mean horisontal wind: ',len(self.horisontal))
        print('Mean angle wind     : ',(self.angle))
        print('Mean vertical wind  : ',(self.vertical))
        print('Mean temperature    : ',(self.temp))


#-----------------------
    def import_weather(self,folder_path):
        for file in self.file_names:
            ts=import_converted_ts(
                folder_path,file)
            self.assign_weather_from_ts(ts,file)
#-----------------------

    def data_axis_array(self,n):
        out=[]
        date_ixs=np.arange(0,len(self.dates),len(self.dates)//n)
        for i in range(len(self.dates)):
            if i in date_ixs:
                out.append(self.dates[i])
            else: 
                out.append(' ')
        return out
 #-----------------------   
    def data_axis_array2(self,n):
        out=[]
        date_ixs=np.arange(0,len(self.dates),len(self.dates)//n)
        for i in range(len(self.dates)):
            if i in date_ixs:
                out.append(self.dates[i])
        return np.arange(len(self.dates))[date_ixs],out
#-----------------------
    def plot_comparisons(self):
        self.set_dates()
        matplotlib.use('module://matplotlib_inline.backend_inline')
        fig=plt.figure(figsize=(7,8))

        dates_ax=self.data_axis_array(15)
        freq_ax=plt.subplot2grid((5,1),(0,0),rowspan=1,colspan=1,fig=fig)
        freq_ax.plot(np.arange(len(dates_ax)),
                     self.trace.frequencies,linewidth=0.4)
        freq_ax.set_ylabel('Frequency')

        freq_ax.tick_params(axis='x', labelrotation=45)
        freq_ax.set_xticks(np.arange(len(dates_ax)),minor=True)
        # freq_ax.set_xticklabels(dates_ax)
        major_x,labels=self.data_axis_array2(15)
        freq_ax.set_xticks(major_x,labels=labels)
        freq_ax.set_xlabel('Date')
        freq_ax.set_xlim(0,len(dates_ax))
        E=np.mean(self.trace.frequencies)
        SIG=np.std(self.trace.frequencies)
        freq_ax.set_ylim(E-3*SIG,E+3*SIG)

        #filter:
        q=len(self.trace.frequencies)//20
        n=8
        sos2 = signal.cheby1(n, 0.05, 0.8 / q, output='sos')
        sos2 = np.asarray(sos2)
        filtered_1 = signal.sosfiltfilt(sos2,self.trace.frequencies-E)
        freq_ax.plot(np.arange(len(dates_ax)),
                     filtered_1+E,linewidth=1)

        hor_ax=plt.subplot2grid((5,1),(1,0),rowspan=1,colspan=1,
                                fig=fig,sharex=freq_ax)
        hor_ax.plot(np.arange(len(dates_ax)),self.horisontal,linewidth=0.4)
        hor_ax.set_ylabel('Horisontal \nmean wind')
        hor_ax.set_ylim(-20,20)


        angle_ax=plt.subplot2grid((5,1),(2,0),rowspan=1,colspan=1,
                                  fig=fig,sharex=freq_ax)
        # angle_ax.plot(np.arange(len(dates_ax)),self.angle,linewidth=0.4)
        angle_ax.set_ylabel('Trends')
        #freq trend:
        m_f=np.max(np.max(filtered_1))
        angle_ax.plot(np.arange(len(dates_ax)),
                     filtered_1/m_f,linewidth=1,label='freq trend')
        ###################
        filtered_1_temp = signal.sosfiltfilt(sos2,self.ullensvang_temps)
        m_t=np.max(np.abs(filtered_1_temp))
        angle_ax.plot(np.arange(len(dates_ax)),
                     filtered_1_temp/m_t,linewidth=1,label='temp trend')  
            
        filtered_hor = signal.sosfiltfilt(sos2,np.abs(self.horisontal))
        m_h=np.max(np.abs(filtered_hor))
        angle_ax.plot(np.arange(len(dates_ax)),
                     filtered_hor/m_h,linewidth=1,label='wind trend')
        angle_ax.legend(ncols=3)


        ####################
        

        temp_ax=plt.subplot2grid((5,1),(3,0),rowspan=1,colspan=1,
                                 fig=fig,sharex=freq_ax)
        # temp_ax.plot(np.arange(len(dates_ax)),self.temp,linewidth=0.4)
        temp_ax.set_ylabel('Temperature')
        temp_ax.tick_params(axis='x', labelrotation=45)
        temp_ax.grid(which='major')
        temp_ax.set_xlabel('Date')
        temp_ax.set_ylim(-20,30)

        temp_ax.plot(np.arange(len(dates_ax)),
                     self.ullensvang_temps,color='red',linewidth=0.4)


        for ax in [hor_ax, angle_ax,freq_ax]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_yticks(ax.get_yticks()[1:])
            ax.tick_params(axis='x', which='minor', bottom=False)  
            ax.grid(which='major')
            ax.set_xlim(0,len(dates_ax))
        # plt.tight_layout(hspace=0)
        fig.subplots_adjust(hspace=0.03)
        plt.savefig(fname='testfig.png',dpi=500)
        plt.close()


    def get_damping_from_trace(self,folder_path):
        self.dampings=[]
        for i,filename in enumerate(self.file_names):
            #iterates the segments in the trace
            omega=self.trace.frequencies[i]*2*np.pi
            ts=import_converted_ts(folder_path,filename)
            cluster_ix=np.argmin(np.abs(ts.lambds_median-omega))
            damping=ts.xi_auto[cluster_ix]
            self.dampings.append(np.median(damping))


    def plot_damping_wind(self):
        plt.figure(figsize=(5,5))
        plt.scatter(self.trace.frequencies,(self.ullensvang_temps))
        # plt.ylim(0,50)
        plt.ylabel('Ullensvang temp')
        plt.xlabel('frequency')
        plt.savefig('temp-freq.png')




def save_list_of_TracePostprocessors(path,trace_list):
    for obj in trace_list:
        mean_freq=round(np.mean(obj.trace.frequencies),3)
        tot_path=path+'/trace_'+str(mean_freq)+'.pkl'
        with open(tot_path, 'wb') as fout:
            pickle.dump(obj, fout)

def import_folder_of_TracePostprocessors(path):
    list=[]
    if not path[-1]=='/':
        path+='/'
    file_names=os.listdir(path)
    for name in file_names:
        TracePostprocessors=import_converted_ts(path,name)
        list.append(TracePostprocessors)
    return list



##################################################################################
###################################################################################

class PostTraceComparison():
    '''A class width method for doing comparison on a list of Postprocessor objects
    '''
    def __init__(self):
        self.Post_traces=[]
    def import_Post_traces_from_folder(self,folder_path):
        if not folder_path[-1]=='/':
            folder_path+='/'
        file_names=os.listdir(folder_path)
        for name in file_names:
            TracePostprocessors=import_converted_ts(folder_path,name)
            self.Post_traces.append(TracePostprocessors)

    def print_trace_check(self):
        for p_trace in self.Post_traces:
            trace_obj=p_trace.trace
            # # print(trace.)
            



    def pair_data(self,ix1,ix2):
        '''Function to extract only the data from the segments included 
        in both traces
        arguments: 
            ix1: int
                index for the first post_trace object
            ix2: int
        returns: 
            dictionary
                {'damping':(damping for ix1,damping for ix2)
                 'dates': list of common dates,
                 'frequencies: (for ix1, for ix2)
                 'ullensvang_temps':(for ix1,forix2)}'''
        return_dict={}
        p_trace1=self.Post_traces[ix1]
        p_trace2=self.Post_traces[ix2]
        common_names=[]
        for i,name in enumerate(p_trace1.file_names):
            if name in (p_trace2.file_names): 
                common_names.append(name)
        ixs_1=[]
        ixs_2=[]

        for ix,i, in enumerate(p_trace1.file_names):
            if i in common_names:
                if not i in p_trace1.file_names[:ix]:
                    ixs_1.append(int(ix))

        for ix,j, in enumerate(p_trace2.file_names):
            if j in common_names:
                if not j in p_trace2.file_names[:ix]:
                    ixs_2.append(int(ix))

        return_dict['file_names']=common_names
        return_dict['damping']=(np.asarray(p_trace1.dampings)[ixs_1],
                                np.asarray(p_trace2.dampings)[ixs_2])
        
        return_dict['horisontal wind']=(np.asarray(p_trace1.horisontal)[ixs_1],
                                np.asarray(p_trace2.horisontal)[ixs_2])
        
        return_dict['frequencies']=(np.asarray(p_trace1.frequencies)[ixs_1],
                                np.asarray(p_trace2.frequencies)[ixs_2])
        
        return_dict['ullensvang_temps']=(np.asarray(
            p_trace1.ullensvang_temps)[ixs_1],
                                np.asarray(p_trace2.ullensvang_temps)[ixs_2])

        return return_dict
        
        
        
    def plot_all_traces(self,*args,interactive=False,fname='Traces.png',n=15,raw_names=None,raw_freqs=None,raw=False,title='',ax=None,legend=False):

        self.file_names=[]
        #finding all file names
        for pt in self.Post_traces:
            self.file_names+=pt.file_names
        self.file_names=np.unique(self.file_names)

        if interactive: 
            matplotlib.use('tkagg')
        else: 
            matplotlib.use('module://matplotlib_inline.backend_inline')
        fig=plt.figure(figsize=(7,4))
        
        handles=[]
        labels=[] 

        #converting to datetime
        if raw:
            plot_names=raw_names
        else: 
            plot_names=self.file_names
        dates=[]
        for file_name in plot_names:
            date=file_name[7:17]
            time=file_name[18:23].replace('-',':')
            dates.append(date+'T'+time) #YYYY-MM-DD_hh-mm
        self.dates=np.asarray(dates,dtype='datetime64[m]')
        #creating label array for date axis

        out=[]
        # print((self.dates))
        date_ixs=range(0,int(len(self.dates)),
                           int(len(self.file_names)//n))
        print(date_ixs)
        for i in range(len(self.dates)):
            if i in date_ixs:
                out.append(np.datetime64(self.dates[i],'M'))
        major_x,labels= np.arange(len(self.dates))[date_ixs],out
        if ax==None:
            freq_ax=fig.add_subplot()
        else:
            freq_ax=ax

        freq_ax.tick_params(axis='x', labelrotation=35)

        freq_ax.set_xticks(np.arange(len(self.dates)),minor=True)
        # freq_ax.set_xticklabels(dates_ax)
        freq_ax.set_xticks(major_x,labels=labels)
        freq_ax.set_xlabel('Date')
        freq_ax.minorticks_off()
        # freq_ax.

        if raw: 
            ixs=[]
            freqs_flat=[]
            for i,freqs in enumerate(raw_freqs):
                ixs+=[i]*len(freqs)
                freqs_flat+=list(freqs)


            freq_ax.scatter(ixs,freqs_flat,s=0.2,c='grey')

        else:          
            labels=[]
            i=1
            colors=[3]*30
            for post_trace in self.Post_traces:
                if len(post_trace.dates)==0:
                    post_trace.set_dates()
                this_dates=post_trace.dates
                this_ixs2=[]

                for ix,seg in enumerate(this_dates):
                    ixx=np.where(np.asarray(self.dates)==seg)
                    this_ixs2.append(ixx)
                    
                
                col,colors=color_from_freq(colors,np.mean(post_trace.frequencies))
                this_ixs=np.where(np.isin(self.dates,this_dates))[0]
                sc=freq_ax.scatter(this_ixs2,post_trace.frequencies,s=1.5,color=col,label=i,edgecolor=(0,0,0,0.4),linewidths=0.05)


                freq=np.median(post_trace.frequencies)
                p_freq=f'{freq:.2}'
                handles.append(sc)
                labels.append('Mode '+f'{str(i):<4}'+'~'+f'{str(p_freq):<4}'+' Hz')


                offset=100+50* (i%2)
              
                # freq_ax.annotate(str(i),(this_ixs2[0][0][0],post_trace.frequencies[0]),
                #                  (this_ixs2[0][0][0]*0-offset,post_trace.frequencies[0]),
                #                  fontsize=7,va='center')


                i+=1

        freq_ax.set_ylabel('Frequency [Hz]')
        freq_ax.grid()
        freq_ax.set_ylim(0,1)
        freq_ax.set_title(title)
        if legend:
            fig.legend(handles=handles,labels=labels,ncols=4,loc='upper center', bbox_to_anchor=(0.5, -0.00),
          fancybox=True,  ncol=4,markerscale=6,title='Numbering of modes')

        # freq_ax.legend(loc='right')
        if ax==None:
            if interactive:
                fig.show()
            else: 
                fig.subplots_adjust(0.1,0.2,0.9,0.9)
                plt.savefig(fname=fname,dpi=500,bbox_inches='tight')

                return fig
        else: 
            return ax
####################################################################
    def plot_all_traces_2(self,*args,interactive=False,fname='Traces.png',n=15,raw_names=None,raw_freqs=None,raw=False,title='',ax=None,legend=False,initial=False,init_freqs=None):

        self.file_names=[]
        #finding all file names
        for pt in self.Post_traces:
            self.file_names+=pt.file_names
        self.file_names=np.unique(self.file_names)

        if interactive: 
            matplotlib.use('tkagg')
        else: 
            matplotlib.use('module://matplotlib_inline.backend_inline')
        fig=plt.figure(figsize=(8,4.5))
        
        handles=[]
        labels=[] 

        #converting to datetime
        if raw:
            plot_names=raw_names
        else: 
            plot_names=self.file_names
        dates=[]
        for file_name in plot_names:
            date=file_name[7:17]
            time=file_name[18:23].replace('-',':')
            dates.append(date+'T'+time) #YYYY-MM-DD_hh-mm
        self.dates=np.asarray(dates,dtype='datetime64[m]')
        #creating label array for date axis

        out=[]
        # print((self.dates))
        date_ixs=range(0,int(len(self.dates)),
                           int(len(self.file_names)//n))
        print(date_ixs)
        for i in range(len(self.dates)):
            if i in date_ixs:
                out.append(np.datetime64(self.dates[i],'M'))
        major_x,labels= np.arange(len(self.dates))[date_ixs],out
        if ax==None:
            if initial==False: 
                freq_ax=fig.add_subplot()
            elif initial=='suggested':
                freq_ax=plt.subplot2grid((1,30),(0,1),rowspan=1,
                                         colspan=29,fig=fig)
                freq_ax.set_xlim(-15)
                init_ax=plt.subplot2grid((1,30),(0,0),rowspan=1,
                                         colspan=1,fig=fig)
                
        else:
            freq_ax=ax

        freq_ax.tick_params(axis='x', labelrotation=35)

        freq_ax.set_xticks(np.arange(len(self.dates)),minor=True)
        # freq_ax.set_xticklabels(dates_ax)
        freq_ax.set_xticks(major_x,labels=labels)
        freq_ax.set_xlabel('Date')
        freq_ax.minorticks_off()
        # freq_ax.

        if raw: 
            ixs=[]
            freqs_flat=[]
            for i,freqs in enumerate(raw_freqs):
                ixs+=[i]*len(freqs)
                freqs_flat+=list(freqs)


            freq_ax.scatter(ixs,freqs_flat,s=0.5,c='grey',edgecolor=(0,0,0,0.4),linewidths=0.05)

        else:          
            labels=[]
            i=1
            colors=[3]*30
            for post_trace in self.Post_traces:
                if len(post_trace.dates)==0:
                    post_trace.set_dates()
                this_dates=post_trace.dates
                this_ixs2=[]

                for ix,seg in enumerate(this_dates):
                    ixx=np.where(np.asarray(self.dates)==seg)
                    this_ixs2.append(ixx)
                    
                
                col,colors=color_from_freq(colors,np.mean(post_trace.frequencies)) #use for others
                # col,colors=color_from_freq(colors,(post_trace.frequencies[0])) # use for favarelli
                this_ixs=np.where(np.isin(self.dates,this_dates))[0]
                sc=freq_ax.scatter(this_ixs2,post_trace.frequencies,s=1.5,color=col,label=i,edgecolor=(0,0,0,0.4),linewidths=0.05)


                freq=np.median(post_trace.frequencies)
                p_freq=f'{freq:.2}'
                handles.append(sc)
                labels.append('Mode '+f'{str(i):<4}'+'~'+f'{str(p_freq):<4}'+' Hz')


                offset=100+50* (i%2)
              
                # freq_ax.annotate(str(i),(this_ixs2[0][0][0],post_trace.frequencies[0]),
                #                  (this_ixs2[0][0][0]*0-offset,post_trace.frequencies[0]),
                #                  fontsize=7,va='center')


                i+=1

        
        if initial == 'suggested':
            
            init_ax.tick_params(axis='x', labelrotation=35,length=0)
            means=[]
            max_len=0
            for trace in init_freqs:
                means.append(np.mean(trace))
                if len(trace)> max_len:
                    max_len=len(trace)
            
            
            colors=[3]*30
            for i,trace in enumerate(init_freqs):
                col,colors=color_from_freq(
                        colors,np.mean(trace))
                n=max_len//len(trace)
                init_ax.scatter(np.arange(0,len(trace),n),trace,color=col,
                                s=8,edgecolor=(0,0,0,0.6),linewidths=0.2)
            
            


            init_ax.set_xticks([0])
            init_ax.set_xticklabels(['Initial'],ha='right')
            # init_ax.set_xticks(['Initial'])



        freq_ax.grid()
        freq_ax.set_ylim(0,1)
        if initial != False:
            # init_ax.set_yticklabels(freq_ax.get_yticklabels())
            freq_ax.set_yticklabels([])
            init_ax.set_ylim(0,1)
            init_ax.set_ylabel('Frequency [Hz]')
            freq_ax.tick_params(axis='y',length=0)


        else:
            freq_ax.set_ylabel('Frequency [Hz]')
        freq_ax.set_title(title)
        if legend:
            fig.legend(handles=handles,labels=labels,ncols=4,loc='upper center', bbox_to_anchor=(0.5, 0.05),
          fancybox=True,  ncol=4,markerscale=6,title='Numbering of modes')

        # freq_ax.legend(loc='right')
        if ax==None:
            if interactive:
                fig.show()
            else: 
                fig.subplots_adjust(0.1,0.2,0.9,0.9,hspace=0,wspace=0)
                
                plt.savefig(fname=fname,dpi=1000,bbox_inches='tight')

                return fig
        else: 
            return ax



#######################################
    def comparison_plots(self,*args,ixs=[0,1,2,3,4,5,6,7]):

        figure=plt.figure(figsize=(7,7))

        for b_row in ixs:
            for b_col in ixs: 
                data_dict=self.pair_data(b_row,b_col)
                ax=plt.subplot2grid((len(ixs),len(ixs)),
                                        (b_row,b_col),rowspan=1,colspan=1,
                                        fig=figure)
                if b_row < b_col: #above diagonal 
                    try:
                        ax.scatter(data_dict['ullensvang_temps'][0],
                               data_dict['ullensvang_temps'][1],s=0.5)
                    except ValueError:
                        ax.text(0,0,'Value error')
                    pass
                elif b_row > b_col: #below diagonal

                    try:
                        ax.scatter(data_dict['frequencies'][0],
                               data_dict['frequencies'][1],s=0.5)
                    except ValueError:
                        ax.text(0,0,'Value error')
                else: #diagonal
                    pass
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()
                # else: #diagonal'

    def gen_mode_plots(self,stop_at=20):
        for i, ptrace in enumerate(self.Post_traces):
            self.gen_single_mode_plot(i)

        
    def gen_single_mode_plot(self,mode_ix):
        ptrace=self.Post_traces[mode_ix]
        phi1_avg=np.median(np.asarray(ptrace.mode_shapes),axis=0)
        freq_avg=np.median(ptrace.frequencies)
        damp_avg=np.median(ptrace.damping)
        n_ids=len(ptrace.damping)
        color,_=color_from_freq([3]*30,freq_avg)
        mode1_vis=mode_plot(phi1_avg)
        mode1_vis.report_figure(mode_ix+1,freq_avg,damp_avg,n_ids,col=color)
        
    def plot_compariosons_tracevice(self,mode_ixs):
        figsize=(5,1.5*len(mode_ixs))
        # fig,axs=plt.subplots(len(mode_ixs),5,sharex='col',figsize=figsize)
        fig = plt.figure(figsize=figsize)
        c_list=[]
        for i in range(len(mode_ixs)):
            s=3
            try: 
                P_trace=self.Post_traces[mode_ixs[i]]
                # t_ax=plt.subplot2grid((len(mode_ixs),27),loc=(i,0),rowspan=1,colspan=5)
                # # t_ax.text(0,0.5,'Trace '+str(mode_ixs[i]+1),ha='left',va='center',fontsize=12)
                # # r_ax[0
                # t_ax.set_axis_off()
                
                
                
                # wind - freq
                wf_ax=plt.subplot2grid((len(mode_ixs),22),
                                       loc=(i,0),rowspan=1,colspan=5)
                wf_ax.scatter(P_trace.horisontal,P_trace.freq_comparison,s=s,
                              alpha=1,edgecolor=(0,0,0,0.1),linewidths=s/20)

                if mode_ixs[i]==mode_ixs[-1]:
                    wf_ax.set_xlabel('Mean wind')
                elif mode_ixs[i]==mode_ixs[0]:
                        wf_ax.set_title('Mean wind')
                wf_ax.set_ylabel('Freq.\n[Hz]',labelpad=-20)
                ymi=wf_ax.get_ylim()[0]
                yma=wf_ax.get_ylim()[1]
                wf_ax.set_yticks([ymi,yma])
                wf_ax.set_yticklabels([f'{ymi:.2}',f'{yma:.2}'])

                c_list=plot_lin_reg(wf_ax,P_trace.horisontal,
                                    P_trace.freq_comparison,0.3,c_list)


            
                wd_ax=plt.subplot2grid((len(mode_ixs),22),
                                       loc=(i,5),rowspan=1,colspan=5)
                wd_ax.scatter(P_trace.temp,P_trace.freq_comparison,s=s
                              ,alpha=1,edgecolor=(0,0,0,0.1),linewidths=s/20)
                if mode_ixs[i]==mode_ixs[-1]:
                    wd_ax.set_xlabel('Temperature')
                elif mode_ixs[i]==mode_ixs[0]:
                    wd_ax.set_title('Temperature')
                wd_ax.set_yticks([])
                      
                c_list=plot_lin_reg(wd_ax,P_trace.temp,
                                    P_trace.freq_comparison,0.3,c_list)

                # # wind - damping 
                ft_ax=plt.subplot2grid((len(mode_ixs),22),
                                       loc=(i,12),rowspan=1,colspan=5)
                ft_ax.set_ylabel(r'$\zeta$'+' [%]',labelpad=-20)
                ft_ax.scatter(P_trace.horisontal,P_trace.damping,s=s,
                              alpha=1,edgecolor=(0,0,0,0.1),linewidths=s/20)
                ft_ax.set_ylim(0)
                if mode_ixs[i]==mode_ixs[-1]:
                    ft_ax.set_xlabel('Mean wind')
                elif mode_ixs[i]==mode_ixs[0]:
                    ft_ax.set_title('Mean wind')
                ymi=ft_ax.get_ylim()[0]
                yma=ft_ax.get_ylim()[1]
                ft_ax.set_yticks([0,yma])
                YMI=ymi*100
                YMA=yma*100
                ft_ax.set_yticklabels([0,f'{YMA:.3}'])
                c_list=plot_lin_reg(ft_ax,P_trace.horisontal,
                                    P_trace.damping,0.3,c_list)
                # T - damping 
                
                td_ax=plt.subplot2grid((len(mode_ixs),22),
                                       loc=(i,17),rowspan=1,colspan=5)

                td_ax.set_yticklabels([])
                td_ax.scatter(P_trace.temp,P_trace.damping,s=s,
                              alpha=1,edgecolor=(0,0,0,0.1),linewidths=s/20)
                td_ax.set_ylim(0)
                td_ax.set_yticks([])
                if mode_ixs[i]==mode_ixs[-1]:
                    td_ax.set_xlabel('Temperature')

                elif mode_ixs[i]==mode_ixs[0]:
                    td_ax.set_title('Temperature')
                c_list=plot_lin_reg(td_ax,P_trace.temp,
                                    P_trace.damping,0.3,c_list)
                # # freq - angle'
                # r_ax[4].scatter(P_trace.angle,P_trace.freq_comparison,s=s)
                
                # # damping - angle    
                # r_ax[5].scatter(P_trace.angle,P_trace.damping,s=s)

                
                
            except IndexError:    
                pass
                # for ax in r_ax:
                #     ydata=ax.get_yda
                #     ax.set_ylim()
               
                    
                #     x_data=P_trace.__getattribute__(x_name)
                #     y_data=P_trace.__getattribute__(y_name)

                #     ax.scatter(x_data,y_data,s=0.4,alpha=0.5)
                #     df=pd.DataFrame([x_data,y_data]).T
                #     corrcoef=df.corr()[0][1]
                #     # ax.set_ylim(0,0.08)
                #     xl=ax.get_xlim()
                #     yl=ax.get_ylim()
                #     x=xl[1]-(xl[1]-xl[0])*0.1
                #     y=yl[1]-(yl[1]-yl[0])*0.1
                #     co=f'{corrcoef:.2}'
                #     ax.text(x,y,str(co),ha='right',va='top',fontsize=8)  
                    
                
                #     pass



        fig.subplots_adjust(left=0,wspace=0.1,hspace=0.4)


        #   figsize=figsize)
        return fig,c_list
        


def plot_lin_reg(ax,X,Y,requirment,c_list):
    df=pd.DataFrame([X,Y]).T
    corrcoef=df.corr()[0][1]
    if np.abs(corrcoef)>requirment:
        X=np.asanyarray(X)
        A= np.vstack([X, np.ones(len(X))]).T
        m,c=np.linalg.lstsq(A,Y,rcond=None)[0]
        ax.plot(X,m*X+c,color='r')
        
        co=f'{corrcoef:.2}'
        x=ax.get_xlim()[1]-(ax.get_xlim()[1]-ax.get_xlim()[0])*0.05
        y=ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.05
        ax.text(x,y,r'$\rho$'+' = '+str(co),ha='right',va='top',fontsize=10)
        c_list.append(corrcoef)
    else: 
        c_list.append(0)
    return c_list

        

# def color_from_freq(freqs,freq):
#     if freqs==[3]*30:
        
#         freqs= [0.051440629250355946,
# 0.10482722230378724,
# 0.11807047942963439,
# 0.14252718085503183,
# 0.18360394583291736,
# 0.20607280204180065,
# 0.2131370191264357,
# 0.27659150315676984,
# 0.3183742890207623,
# 0.33352644142966603,
# 0.3698342678207916,
# 0.4005930355156324,
# 0.40598810432885546,
# 0.4167164128499862,
# 0.4646207325855476,
# 0.47057342801252056,
# 0.5142581248305934,
# 0.5273517159685219,
# 0.546193977572144,
# 0.5508938441295251,
# 0.5838064610192906,
# 0.5977192674857554,
# 0.6256467024923776,
# 0.7121826238745573,
# 0.8029923734122486,
# 0.8169919318758947,
# 0.8999146751028894,
# 0.9960805506555608]
#     ix=np.argmin(np.abs(freqs-freq))
#     # if freq > 0.78:
#     # freqs[ix]=3
#     colors=sns.color_palette('deep',60)
#     return colors[ix],freqs
        
def color_from_freq(used,freq):
    if used==[3]*30:
        used=[]
    freqs= [0.051440629250355946,
    0.10482722230378724,
    0.11807047942963439,
    0.14252718085503183,
    0.18360394583291736,
    0.20607280204180065,
    0.2131370191264357,
    0.27659150315676984,
    0.3183742890207623,
    0.33352644142966603,
    0.3698342678207916,
    0.4005930355156324,
    0.40598810432885546,
    0.4167164128499862,
    0.4646207325855476,
    0.47057342801252056,
    0.5142581248305934,
    0.5273517159685219,
    0.546193977572144,
    0.5508938441295251,
    0.5838064610192906,
    0.5977192674857554,
    0.6256467024923776,
    0.7121826238745573,
    0.8029923734122486,
    0.8169919318758947,
    0.8999146751028894,
    0.9960805506555608]
    ix=np.argmin(np.abs(freqs-freq))
    c=used.count(ix)
    used.append(ix)
    ix+=3*c
    
    colors=sns.color_palette('deep',60+2*ix)
    return colors[ix],used