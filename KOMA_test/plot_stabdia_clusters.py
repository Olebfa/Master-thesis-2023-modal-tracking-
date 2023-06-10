import matplotlib.pyplot as plt
import numpy as np


def plot_stab_from_KOMA(sorted_freqs,sorted_orders,*args,**kwargs):

    """Returns a figure with the clusters color coded  
    and with lines in between the poles in one cluster
    input: 
    
    """
    if not len(args)>0: 
        lambd=sorted_freqs
    if not len(args)>0:
        all_orders=sorted_orders
    else: 
        lambd=args[0]
        all_orders=args[1]

    #make sure lambd and all_orders hvae the same shape:
    temp=np.empty_like(lambd)
    for i,ord in enumerate(lambd):
        temp[i]=ord*0+all_orders[i]
    all_orders=temp
    # Make a arrays of colors. 
    colors=[]
    num=len(sorted_freqs)
    root=int(num**(1/3))
    root_rest=int(num/root**2)+1
    colors=[]
    for r in range(root):
        for g in range(root):
            for b in range(root_rest): 
                colors.append([r/root,g/root,b/root_rest])
                
    #create figure and plot scatter
    if 'figsize' in kwargs.keys():
        fig_s=kwargs['figsize']
    else:
        fig_s=(9,9)
    figur_h=plt.figure(figsize=fig_s)
    axes=figur_h.add_subplot(1,1,1)

    
    # plotting all the discarded poles: 
    for i,order_i in enumerate(all_orders):
        lambd_i=lambd[i]         
        xi=-np.real(lambd_i)/np.abs(lambd_i)
        freq_i=np.imag(lambd_i)/(np.sqrt(1-xi**2))
        # freq_i=all_freqs
        col='black'
        if i==0:
            axes.scatter(freq_i/(2*np.pi),order_i,color=col,marker='x',
            s=30,label='Deleted poles')
        else:
            axes.scatter(freq_i/(2*np.pi),order_i,color=col,marker='x',s=30)

    #plotting the poles to keep, and the lines between them
    for i,order_i in enumerate(sorted_orders):
        freq_i=sorted_freqs[i]
        col=colors[i]
        size=50
        axes.scatter(freq_i/(2*np.pi),order_i,marker='s',
        color=col,s=size,label='Mode '+str(i+1))
        axes.plot(freq_i/(2*np.pi),order_i,color=col,linewidth=size/20)
       
    axes.legend()
    axes.set_ylabel('Modal order')
    axes.set_xlabel('Frequencies [Hz]')
    axes.set_xlim(0,)
    plt.tight_layout()   
    return figur_h    