import numpy as np 

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




def transform(modes_dict):
    '''Function that transforms from STRID dictionary 
    of modes '''
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



