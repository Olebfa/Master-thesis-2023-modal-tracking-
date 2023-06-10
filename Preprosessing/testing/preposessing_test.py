import os
import mat73  #install with pip install mat73
import numpy as np
import matplotlib.pyplot as plt
import pickle



class import_file(): 
    '''Class to import and copy wanted data to a new object and save it for further use. 
    Arguments: 
        folder_path: string
            path to find the folder whre the file import is located.
        file_name: string 
            name of the .mat file to import.'''
    def __init__(self,folder_path,file_name):
        self.folder_path=folder_path
        self.file_name=file_name
        self.path=os.path.join(folder_path,file_name)
        #importing the WHOLE file
        self.data = mat73.loadmat(self.path)

    def get_adjusted_data(self):
        '''
        Function to extract the adjusteted data from an
        imported file. 

        returns: 
            data: dict
                keys are sensor names. each key contains three time series.
                x, y and z accelerations 
        '''
        ad_data={}
        for component in self.data.get('recording').get('sensor'):
            key=component.get('sensor_name')
            if key[0]=='H': #skips the wind
                sens=component.get('component')
                data=sens.get('adjusted_data')
                out=np.array([data[0][0],data[1][0],data[2][0]])
                ad_data[key]=out
        return ad_data

    def get_raw_data(self):
        '''
        Function to extract the raw data from an
        imported file. 

        returns: 
            data: dict
                keys are sensor names. each key contains three time series.
                x, y and z accelerations 
        '''
        rw_data={}
        for component in self.data.get('recording').get('sensor'):
            key=component.get('sensor_name')
            if key[0]=='H': #skips the wind
                sens=component.get('component')
                data=sens.get('raw_data')
                out=np.array([data[0][0],data[1][0],data[2][0]])
                rw_data[key]=out
        return rw_data
    
    def get_component_atr(self,atr): 
        pass

    def convert(self):
        '''Function to create a new object from class
         'create_simple_ts' and assign wanted values to it.  
         
         returns: object, type 'create_simple_ts'
                This object can be further manipulated with functions 
                from its own class. 
         '''
        self.component_atr_to_transfer=[
        'component_no',
        'data_quality',
        'mean',
        ]
        new_ts=create_simple_ts()
        data_dict={}
        for sensor in self.data.get('recording').get('sensor'):
            sensor_name=sensor.get('sensor_name').replace(' ','_')
            if sensor_name[0]=='H': #skips the wind
                # transferring the data for each senor
                component=sensor.get('component')
                adjusted_data=component.get('adjusted_data')
                for i,dir in enumerate(['_x','_y','_z']):
                    comp_name=sensor_name+dir
                    data_dict[comp_name]=np.array(
                        adjusted_data[i][0])
                    
                    new_ts.sensor_metadata[comp_name]={}
                    # transfering sensor metadata
                    for atr in self.component_atr_to_transfer:
                        new_ts.sensor_metadata[
                            comp_name][atr]=component.get(atr)[i][0]
                    new_ts.sensor_metadata[comp_name][
                        'adjusted_samplerate']=sensor.get(
                            'adjusted_samplerate')
        length=len(adjusted_data[0][0])
        # setting the metadata for the whole series: 
        new_ts.series_metadata['file_name']=self.file_name[:-4]
        
        dur=self.data.get(
            'recording').get('general').get('duration')
        new_ts.set_atr('ac_data',data_dict)
        new_ts.series_metadata['duration']=dur
        new_ts.set_atr('timeseries',np.linspace(0,dur,length))

        return new_ts

################################################                
        
def import_converted_ts(path,name): 
    '''Function to import a .pkl object of class 'create_simple_ts'
    arguments: 
            path:  string 
                folder path
            name: string 
                filename. '''
    if name[-4:]=='.pkl':
        tot_path=path+name
    else: tot_path=path+name+'.pkl'
    with open(tot_path, "rb") as file:
        return pickle.load(file)


##########################################################

class create_simple_ts():
    '''Class for creating a new object from 
    data extracted with "import_file" class. '''
    def __init__(self):
        self.sensor_metadata={}
        self.series_metadata={}

    def set_general_metadata(self,key,value): 
        self.series_metadata[key]=value
        
    def get_from_general_metadata(self,key):
        if key in self.series_metadata.keys():
            return self.series_metadata.get(key)
        else: return None

    def set_atr(self,attribute_name,sensor_data):

        setattr(self,attribute_name,sensor_data)

    def save(self,path):
        '''save the object as .pkl-file at the dessignated 
        location. The name is picked from the object. 
        Arguments: 
            path: string'''
        tot_path=path+self.series_metadata['file_name']+'.pkl'
        with open(tot_path, 'wb') as fout:
            pickle.dump(self, fout)

#########################################

def quality_check(ts): 

    '''Function to check the quality of the imported series:
    arguments: 
        ts: object of class 'create_simple_ts'''
    good_quality=True
    for key in ts.sensor_metadata.keys(): 
        string=ts.sensor_metadata.get(key).get('data_quality')
        if 'poor' in string:
            good_quality=False
            return good_quality
        test2=ts.sensor_metadata.get(key).get('component_no')

        if not 'x' in key and test2==1:
            good_quality=False
            return good_quality
        if not 'y' in key and test2==2:
            good_quality=False
            return good_quality
        if not 'z' in key and test2==3:
            good_quality=False
            return good_quality
    if not len(ts.sensor_metadata.keys())==60:
        good_quality=False
    return good_quality

###################################################

def convert_folder(path_in,*args,path_out=False):
    '''Function to convert a whole folder of .mat -files to 
    "create_simple_ts" objects and save them as .pkl files
    arguments: 
        path_in: string 
            path of the folder with the .mat files. 
            can contain other folders, but not other file types
        
        *args: 
        path_out: string, optional
            Where to save the .pkl files. Creates a subfolder called 
            'converted_ts' in the input folder by default.  '''
    if not path_out:
        path_out=path_in+'/converted_ts/'
    try:
        os.mkdir(path_out)
    except FileExistsError:
        pass
    
    dir = os.listdir(path_in)
    for entry in dir:
        file_path=os.path.join(path_in,entry)
        if os.path.isfile(file_path):
            imported=import_file(path_in,entry)
            new_ts=imported.convert()
            if quality_check(new_ts):
                new_ts.save(path_out)

#######################################################
def get_ts_folder(path_in): 
    '''Function for importing all .pkl files from a folder and 
    store the objects in a list. '''
    time_series=[]
    dir = os.listdir(path_in)
    for entry in dir:
        file_path=os.path.join(path_in,entry)
        if os.path.isfile(file_path):
            ts=import_converted_ts(path_in,entry)
            time_series.append(ts)
    return time_series
            
#######################################################

def adjust_means(ts): 
    for key in ts.ac_data.keys():
        mean=ts.sensor_metadata.get(key).get('mean')
        ts.ac_data[key]=ts.ac_data.get(key)-mean
    return ts
