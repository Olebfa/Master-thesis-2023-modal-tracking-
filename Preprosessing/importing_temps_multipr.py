from concurrent.futures import ProcessPoolExecutor
import os
import sys
path='../src/'
if path not in sys.path: 
    sys.path.insert(1,path)
from preposessing import import_file
import pickle


folder_path='D:/ProcessedData_Hardanger_mat/'
names=os.listdir(folder_path)

def middle(array):
    return array[len(array)//3:2*len(array)//3]

def quality_check(list):
    for item in list: 
        print(item)
        if 'acceptable' in item[0]:
            pass
        elif 'good' in item:
            pass
        else: 
            return False
    return True 

def import_temp_series(name):
    path='../../A6_wind/'
    if name[:-4]+'.pkl' not in os.listdir(path):
        try:
            file=import_file(folder_path,name)
            A6=file.get_A6_wind()
            qual=A6['quality']
            hor=middle(A6['A6'][1])
            ver=middle(A6['A6'][2])
            angle=middle(A6['A6'][0])
            temp=middle(A6['A6'][3])
            
            if quality_check(qual):
                with open(path+name[:-4]+'.pkl','wb') as file:
                    pickle.dump((hor,ver,angle,temp),file)
        except:
            f=open('failed_wind.txt','a')
            f.write(name+'\n')
            f.close()


if __name__ =='__main__':
    print('main started')
    executor =ProcessPoolExecutor(max_workers=20)
    for result in executor.map(import_temp_series, names,chunksize=1):
        re=result