import torch
import torch.utils.data as data
import os
import numpy as np
from natsort import natsorted

class DatasetFromFolderPy(data.Dataset):
    def __init__(self, dir_input_cross, dir_target_cross, train_dir, direction = 'BtoA',transform=None, transform_target = None, input_size=None, resize_scale=None, crop_size=None, fliplr = False, flipud = False):
        super(DatasetFromFolderPy, self).__init__()
        self.path_input_cross = os.path.join(dir_input_cross)
        self.path_target_cross = os.path.join(dir_target_cross)
        
        #self.damper = damper
                
        self.idx = np.load(train_dir)
        self.filenames_input_cross = [x for x in natsorted(os.listdir(self.path_input_cross))]
        self.filenames_target_cross = [x for x in natsorted(os.listdir(self.path_target_cross))]
             
        self.set_input_cross = np.array(self.filenames_input_cross)[self.idx]
        self.set_target_cross = np.array(self.filenames_target_cross)[self.idx]
        
        set = []
        for k in range(len(self.set_input_cross)):
            set.append(self.path_input_cross+ self.set_input_cross[k] )
        self.set_input_cross = set
        
        set = []
        for k in range(len(self.set_target_cross)):
            set.append(self.path_target_cross+ self.set_target_cross[k])
        self.set_target_cross = set
   
        self.direction = direction
        self.transform=transform
        self.transform_target = transform_target
        self.resize_scale=resize_scale
        self.crop_size=crop_size
        self.fliplr = fliplr
        self.flipud = flipud
        self.input_size = input_size
        
    def __getitem__(self, index):
        fn_input_cross = os.path.join(self.set_input_cross[index])
        input_cross = np.load(fn_input_cross)
        
        fn_target_cross = os.path.join(self.set_target_cross[index])
        target_cross = np.load(fn_target_cross)
        
        if self.direction == 'AtoB':
            input_cross = input_cross
            target_cross = target_cross
        elif self.direction == 'BtoA':
            input_cross = target_cross
            target_cross = input_cross
            
        input_cross_real = input_cross.real
        input_cross_imag = input_cross.imag
        
        target_cross_real = target_cross.real
        target_cross_imag = target_cross.imag
       
        
        
        if self.transform is not None:
            input_cross_real = self.transform(input_cross_real.copy())
            input_cross_imag = self.transform(input_cross_imag.copy())
            input_cross_real = input_cross_real.type(torch.cuda.FloatTensor)
            input_cross_imag = input_cross_imag.type(torch.cuda.FloatTensor)
            
            target_cross_real = self.transform_target(target_cross_real.copy())
            target_cross_imag = self.transform_target(target_cross_imag.copy())
            target_cross_real = target_cross_real.type(torch.cuda.FloatTensor)
            target_cross_imag = target_cross_imag.type(torch.cuda.FloatTensor)
            
        input_data = torch.cat((input_cross_real,input_cross_imag))
        target_data = torch.cat((target_cross_real,target_cross_imag))
        return input_data, target_data
    
    def __len__(self):
        return len(self.set_input_cross)