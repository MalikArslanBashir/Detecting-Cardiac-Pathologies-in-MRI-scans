import os
import sys
from train             import Training
from data_io           import load_data
from parse_config      import parse_config

class MainClass():
    def __init__(self, config_file):        
        config                          = parse_config(config_file)
        self.config_data                = config['Data']
        self.train_path                 = self.config_data.get('train_directory')
        self.val_path                   = self.config_data.get('val_directory')
        self.read_pickle                = self.config_data.get('read_pkl')
        self.save_pickle                = self.config_data.get('save_pkl')

##########Main#######################################
if __name__== '__main__':    

    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print(' python main.py config.txt')
        exit()

    config_file     = str(sys.argv[1])
    
    assert(os.path.isfile(config_file))
    
    Obj             = MainClass(config_file)

    # Reading subjects
    print("................Loading dataset.....................")
    train_images, train_labels, val_images, val_labels = load_data(Obj.train_path,
                                                                   Obj.val_path,
                                                                   Obj.read_pickle, 
                                                                   Obj.save_pickle)

    # Printing data info
    print('----------------------------------Data summary---------------------------------------:')
    print(' ---------------------------Training and Validation Images---------------------------:')
    print('----------------------', train_images.shape, val_images.shape ,'---------------------:')
    print('----------------------------Type:', train_images.dtype, '----------------------------:')
    print(' -------------------------Training and Validation Labels:----------------------------:')
    print('-----------------------', train_labels.shape, val_labels.shape ,'--------------------:')
    print('----------------------------Type:', train_labels.dtype ,'----------------------------:')
    # Training
    TrainSession    = Training(config_file)
    TrainSession(train_images, train_labels, val_images, val_labels)
    
###############################################################################
