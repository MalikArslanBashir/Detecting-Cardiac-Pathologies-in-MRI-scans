import math
import numpy                            as np
from utils                              import crop_or_pad_image_to_size, crop_or_pad_slice_to_size, MVN_np
from distortion                         import Distortion
from data_io                            import load_data
try:
  from tensorflow.python.keras.utils    import to_categorical, Sequence
except:
  from keras.utils                      import to_categorical, Sequence

# Batch Generator:
class DataGenerator(Sequence):
    def __init__(self, images, labels, crop_size, num_classes, num_channels, config, mode):
        
        # Preparing random generator
        # np.random.seed(self.config_train.get('random_seed'))
        
        # Fetching parameters
        self.config_train               = config['Train']
        self.images                     = images
        self.labels                     = labels
        self.num_classes                = num_classes
        self.num_channels               = num_channels
        self.crop_size                  = crop_size

        # Augmentation parameters
        if (mode == 'train'):
            self.batch_size                 = self.config_train.get('train_batch_size')
            self.augmentation_flag          = self.config_train.get('augmentation_train')
            self.augmentations_accumulate   = self.config_train.get('augments_train_accumulate')

        elif (mode == 'val'):
            self.batch_size                 = self.config_train.get('val_batch_size')
            self.augmentation_flag          = self.config_train.get('augmentation_val')
            self.augmentations_accumulate   = self.config_train.get('augments_val_accumulate')
            
        else:
            raise Exception('Mode is specified incorrectly')
        


        if self.augmentation_flag:
            self.augmentations_to_do    = self.config_train.get('augmentations_to_do')
            self.augmentation_class     = Distortion({
                                                        'augmentations_to_do':self.augmentations_to_do,
                                                        'augmentations_accumulate':self.augmentations_accumulate
                                                     }, data_channels=self.num_channels)
        
        self.MVN_flag                       = self.config_train.get('mvn_flag')
        self.shuffle                        = self.config_train.get('shuffle')
        self.center_crop_flag               = self.config_train.get('center_crop_flag')
        
        self.images                         = np.asarray(self.images)
        self.indexes                        = np.arange(self.images.shape[2])
        self.batches                        = math.ceil(len(self.indexes) / self.batch_size)


        # First shuffling of indexes
        np.random.shuffle(self.indexes)

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.batches

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Fetch slices for current batch
        # print('Batch IDs:',list_IDs_temp)
        list_IDs_temp   = self.indexes[index*self.batch_size : min( (index + 1)*self.batch_size, len(self.indexes) )]

        # Generate data
        X, y            = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        
        images_train      = []
        labels_train      = []

        for n in list_IDs_temp:
            image         = self.images[:,:,n]
            label         = self.labels[:,:,n]
            
            if (self.augmentation_flag):
                image                   = np.expand_dims(image, axis=-1)
                label                   = np.expand_dims(label, axis=-1)
                
                image, label            = self.augmentation_class.apply_distortion(image, label)
                
                image_slice             = []
                label_slice             = []
                
                for j in range(len(image)):
                    image_slice         = image[j]
                    label_slice         = label[j]
                              
                    if (self.center_crop_flag):
                        image_slice     = crop_or_pad_slice_to_size(image_slice, self.crop_size[0], self.crop_size[1])
                        label_slice     = crop_or_pad_slice_to_size(label_slice, self.crop_size[0], self.crop_size[1])

                    if (self.MVN_flag):
                        image_slice     = MVN_np(image_slice)
                        
                    images_train.append(image_slice)
                    labels_train.append(label_slice)
                
                #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',len(images_train),len(labels_train))
            else:
                image                   = np.expand_dims(image, axis=-1)
                label                   = np.expand_dims(label, axis=-1)

                if (self.center_crop_flag):
                    image           = crop_or_pad_slice_to_size(image, self.crop_size[0], self.crop_size[1])
                    label           = crop_or_pad_slice_to_size(label, self.crop_size[0], self.crop_size[1])
                if (self.MVN_flag):                
                    image           = MVN_np(image)                 
                    
                images_train.append(image)
                labels_train.append(label)

        # Making the selected batch of slices as np array
        images_train = np.stack(images_train, axis=0)
        labels_train = np.stack(labels_train, axis=0)

        # Adding an extra channel dimension required by Keras
        #images_train = np.expand_dims(images_train, axis=-1)
        labels_train2= []
  
        labels_train2 = crop_or_pad_image_to_size(labels_train,128,128)
        labels_train2 = np.array(labels_train2)

        labels_train3= []
  
        #labels_train3 = crop_or_pad_image_to_size(labels_train,64,64)
        #labels_train3 = np.array(labels_train3)


        labels_train = to_categorical(labels_train, self.num_classes)
        labels_train2 = to_categorical(labels_train2, self.num_classes)
        #labels_train3 = to_categorical(labels_train3, self.num_classes)

        return images_train, [labels_train,labels_train2]

"""################Test Routine#################"""
import sys
from parse_config  import parse_config

class MainClass():
    def __init__(self, config_file):
        self.config                     = parse_config(config_file)
        self.config_data                = self.config['Data']
        self.config_train               = self.config['Train']
        self.train_data_path            = self.config_data.get('train_val_data_path')
        self.num_classes                = self.config_train.get('num_classes')
        self.num_channels               = self.config_train.get('num_channels')
        self.crop_size                  = self.config_train.get('crop_size')


if __name__=='__main__':
    
    config_file                                    = str(sys.argv[1])
    parsed_config                                  = parse_config(config_file)
    obj                                            = MainClass(config_file)
    volume_train,label_train,volume_val,label_val  = load_data(obj.train_data_path,False,False)
    train_generator                                = DataGenerator(volume_train,
                                                                   label_train,
                                                                   obj.crop_size,
                                                                   obj.num_classes,
                                                                   obj.config,'train')
    for i in range(len(train_generator)):
        x, y = train_generator[i]
        print('ITERATION:', i, x.shape, y.shape)

###############################################################################
