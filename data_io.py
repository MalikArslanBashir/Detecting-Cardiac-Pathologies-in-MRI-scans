import os
import pickle                                             as pkl
import numpy                                              as np
import nibabel                                            as nib

def read_nii_files(path, subject):    

    newpath    = (path     + subject + '/')

    ED         = (newpath  + subject + '_ED.nii.gz')
    ES         = (newpath  + subject + '_ES.nii.gz')
    EDlabel    = (newpath  + subject + '_ED_gt.nii.gz')
    ESlabel    = (newpath  + subject + '_ES_gt.nii.gz')

    imED       = nib.load(ED).get_data().astype(np.float32)
    labelED    = nib.load(EDlabel).get_data().astype(np.uint8)

    imES       = nib.load(ES).get_data().astype(np.float32)
    labelES    = nib.load(ESlabel).get_data().astype(np.uint8)

    return imED, imES, labelED, labelES

############################Loading dataset#########################################
def loading_data(dataset_path, read_pkl, save_pkl):
    
    path            = dataset_path  
    
    if read_pkl:        
        assert(os.path.isfile(os.path.join(path, "images_volume.pkl")))
        assert(os.path.isfile(os.path.join(path, "labels_volume.pkl")))
        
        print('pickle files exists')
        
        volumes     = pkl.load(open(os.path.join(path, "images_volume.pkl"), "rb"))
        labels      = pkl.load(open(os.path.join(path, "labels_volume.pkl"), "rb"))

    else:
        subjects    = os.listdir(path)
        volumes     = []
        labels      = []
        ED_volumes  = []
        ES_volumes  = []
        ED_labels   = []
        ES_labels   = []
        
        for subject in subjects:
            
            print('Reading', subject, '\n')
            image_ED, image_ES, label_ED, label_ES = read_nii_files(path, subject)

            ED_volumes   += [image_ED]
            ED_labels    += [label_ED]
            ES_volumes   += [image_ES]
            ES_labels    += [label_ES]
            volumes      += [image_ED]
            volumes      += [image_ES]
            labels       += [label_ED]
            labels       += [label_ES]

        ED_volumes  = np.concatenate(ED_volumes , axis=2)
        ED_labels   = np.concatenate(ED_labels  , axis=2)
        ES_volumes  = np.concatenate(ES_volumes , axis=2)
        ES_labels   = np.concatenate(ES_labels  , axis=2)
        volumes     = np.concatenate(volumes    , axis=2)
        labels      = np.concatenate(labels     , axis=2)
                
        if save_pkl:            
            pkl.dump(ED_volumes   , open(path + "ED_images_volume.pkl"    , "wb"))
            pkl.dump(ED_labels    , open(path + "ED_labels_volume.pkl"    , "wb"))
            pkl.dump(ES_volumes   , open(path + "ES_images_volume.pkl"    , "wb"))
            pkl.dump(ES_labels    , open(path + "ES_labels_volume.pkl"    , "wb"))
            pkl.dump(volumes      , open(path + "images_volume.pkl"       , "wb"))
            pkl.dump(labels       , open(path + "labels_volume.pkl"       , "wb"))
		
        '''
		#visualization of volumes
		for k in range(volumes.shape[2]):
			#print(volumes[:,:,k].dtype)
			plt.imshow(volumes[:,:,k])
			plt.show()
			#print(labels[:,:,k].dtype)
			plt.imshow(labels[:,:,k])
			plt.show()

        print(np.unique(labels))
        '''
    return volumes, labels

def load_data(train_dataset_path, val_dataset_path, read_pkl, save_pkl):

	train_images, train_labels = loading_data(train_dataset_path, read_pkl, save_pkl)
	val_images, val_labels     = loading_data(val_dataset_path, read_pkl, save_pkl)

	return train_images, train_labels, val_images, val_labels

###############################################################################
