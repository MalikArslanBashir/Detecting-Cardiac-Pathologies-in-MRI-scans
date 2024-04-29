import os, sys, cv2
import pickle                                                               as pkl
import numpy                                                                as np
import nibabel                                                              as nib
import scipy.io                                                             as sio
import pandas                                                               as pd

from parse_config                        import parse_config
from test_preprocessing_utilities        import (RescaleImage, winsorizing, MVN_np, 
                                                 crop_or_pad_slice_to_size, change_labels)


def  cfg_read(file_name):
    
    myvars             = {}
    with open(file_name) as myfile:
        for line in myfile:
            name,var              = line.partition(": ")[::2]
            myvars[name.strip()]  = (var.strip()).zfill(2)
            
    return myvars

def Read_nii_files(data_path, subject):
    
    image_path          = data_path + '/' + subject
    dictionary          = cfg_read(os.path.join(image_path, 'Info.cfg'))
    
    # Reading End-Diastolic frame
    ED_frame            = nib.load(image_path + '/' + subject + '_frame%s'%dictionary['ED'] + '.nii.gz')
    ED_affine           = ED_frame.affine
    ED_header           = ED_frame.header
    ED_pix_size         = (ED_frame.header['pixdim'][1], 
                           ED_frame.header['pixdim'][2],
                           ED_frame.header['pixdim'][3])
    ED_frame            = ED_frame.get_data().astype(np.float32)
    
    ED_label            = nib.load(image_path + '/' + subject + '_frame%s'%dictionary['ED'] + '_gt.nii.gz')
    ED_label            = ED_label.get_data().astype(np.uint8)

    # Reading End-Systolic frame
    ES_frame            = nib.load(image_path + '/' + subject + '_frame%s'%dictionary['ES'] + '.nii.gz')
    ES_affine           = ES_frame.affine
    ES_header           = ES_frame.header
    ES_pix_size         = (ES_frame.header['pixdim'][1], 
                           ES_frame.header['pixdim'][2],
                           ES_frame.header['pixdim'][3])
    ES_frame            = ES_frame.get_data().astype(np.float32)
    
    ES_label            = nib.load(image_path + '/' + subject + '_frame%s'%dictionary['ES'] + '_gt.nii.gz')
    ES_label            = ES_label.get_data().astype(np.uint8)
    
    return (ED_frame, ES_frame, ED_label, ES_label, ED_affine, ES_affine, 
            ED_pix_size, ES_pix_size, ED_header, ES_header)


def save_in_pkl(path, ED_images, ED_labels, ES_images, ES_labels, images, labels):
    
    ED_images           = np.concatenate(ED_images, axis=2)
    ED_labels           = np.concatenate(ED_labels, axis=2)
    ES_images           = np.concatenate(ES_images, axis=2)
    ES_labels           = np.concatenate(ES_labels, axis=2)
    images              = np.concatenate(images, axis=2)
    labels              = np.concatenate(labels, axis=2)

    os.mkdir(path + '/Pkl')
    Pkl_path            = path + '/Pkl'

    pkl.dump(ED_images, open(Pkl_path + "/ED_images.pkl", "wb"))
    pkl.dump(ED_labels, open(Pkl_path + "/ED_labels.pkl", "wb"))
    pkl.dump(ES_images, open(Pkl_path + "/ES_images.pkl", "wb"))
    pkl.dump(ES_labels, open(Pkl_path + "/ES_labels.pkl", "wb"))

    pkl.dump(images, open(Pkl_path + "/images.pkl", "wb"))
    pkl.dump(labels, open(Pkl_path + "/labels.pkl", "wb"))

def save_in_mat(path, ED_images, ED_labels, ES_images, ES_labels, images, labels):
	
    ED_images           = np.concatenate(ED_images, axis=2)
    ED_labels           = np.concatenate(ED_labels, axis=2)
    ES_images           = np.concatenate(ES_images, axis=2)
    ES_labels           = np.concatenate(ES_labels, axis=2)
    images              = np.concatenate(images, axis=2)
    labels              = np.concatenate(labels, axis=2)
	
    os.mkdir(path + '/Mat')
    mat_path            = path + '/Mat'

    sio.savemat(os.path.join(mat_path, "ED_images.mat"), mdict={'ED_images':ED_images})
    sio.savemat(os.path.join(mat_path, "ED_labels.mat"), mdict={'ED_labels':ED_labels})
    sio.savemat(os.path.join(mat_path, "ES_images.mat"), mdict={'ES_images':ES_images})
    sio.savemat(os.path.join(mat_path, "ES_labels.mat"), mdict={'ES_labels':ES_labels})
 
    sio.savemat(os.path.join(mat_path, "Images.mat"), mdict={'images':images})
    sio.savemat(os.path.join(mat_path, "Labels.mat"), mdict={'labels':labels})

def Preprocess_data(config_file):
    
    config_Dataprep         = config["Dataprep"]
    data_path               = config_Dataprep.get("data_path")
    save_path               = config_Dataprep.get("processed_data_path")
    preprocessing_to_do     = config_Dataprep.get("preprocessing_to_do")
    
    save_pkl                = config_Dataprep.get("save_pkl_flag")
    save_mat                = config_Dataprep.get("save_mat_flag")
    save_nii                = config_Dataprep.get("save_nii_flag")
        
    images                  = []
    ED_images               = []
    ES_images               = []
    labels                  = []
    ED_labels               = []
    ES_labels               = []
    
    subjects                = os.listdir(data_path)
    
    for subject in subjects:
        
        steps               = ''
        print('Reading',subject,'\n')
        
        (ED_frame, ES_frame, 
         ED_label, ES_label, 
         ED_affine, ES_affine, 
         ED_pix_size, ES_pix_size,
         ED_header, ES_header)        = Read_nii_files(data_path, subject)

        
        print('Dimension of ED and ES frames before preprocessing of a patient:', ED_frame.shape, ES_frame.shape)

        if 'filtering' in preprocessing_to_do:
            # function smoothes an image using the median filter with 
            # the ksize x ksize aperture. Each channel of a multi-channel
            # image is processed independently. In-place operation is supported
            
            win_size            = config_Dataprep.get("filter_win_size")
            ED_frame            = cv2.medianBlur(ED_frame, win_size) 
            ES_frame            = cv2.medianBlur(ES_frame, win_size) 
									
            steps               = '_filt'

        if 'resampling' in preprocessing_to_do:            
            mode                = config_Dataprep.get("scaling_mod")
            target_resolution   = config_Dataprep.get("target_resolution")            
            interp_image        = config_Dataprep.get("interpolate_image")
            interp_label        = config_Dataprep.get("interpolate_label")
            
            ED_scale_vector     = [ED_pix_size[0] / target_resolution[0],
                                   ED_pix_size[1] / target_resolution[1],
                                   1]                                           # ED_pix_size[2] / target_resolution[2]
            
            ES_scale_vector     = [ES_pix_size[0] / target_resolution[0],
                                   ES_pix_size[1] / target_resolution[1],
                                   1]                                           # ES_pix_size[2] / target_resolution[2]
            
            ED_frame, ED_label  = RescaleImage(ED_frame, ED_label, ED_scale_vector,
                                               interp_image, interp_label, mode)
            
            ES_frame, ES_label  = RescaleImage(ES_frame, ES_label, ES_scale_vector,
                                               interp_image, interp_label, mode)

            print('Dimension of ED and ES frame after Resampling:', 'ED:', 
                  ED_frame.shape,'ES;', ES_frame.shape)
            steps              += '_resamp'

        if 'winsorizing' in preprocessing_to_do:
            clip_low            = config_Dataprep.get("clip_low")
            clip_high           = config_Dataprep.get("clip_high")
            
            ED_frame            = winsorizing(ED_frame, clip_low, clip_high)
            ES_frame            = winsorizing(ES_frame, clip_low, clip_high)
            ED_frame            = np.array(ED_frame, dtype=np.float32)
            ES_frame            = np.array(ES_frame, dtype=np.float32)
            
            print('Dimension of ED and ES frame after Resampling and winsorizing:', 'ED:', 
                  ED_frame.shape,'ES;', ES_frame.shape)
            steps              += '_clip'            

        if 'cropping' in preprocessing_to_do:
            crop_factor         = config_Dataprep.get("crop_factor")
            
            ED_frame            = crop_or_pad_slice_to_size(ED_frame, crop_factor[0], crop_factor[1])
            ED_label            = crop_or_pad_slice_to_size(ED_label, crop_factor[0], crop_factor[1])
            ED_frame            = np.array(ED_frame, dtype=np.float32)
            ED_label            = np.array(ED_label, dtype=np.uint8)
            
            ES_frame            = crop_or_pad_slice_to_size(ES_frame, crop_factor[0], crop_factor[1])
            ES_label            = crop_or_pad_slice_to_size(ES_label, crop_factor[0], crop_factor[1])
            ES_frame            = np.array(ES_frame, dtype=np.float32)
            ES_label            = np.array(ES_label, dtype=np.uint8)

            print('Dimension of ED and ES frame after Resampling and Cropping:', 'ED:', 
                  ED_frame.shape,'ES;', ES_frame.shape)
            steps              += '_crop'

        if 'normalize' in preprocessing_to_do:
            ED_frame            = MVN_np(ED_frame)
            ES_frame            = MVN_np(ES_frame)
            steps              += '_norm'
            
        if 'change_label' in preprocessing_to_do:
            old_labels          = config_Dataprep.get("old_labels")
            new_labels          = config_Dataprep.get("new_labels")
            
            ED_label, ES_label  = change_labels(ED_label,
                                                ES_label,
                                                old_labels,
                                                new_labels)

        if save_pkl or save_mat:            
            ED_images          += [ED_frame]
            ED_labels          += [ED_label]
            ES_images          += [ES_frame]
            ES_labels          += [ES_label]
            images             += [ED_frame]
            images             += [ES_frame]
            labels             += [ED_label]
            labels             += [ES_label]
                    
        if save_nii:
            os.makedirs(save_path + '/' + subject)
            new_path                    = save_path + '/' + subject
            
            ED_frame_path               = '{0}/{1}_{2}.nii.gz'.format(new_path, subject, 'ED')
            ED_header['pixdim'][1:4]    = [target_resolution[0], target_resolution[1], ED_pix_size[2]]
            ED_frame                    = nib.Nifti1Image(ED_frame.astype(np.float32), ED_affine, ED_header)
            #ED_frame                    = nib.Nifti1Image(ED_frame.astype(np.float32), None)
            nib.save(ED_frame, ED_frame_path)

            ED_label_path               = '{0}/{1}_{2}_gt.nii.gz'.format(new_path, subject, 'ED')
            ED_label                    = nib.Nifti1Image(ED_label.astype(np.uint8), ED_affine, ED_header)
            #ED_label                    = nib.Nifti1Image(ED_label.astype(np.uint8), None)
            nib.save(ED_label,ED_label_path)

            ES_frame_path               = '{0}/{1}_{2}.nii.gz'.format(new_path, subject, 'ES')
            ES_header['pixdim'][1:4]    = [target_resolution[0], target_resolution[1], ES_pix_size[2]]
            ES_frame                    = nib.Nifti1Image(ES_frame.astype(np.float32), ES_affine, ES_header)
            #ES_frame                    = nib.Nifti1Image(ES_frame.astype(np.float32), None)
            nib.save(ES_frame, ES_frame_path)

            ES_label_path               = '{0}/{1}_{2}_gt.nii.gz'.format(new_path, subject, 'ES')
            ES_label                    = nib.Nifti1Image(ES_label.astype(np.uint8), ED_affine, ED_header)
            #ES_label                    = nib.Nifti1Image(ES_label.astype(np.uint8), None)
            nib.save(ES_label, ES_label_path)

    
    if save_pkl:
        save_in_pkl(save_path, ED_images, ED_labels, ES_images, ES_labels, images, labels)
        
    if save_mat:
        save_in_mat(save_path, ED_images, ED_labels, ES_images, ES_labels, images, labels)

###############################################################################
if __name__=='__main__':
    
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print(' python main.py config_pre.txt')
        exit()

    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    
    config      = parse_config(config_file)

    Preprocess_data(config_file)
###############################################################################
