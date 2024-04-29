
from __future__                                 import absolute_import, print_function
import math
import numbers
import numpy                                    as np
import tensorflow                               as tf
import                                      nibabel     as         nib
from   medpy.metric.binary                      import hd95, dc

epsilon = 1e-7
axes    = (0,1,2)

def pad(tensor, pad_factor):
    # Used only in Vup-tran code
    
    padded = tf.pad(tensor, [[0,0], [pad_factor,pad_factor], [pad_factor,pad_factor], [0,0]], 
                    mode='CONSTANT', name=None, constant_values=0)

    return padded

###############################################################################

def crop(tensors):
    '''List of 2 tensors, the second tensor having larger spatial dimensions.'''
    # Used only in Vup-tran code
    
    h_dims = []
    w_dims = []
    
    for t in tensors:
        b, h, w, d  = t.get_shape().as_list()
        h_dims.append(h)
        w_dims.append(w)
    
    crop_h, crop_w  = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h           = crop_h % 2
    rem_w           = crop_w % 2
	
    crop_h_dims     = (crop_h // 2, crop_h // 2 + rem_h)  
    crop_w_dims     = (crop_w // 2, crop_w // 2 + rem_w)

    print (crop_h_dims)
    print (crop_w_dims)
    
    cropped         = tf.keras.layers.Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])
    
    return cropped

###############################################################################

def average(tensors):
    # Used only in Vup-tran code
    
    output = tensors[0]
    for i in range(1, len(tensors)):
        output     += tensors[i]
        output      = output / len(tensors)
    
    return output

###############################################################################

def MVN_tf(image):
    '''Performs mean-variance normalization with Tensorflow'''
    
    mean, var       = tf.nn.moments(image, axis=axes, keep_dims=True)
    std             = (var)*0.5
    mvn             = (image - mean) / (std + epsilon)
	
    return mvn

###############################################################################

def MVN_np (image):
    '''Performs mean-variance normalization with NumPy'''
    
    mean            = np.mean(image, axis=axes)
    std             = np.std(image, axis=axes)
    image           = (image - mean) / (std + epsilon)

    return image

###############################################################################

def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 2 or 3 (height, width, depth).
    Argument crop_size is an integer for square cropping only.
    Perform center cropping to a specified size.
    '''
    
    if (len(ndarray.shape) == 2):
        h, w = ndarray.shape
        
        if crop_size == 0:
            raise ValueError('argument crop_size must be non-zero integer')
        
        if any([dim < crop_size for dim in (h, w)]):
            # zero pad along each (h, w) dimension before center cropping
            pad_h       = (crop_size - h) if (h < crop_size) else 0
            pad_w       = (crop_size - w) if (w < crop_size) else 0
            rem_h       = math.floor((pad_h % 2.0))
            rem_w       = math.floor((pad_w % 2.0))
            pad_dim_h   = (math.floor(pad_h/2), math.floor(pad_h/2) + rem_h)
            pad_dim_w   = (math.floor(pad_w/2), math.floor(pad_w/2) + rem_w)
            
            # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
            npad        = (pad_dim_h, pad_dim_w)
            ndarray     = np.pad(ndarray, npad, 'constant', constant_values=0)
            h, w        = ndarray.shape
		
        # center crop
        h_offset        = math.floor((h - crop_size) / 2)
        w_offset        = math.floor((w - crop_size) / 2)
        cropped         = ndarray[h_offset:(h_offset+crop_size),
                                  w_offset:(w_offset+crop_size)]

        return cropped
    
    if (len(ndarray.shape) == 3):
        h, w, d = ndarray.shape
        
        if crop_size == 0:
            raise ValueError('argument crop_size must be non-zero integer')

        if any([dim < crop_size for dim in (h, w)]):
            # zero pad along each (h, w) dimension before center cropping
            pad_h       = (crop_size - h) if (h < crop_size) else 0
            pad_w       = (crop_size - w) if (w < crop_size) else 0
            rem_h       = math.floor((pad_h % 2.0))
            rem_w       = math.floor((pad_w % 2.0))
            pad_dim_h   = (math.floor(pad_h/2), math.floor(pad_h/2) + rem_h)
            pad_dim_w   = (math.floor(pad_w/2), math.floor(pad_w/2) + rem_w)
			
            # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
            npad        = (pad_dim_h, pad_dim_w, (0,0))
            ndarray     = np.pad(ndarray, npad, 'constant', constant_values=0)
            h, w, d     = ndarray.shape
            
        # center crop
        h_offset        = math.floor((h - crop_size) / 2)
        w_offset        = math.floor((w - crop_size) / 2)
        cropped         = ndarray[h_offset:(h_offset+crop_size),
                                  w_offset:(w_offset+crop_size), :]

        return cropped


def crop_or_pad_slice_to_size(slice, nx, ny):
    ''' Perform 3D Cropping'''
    x, y, z         = slice.shape
    x_s             = math.floor((x - nx) / 2)
    y_s             = math.floor((y - ny) / 2)
    x_c             = math.floor((nx - x) / 2)
    y_c             = math.floor((ny - y) / 2)

    if x > nx and y > ny:
        slice_cropped                                   = slice[x_s:x_s + nx, y_s:y_s + ny, :]
        
    else:
        slice_cropped                                   = np.zeros((nx, ny, z))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :, :]            = slice[:, y_s:y_s + ny, :]
            
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y,:]             = slice[x_s:x_s + nx, :, :]
            
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y, :]  = slice[:, :, :]
    
    return slice_cropped



def crop_or_pad_image_to_size(image, nx, ny):
    ''' Perform 3D Cropping'''
    b,x, y, z         = image.shape
    x_s             = math.floor((x - nx) / 2)
    y_s             = math.floor((y - ny) / 2)
    x_c             = math.floor((nx - x) / 2)
    y_c             = math.floor((ny - y) / 2)

    if x > nx and y > ny:
        image_cropped                                   = image[:,x_s:x_s + nx, y_s:y_s + ny, :]
        
    else:
        image_cropped                                   = np.zeros((b,nx, ny, z))
        if x <= nx and y > ny:
            image_cropped[:,x_c:x_c + x, :, :]            = image[:,:, y_s:y_s + ny, :]
            
        elif x > nx and y <= ny:
            image_cropped[:,:, y_c:y_c + y,:]             = image[:,x_s:x_s + nx, :, :]
            
        else:
            image_cropped[:,x_c:x_c + x, y_c:y_c + y, :]  = image[:,:, :, :]
    
    return image_cropped#################




##############################################################

class Randomcrop(object):

    def __init__(self, size, random_state=np.random):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.random_state = random_state 

    def __call__(self, img,mask):

        w, h    = img.shape[0],img.shape[1]
        th, tw  = self.size

        if w == tw and h == th:
            return img

        x1 = self.random_state.randint(0, w - tw)
        y1 = self.random_state.randint(0, h - th)

        return img[x1:x1 + tw, y1: y1 + th], mask[x1:x1 + tw, y1: y1 + th]
    
###############################################################################



def Read_nii_files(path,subject):
	path1               = path+'/'+subject
	ED_frame            = nib.load(path1+'/'+subject+'_ED.nii.gz')
	ED_affine           = ED_frame.affine
	ED_header           = ED_frame.header
	ED_pix_size         = (ED_frame.header['pixdim'][1], 
                               ED_frame.header['pixdim'][2],
                               ED_frame.header['pixdim'][3])

	ED_frame          = ED_frame.get_data().astype(np.float32)
	ED_label          = nib.load(path1+'/'+subject+'_ED_gt.nii.gz')
	ED_label          = ED_label.get_data().astype(np.uint8)

	ES_frame          = nib.load(path1+'/'+subject+'_ES.nii.gz')
	ES_affine         = ES_frame.affine
	ES_header         = ES_frame.header
	ES_pix_size       = (ES_frame.header['pixdim'][1], 
                             ES_frame.header['pixdim'][2],
                             ES_frame.header['pixdim'][3])

	ES_frame          = ES_frame.get_data().astype(np.float32)
	ES_label          = nib.load(path1+'/'+subject+'_ES_gt.nii.gz')
	ES_label          = ES_label.get_data().astype(np.uint8)

	return ED_frame, ES_frame, ED_label, ES_label, ED_affine, ES_affine, ED_pix_size, ES_pix_size

def save_nii(img_path, data, affine, header=None):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg                 =             nib.Nifti1Image(data.astype(np.uint8), affine=affine, header=header)
    nimg.to_filename(img_path)


def metrics(img_gt, img_pred, resolution):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.



    Return
    ------
    A list of metrics in this order, [Dice LV,Dice RV,Dice MYO]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
 
   # Loop on each classes of the input images
    for c in [1,2,3]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i    = np.clip(gt_c_i, 0, 1)
        pred_c_i  = np.clip(pred_c_i, 0, 1)
        #print('GT',np.unique(gt_c_i),'Pred',np.unique(pred_c_i))
        # Compute the Hausdroff      #    Compute the Dice

        if (len(np.unique(pred_c_i))==1 and len(np.unique(gt_c_i))==1):
            hd_dis = 0
            dice   = 1
        elif (len(np.unique(pred_c_i))!=2 or len(np.unique(gt_c_i))!=2):

            hd_dis  = np.sqrt(np.square(gt_c_i.shape[0]*resolution[0])+
                       np.square(gt_c_i.shape[1]*resolution[1])+
                       np.square(gt_c_i.shape[2]*resolution[2]))
            dice    = 0
        else:
            hd_dis  = hd95(gt_c_i, pred_c_i, resolution)
            dice    =   dc(gt_c_i, pred_c_i)


        # Compute volume
       # volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
       # volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice,hd_dis]#, volpred, volpred-volgt]
    return res

def Dice_only(img_gt, img_pred):
    """
    Function to compute the Dice between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.



    Return
    ------
    A list of metrics in this order, [Dice LV,Dice RV,Dice MYO]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
 
   # Loop on each classes of the input images
    for c in [1, 3, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i    = np.clip(gt_c_i, 0, 1)
        pred_c_i  = np.clip(pred_c_i, 0, 1)
        #print('GT',np.unique(gt_c_i),'Pred',np.unique(pred_c_i))
        # Compute the Hausdroff      #    Compute the Dice

        if (len(np.unique(pred_c_i))==1 and len(np.unique(gt_c_i))==1):
            dice   = 1
        elif (len(np.unique(pred_c_i))!=2 or len(np.unique(gt_c_i))!=2):

            dice    = 0
        else:
            dice    =   dc(gt_c_i, pred_c_i)


        # Compute volume
       # volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
       # volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice]#, volpred, volpred-volgt]

    return res
def clinical_metrics(img, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img: np.array
    Array of the segmentation map.
    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Volume LV, Volume RV,Volume MYO]
    """

    if img.ndim != 3:
        raise ValueError("The arrays 'img' should have the dimension 3")

    res = []
    # Loop on each classes of the input images
    for c in [1, 3, 2]:

        # Copy the pred image to not alterate the input
        img_c_i = np.copy(img)
        img_c_i[img_c_i != c] = 0

        # Clip the value to compute the volumes
        img_c_i = np.clip(img_c_i, 0, 1)




        # Compute volume
        vol = img_c_i.sum() * np.prod(voxel_size) / 1000.


        res += [vol]



    return res

