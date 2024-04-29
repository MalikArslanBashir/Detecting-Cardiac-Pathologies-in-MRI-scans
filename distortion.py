"""
Project Started: August 1, 2019
THIS FILE NEEDS TO BE REVIEWED AND TESTED
"""
import time
import cv2
import numpy                             as np
from scipy                               import ndimage
from scipy.ndimage.filters               import gaussian_filter
from scipy.ndimage.interpolation         import map_coordinates

def transform_matrix_offset_center(matrix, y, x):
    o_x                     = (x - 1) / 2.0
    o_y                     = (y - 1) / 2.0
    offset_matrix           = np.array([[1, 0, o_x],
                                        [0, 1, o_y],
                                        [0, 0,   1]])
    reset_matrix            = np.array([[1, 0, -o_x],
                                        [0, 1, -o_y],
                                        [0, 0,    1]])
    transform_matrix        = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def affine_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0., order=1):
    x                       = np.rollaxis(x, channel_index, 0)
    final_affine_matrix     = transform_matrix[:2, :2]
    final_offset            = transform_matrix[:2,  2]
    channel_images          = [ndimage.interpolation.affine_transform(
                                x_channel, final_affine_matrix, final_offset,
                                order=order, mode=fill_mode, cval=cval
                                ) for x_channel in x]
    x                       = np.stack(channel_images, axis=0)
    x                       = np.rollaxis(x, 0, channel_index + 1)
    return x

def affine_transform_cv2(x, transform_matrix, flags=None, border_mode='constant'):
    rows, cols              = x.shape[0], x.shape[1]
    if flags==1:
        flags               = cv2.INTER_AREA
    elif flags==0:
        flags               = cv2.INTER_NEAREST
    else:
        print('WRONG INTERPOLATION!')
    if border_mode is 'constant':
        border_mode         = cv2.BORDER_CONSTANT
    elif border_mode is 'replicate':
        border_mode         = cv2.BORDER_REPLICATE
    else:
        raise Exception("unsupport border_mode, check cv.BORDER_ for more details.")
    return cv2.warpAffine(x, transform_matrix[0:2,:], (cols, rows), flags=flags,
                                                        borderMode=border_mode)

def affine_zoom_matrix(zoom_range=(0.8, 1.1)):
    if isinstance(zoom_range, (float, int)):
        scale               = zoom_range
    elif isinstance(zoom_range, tuple):
        scale               = np.random.uniform(zoom_range[0], zoom_range[1])
    else:
        raise Exception("zoom_range: float or tuple of 2 floats")

    zoom_matrix             = np.array([[scale, 0,      0],
                                        [0,     scale,  0],
                                        [0,     0,      1]])
    return zoom_matrix

def gamma_correction_fn(image, gamma):
    """gamma correction code, as implemented in No New Net (Isensee et al.)"""
    intensity_range         = np.abs(image.max() - image.min())
    return (((image - image.min())/(intensity_range + 1e-7))**gamma) * \
            intensity_range + image.min()

class Distortion(object):
    def __init__(self, config_train_validate_test={}, data_channels=1):
        self.augmentations_to_do    = config_train_validate_test.get('augmentations_to_do')
        self.accumulate             = config_train_validate_test.get('augmentation_flag', False)
        
        # available methods: [brightness, vertical_flip, horizontal_flip, elastic_deform, rotation, shift, shear, zoom]
        self.data_channels          = data_channels
        self.METHODS_DICT           = {
                                        'brightness':self.Brightness(is_random=True),
                                        'vertical_flip':self.Flip(axis=0, is_random=True),
                                        'horizontal_flip':self.Flip(axis=1, is_random=True),
                                        'elastic_deform':self.Elastic_Transform(alpha=720, sigma=24, is_random=False),
                                        'rotation':self.Rotation(angle=30, is_random=True, fill_mode='constant'),
                                        'shift':self.Shift(wrg=0.1, hrg=0.1, is_random=True, fill_mode='constant'),
                                        'shear':self.Shear(intensity=0.2, is_random=True, fill_mode='constant'),
                                        'zoom':self.Zoom(zoom_range=(0.7, 1.4), flags=None, border_mode='constant')
                                      }

    def apply_distortion(self, image, label):
        slices_list                     = [image[:,:,i,np.newaxis] for i in range(self.data_channels)] + [label]
        
        if self.accumulate:             # original plus all distortions applied on original
            data_list, label_list       = [image], [label]
            for method in self.augmentations_to_do:
                distorted_slices_list   = self.METHODS_DICT.get(method, None)(slices_list)
                if method=='zoom':
                    distorted_slices_list = [np.expand_dims(i, axis=-1) for i in distorted_slices_list]
                data_list.append(np.concatenate(distorted_slices_list[:-1], axis=-1))
                label_list.append(distorted_slices_list[-1])
            return data_list, label_list
        
        else:                           # perform sequence of operations and return result
            
            for method in self.augmentations_to_do:
                slices_list             = self.METHODS_DICT.get(method, None)(slices_list)
                if method=='zoom':
                    slices_list         = [np.expand_dims(i, axis=-1) for i in slices_list]
            distorted_image             = np.concatenate(slices_list[:-1], axis=-1)
            distorted_label             = slices_list[-1]
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Distortion is Done>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")		
            return [distorted_image], [distorted_label]

    @staticmethod
    def Brightness(gamma=0.2, gain=1, is_random=False):
        def Brightness_function(images):
            gamma                   = 0.2
            if is_random:
                gamma               = np.random.uniform(1 - gamma, 1 + gamma)
            brightness_distorted_images = []
            for i in range(len(images)-1):
                #brightness_distorted_images.append(exposure.adjust_gamma(images[i], gamma, gain))
                brightness_distorted_images.append(gamma_correction_fn(images[i], gamma))
            brightness_distorted_images.append(images[-1])
            return brightness_distorted_images
        return Brightness_function

    @staticmethod
    def Flip(axis=1, is_random=False):
        def Flip_function(images):
            flipped_images          = []
            if is_random:
                factor              = np.random.uniform(-1, 1)
                if factor > 0:
                    for image in images:
                        image       = np.asarray(image).swapaxes(axis, 0)
                        image       = image[::-1, ...]
                        image       = image.swapaxes(0, axis)
                        flipped_images.append(image)
                    return flipped_images
                else:
                    return images
            else:
                for image in images:
                    image           = np.asarray(image).swapaxes(axis, 0)
                    image           = image[::-1, ...]
                    image           = image.swapaxes(0, axis)
                    flipped_images.append(image)
                return flipped_images
        return Flip_function

    @staticmethod
    def Elastic_Transform(alpha, sigma, mode='constant', cval=0, is_random=False):
        def Elastic_Transform_function(images):
            if is_random is False:
                random_state        = np.random.RandomState(None)
            else:
                random_state        = np.random.RandomState(int(time.time()))

            shape                   = images[0].shape
            if len(shape)==3:
                shape               = (shape[0], shape[1])
            new_shape               = random_state.rand(*shape)

            elastic_deformed_images = []
            transform_order         = 1
            for i in range(len(images)):
                if i == len(images)-1:
                    transform_order = 0
                image               = images[i]
                is_3d               = False
                if len(image.shape) == 3 and image.shape[-1] == 1:
                    image           = image[:, :, 0]
                    is_3d           = True
                elif len(image.shape) == 3 and image.shape[-1] != 1:
                    raise Exception("Only support greyscale image")
                if len(image.shape) != 2:
                    raise AssertionError("input should be grey-scale image")

                dx                  = gaussian_filter((new_shape * 2 - 1), sigma, mode=mode, cval=cval) * alpha
                dy                  = gaussian_filter((new_shape * 2 - 1), sigma, mode=mode, cval=cval) * alpha
                x_, y_              = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
                indices             = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
                if is_3d:
                    elastic_deformed_images.append(
                                    map_coordinates(image,
                                                    indices,
                                                    order=transform_order).reshape((shape[0], shape[1], 1)))
                else:
                    elastic_deformed_images.append(
                                    map_coordinates(image,
                                                    indices,
                                                    order=transform_order).reshape(shape))
            return elastic_deformed_images
        return Elastic_Transform_function

    @staticmethod
    def Rotation(angle=30, is_random=False, row_index=0, col_index=1,
                channel_index=2, fill_mode='constant', cval=0., order=1):
        def Rotation_function(images):
            if is_random:
                theta               = np.random.uniform(-angle, angle) * np.pi/180
            else:
                theta               = angle * np.pi/180
            rotation_matrix         = np.array([[np.cos(theta), -np.sin(theta), 0],
                                                [np.sin(theta),  np.cos(theta), 0],
                                                [            0,              0, 1]])
            height, width           = images[0].shape[row_index], images[0].shape[col_index]
            transform_matrix        = transform_matrix_offset_center(rotation_matrix, height, width)
            rotated_images          = []
            for i in range(len(images)-1):
                image               = affine_transform(images[i], transform_matrix, channel_index, fill_mode, cval, order)
                rotated_images.append(image)
            label                   = affine_transform(images[-1], transform_matrix, channel_index, fill_mode, cval, order=0)
            rotated_images.append(label)
            return rotated_images
        return Rotation_function

    @staticmethod
    def Shift(wrg=0.1, hrg=0.1, is_random=False, row_index=0, col_index=1,
                        channel_index=2, fill_mode='nearest', cval=0., order=1):
        def Shift_function(images):
            height, width           = images[0].shape[row_index], images[0].shape[col_index]
            if is_random:
                tx                  = np.random.uniform(-hrg, hrg) * height
                ty                  = np.random.uniform(-wrg, wrg) * width
            else:
                tx, ty              = hrg * height, wrg * width
            translation_matrix      = np.array([[1, 0, tx],
                                                [0, 1, ty],
                                                [0, 0,  1]])
            shifted_images          = []
            for i in range(len(images)-1):
                image               = affine_transform(images[i], translation_matrix,
                                        channel_index, fill_mode, cval, order)
                shifted_images.append(image)
            label                   = affine_transform(images[-1], translation_matrix,
                                    channel_index, fill_mode, cval, order=0)
            shifted_images.append(label)
            return shifted_images
        return Shift_function

    @staticmethod
    def Shear(intensity=0.1, is_random=False, row_index=0, col_index=1,
                    channel_index=2, fill_mode='nearest', cval=0., order=1):
        def Shear_function(images):
            """
            Shear an image randomly or non-randomly.
            Args:
                images : list of numpy.arrays, images with dimensions of [row, col, channel] (default).
                intensity : float, Percentage of shear, usually -0.5 ~ 0.5 (is_random==True),
                            0 ~ 0.5 (is_random==False), you can have a quick try by shear(X, 1).
                is_random : boolean, If True, randomly shear. Default is False.
                row_index col_index and channel_index : int, Index of row, col and channel,
                            default (0, 1, 2), for theano (1, 2, 0).
                fill_mode : str, Method to fill missing pixel, default `nearest`,
                            more options `constant`, `reflect` or `wrap`.
                cval : float, Value used for points outside the boundaries of the input
                            if mode='constant'. Default is 0.0.
                order : int, The order of interpolation. The order has to be in the range 0-5.
            Returns:
                list of numpy.arrays, processed images.
            """
            if is_random:
                shear               = np.random.uniform(-intensity, intensity)
            else:
                shear               = intensity
            shear_matrix            = np.array([[1, -np.sin(shear), 0],
                                                [0,  np.cos(shear), 0],
                                                [0,              0, 1]])

            height, width           = images[0].shape[row_index], images[0].shape[col_index]
            transform_matrix        = transform_matrix_offset_center(shear_matrix,
                                            height, width)
            sheared_images          = []
            for i in range(len(images)-1):
                image               = affine_transform(images[i], transform_matrix,
                                            channel_index, fill_mode, cval, order)
                sheared_images.append(image)
            label                   = affine_transform(images[-1], transform_matrix,
                                        channel_index, fill_mode, cval, order=0)
            sheared_images.append(label)
            return sheared_images
        return Shear_function

    @staticmethod
    def Zoom(zoom_range=(0.9, 1.1), flags=None, border_mode='constant'):
        def Zoom_function(images):
            """Zooming/Scaling a single image that height and width are changed together.
            Args:
                images : list of numpy.arrays, images with dimension of [row, col, channel] (default).
                zoom_range : float or tuple of 2 floats, The zooming/scaling ratio, greater than 1 means larger.
                        - float, a fixed ratio.
                        - tuple of 2 floats, randomly sample a value as the ratio between 2 values.
                border_mode : str
                    - `constant`, pad the image with a constant value (i.e. black or 0)
                    - `replicate`, the row or column at the very edge of the original is replicated to the extra border.
            Returns:
                numpy.arrays, processed images.
            """
            zoomed_images           = []
            zoom_matrix             = affine_zoom_matrix(zoom_range=zoom_range)
            for i in range(len(images)-1):
                image               = images[i]
                height, width       = image.shape[0], image.shape[1]
                transform_matrix    = transform_matrix_offset_center(zoom_matrix, height, width)
                image               = affine_transform_cv2(image, transform_matrix, flags=1, border_mode=border_mode)
                zoomed_images.append(image)
            label                   = affine_transform_cv2(images[-1], transform_matrix, flags=0, border_mode=border_mode)
            zoomed_images.append(label)
            return zoomed_images
        return Zoom_function
    
################################################################################
