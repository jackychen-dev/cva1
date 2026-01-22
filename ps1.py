import math
import numpy as np
import cv2

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    # OpenCV uses BGR, so red is channel index 2
    red_channel = temp_image[:, :, 2]
    
    return red_channel


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """

    temp_image = np.copy(image)
    green_channel = temp_image[:, :, 1]
    return green_channel
    


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image = np.copy(image)
    blue_channel = temp_image[:, :, 0]
    return blue_channel


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """

    temp_image = np.copy(image)
    green_channel = extract_green(temp_image)
    blue_channel = extract_blue(temp_image)
    temp_image[:, :, 1] = blue_channel
    temp_image[:, :, 0] = green_channel
    swapped_image = temp_image
    return swapped_image


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """

    temp_image = np.copy(dst)

    src_H, src_W = src.shape #'this is height of the source image and width of the source image
    dst_H, dst_W = dst.shape
    h, w = int(shape[0]), int(shape[1]) #'this is height and width of the section to be copied'

    src_h_start = (src_H - h) // 2
    src_w_start = (src_W - w) // 2

    src_middle = src[
        src_h_start:src_h_start + h,
        src_w_start:src_w_start + w
    ]

    dst_h_start = (dst_H - h) // 2
    dst_w_start = (dst_W - w) // 2

    temp_image[ 
        dst_h_start:dst_h_start + h,
        dst_w_start:dst_w_start + w
    ] = src_middle

    return temp_image



def copy_paste_middle_circle(src, dst, radius):
    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """


    temp_image = np.copy(dst)

    src_H, src_W = src.shape #this is height of the source image and width of the source image
    dst_H, dst_W = dst.shape
    r = int(radius) # this is radius of the section that needs to be copied

    src_cx= (src_W-1)//2 
    src_cy= (src_H-1)//2
    dst_cx= (dst_W-1)//2
    dst_cy= (dst_H-1)//2
    

    for y in range(-r, r+1):
        for x in range(-r, r+1):
            if x*x + y*y <= r*r:
                temp_image[dst_cy+y, dst_cx+x] = src[src_cy+y, src_cx+x]
      
    return temp_image


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image)

    min_val = np.min(temp_image)
    max_val = np.max(temp_image)
    mean_val = np.mean(temp_image)
    stddev_val = np.std(temp_image)
    
    statistics=( float(min_val),  float(max_val),float(mean_val), float(stddev_val),)
    return statistics 


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """

    temp_image = np.copy(image)
   
    mean_val = np.mean(temp_image)
    stddev_val = np.std(temp_image)
    
    temp_image = (temp_image - mean_val) / stddev_val * scale + mean_val


    return temp_image


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image)
    height, width= temp_image.shape

    temp_image[:,:width-shift] = temp_image[:,shift:] #this is the part that is shifted to the left
    temp_image[:, width-shift:] = temp_image[:, width-shift-1:width-shift]

    return temp_image


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """

    temp_image1 = np.copy(img1).astype(np.float64)
    temp_image2 = np.copy(img2).astype(np.float64)

    temp_image = temp_image1 - temp_image2
    min_val = np.min(temp_image)
    max_val = np.max(temp_image)
    mean_val = np.mean(temp_image)
    stddev_val = np.std(temp_image)

    if max_val == min_val:
       return np.zeros_like(temp_image, dtype=np.float64) #identical images

    temp_image = (temp_image - min_val) / (max_val - min_val) * 255.0

    return temp_image


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """


    temp_image = np.copy(image).astype(np.float64)
    H,W,_=temp_image.shape

    noise = np.random.normal(loc=0.0, scale=sigma, size=(H, W))
    temp_image[:, :, channel] += noise



    return temp_image


def build_hybrid_image(image1, image2, cutoff_frequency):
    """ 
    Takes two images and creates a hybrid image given a cutoff frequency.
    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        cutoff_frequency: scalar
    
    Returns:
        hybrid_image: numpy nd-array of dim (m, n, c)

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """

    filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                                   sigma=cutoff_frequency) #this is the gaussian filter
    filter = np.dot(filter, filter.T) #this is the gaussian filter matrix
    
    low_frequencies = cv2.filter2D(image1,-1,filter)#this is the low frequencies of the image 1

    high_frequencies = image2 - cv2.filter2D(image2,-1,filter)#this is the high frequencies of the image 2

    hybrid_image = low_frequencies + high_frequencies
    
    return hybrid_image

def vis_hybrid_image(hybrid_image):
    """ 
    Tools to visualize the hybrid image at different scale.

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """


    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
      # add padding
      output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                          dtype=np.float32)))

      # downsample image
      cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

      # pad the top to append to the output
      pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                     num_colors), dtype=np.float32)
      tmp = np.vstack((pad, cur_image))
      output = np.hstack((output, tmp))

    return output
