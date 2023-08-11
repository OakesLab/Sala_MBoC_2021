'''Functions used for analysis of Stefano's data for the testin paper

'''

import numpy as np                                                              # for matrix math
import untangle                                                                 # for parsing XML
import cv2                                                                      # for filtering vector fields
from scipy.ndimage import morphology                                            # morphological operations
from skimage.morphology import opening, disk, dilation, remove_small_objects    # morphology operations
from scipy.fft import fft2, ifft2, ifftshift, fftshift                          # for fourier transform operations
from skimage.registration import phase_cross_correlation                        # registration function

import matplotlib.pyplot as plt                                                 # for plotting
import numpy.matlib as matlib                                                   # matrix operations
from skimage.transform import rotate                                            # image rotation
import pandas as pd                                                             # for dataframes
from scipy import optimize                                                      # to fit photobleach decay curve
import skimage.io as io                                                         # for reading in images
import os

def parse_XML(XML_filename):
    '''
    This function is built with Slidebook6 xml files in mind
    '''
    # use untangle to parse XML
    XML_data = untangle.parse(XML_filename)
    
    # get the number of channels in the movie
    N_channels = int(XML_data.OME.Image.Pixels['SizeC'])
    
    # get the number of frames in the movie
    N_frames = int(XML_data.OME.Image.Pixels['SizeT'])
    
    # get the number of rows and columns
    N_rows = int(XML_data.OME.Image.Pixels['SizeX'])
    N_cols = int(XML_data.OME.Image.Pixels['SizeY'])
    
    # get the time points in the movie
    # create an empty list to hold time points
    time = []
    for dt in XML_data.OME.Image.Pixels.Plane:
        time.append(float(dt['DeltaT']))

    # Take every Nth frame depending on the number of channels
    time = time[::N_channels]  
    # turn the list into an array
    time_points = np.asarray(time)

    # find the time interval
    frame_interval = np.mean(np.diff(time_points))
    
    # get the detector
    if len(XML_data.OME.Image.Pixels.Channel) > 1:
        camera = XML_data.OME.Image.Pixels.Channel[0].DetectorSettings['ID']
    else:
        camera = XML_data.OME.Image.Pixels.Channel.DetectorSettings['ID']
    if camera == 'Detector:0':
        camera_name = 'Prime'
        camera_pixelsize = 11.0
    elif camera == 'Detector:1':
        camera_name = 'Fusion'
        camera_pixelsize = 6.5
    elif camera == 'Detector:2':
        camera_name = 'Evolve'
        camera_pixelsize = 16
        
    # get the objective magnification
    objective = XML_data.OME.Image.ObjectiveSettings['ID']
    objective_magnification = float(XML_data.OME.Instrument.Objective[int(objective[-1])]['CalibratedMagnification'])
    
    # resolution calculation
    xy_resolution = camera_pixelsize / objective_magnification
    
    # get the ablation_timepoint
    ablation_timepoint = int(XML_data.OME.ROI[1].Union.Shape['theT'])

    # get the ablation center coordinate
    ROI_col = int(XML_data.OME.ROI[0].Union.Shape.Rectangle['X'])
    ROI_row = int(XML_data.OME.ROI[0].Union.Shape.Rectangle['Y'])
    ROI_w = int(XML_data.OME.ROI[0].Union.Shape.Rectangle['Width'])
    ROI_h = int(XML_data.OME.ROI[0].Union.Shape.Rectangle['Height'])
    ROI_center_row = int(ROI_row + ROI_h / 2)
    ROI_center_col = int(ROI_col + ROI_w / 2)
    
    # Create a dictionary to hold all the data
    Exp_details = {
        'N_channels' : N_channels,
        'N_frames' : N_frames,
        'N_rows' : N_rows,
        'N_cols' : N_cols,
        'time_pts' : time_points,
        'time_interval' : frame_interval,
        'camera' : camera_name,
        'camera_pixelsize' : camera_pixelsize,
        'objective_magnification' : int(objective_magnification),
        'xy_resolution' : xy_resolution,
        'ablation_timepoint' : ablation_timepoint,
        'ROI_col' : ROI_col,
        'ROI_row' : ROI_row,
        'ROI_w' : ROI_w,
        'ROI_h' : ROI_h,
        'ROI_center_row' : ROI_center_row,
        'ROI_center_col' : ROI_center_col,
        'file' : XML_filename.split('/')[-1].split('_')[0],
        'base_folder' : XML_filename[:XML_filename.rfind('/')] + '/'
    }
    
    # return the dictionary
    return Exp_details

def make_ablation_ROI(exp_details, **kwargs):
    # kwargs = 'target_circle_radius', 'ROI_multiple'
    
    # check for ROI_multiple
    if 'ROI_multiple' in kwargs:
        ROI_multiple = kwargs['ROI_multiple']
    else:
        ROI_multiple = 1
    
    # define target region location and size
    target_circle_col = exp_details['ROI_center_col']
    target_circle_row = exp_details['ROI_center_row']
    if 'ROI_radius' in kwargs:
        target_circle_radius = kwargs['ROI_radius']
    else:
        target_circle_radius = np.max((exp_details['ROI_w'], exp_details['ROI_h']))

    # make a mask
    mask = np.zeros((exp_details['N_rows'],exp_details['N_cols']))
    # set the center of the targer ROI in the mask equal to 1
    mask[target_circle_row,target_circle_col] = 1
    # make a structuring element to filter the mask
    target_ROI = disk(target_circle_radius * ROI_multiple)
    # filter the mask with the structuring element to define the ROI
    mask = cv2.filter2D(mask, -1, target_ROI)
    
    # bg mask
    bg = np.zeros((exp_details['N_rows'],exp_details['N_cols']))
    # set the center of the targer ROI in the mask equal to 1
    bg[target_circle_row,target_circle_col] = 1
    # make a structuring element to filter the mask
    bg_ROI = disk(target_circle_radius * ROI_multiple * 2)
    # filter the mask with the structuring element to define the ROI
    bg = cv2.filter2D(bg, -1, bg_ROI)
    # invert the mask
    bg = -1 * (bg - 1)

    return mask, bg

def register_image_stack(image_stack):
    '''
    Registers an imagestack to the first plane
    '''
    # Get the number of images in the stack
    N_images, N_rows, N_cols = image_stack.shape

    # reference image is the first image in the stack
    reference_image = image_stack[0].copy()

    # Get the reference image intensity
    reference_image_intensity = np.sum(reference_image)

    # Create an empty array to hold the registered images
    image_stack_registered = np.zeros_like(image_stack)

    # Create an empty array to hold the shift coordinates to reference for use in other channels
    shift_coordinates = np.zeros((N_images, 4))

    # Create a matrix with the row and column numbers for the registered image calculation
    Nr = ifftshift(np.arange(-1 * np.fix(N_rows/2), np.ceil(N_rows/2)))
    Nc = ifftshift(np.arange(-1 * np.fix(N_cols/2), np.ceil(N_cols/2)))
    Nc, Nr = np.meshgrid(Nc, Nr)

    # Define the subpixel resolution
    subpixel_resolution = 100

    # For loop to register each plane in the stack
    for plane, image in enumerate(image_stack):
        # Perform the subpixel registration
        shift, error, diffphase = phase_cross_correlation(reference_image, image, upsample_factor = subpixel_resolution)
        # Store the shift coordinates
        shift_coordinates[plane] = np.array([shift[0], shift[1], error, diffphase])

        # Calculate the shifted image
        shifted_image_fft = fft2(image) * np.exp(
                1j * 2 * np.pi * (-shift[0] * Nr / N_rows - shift[1] * Nc / N_cols))
        shifted_image_fft = shifted_image_fft * np.exp(1j * diffphase)
        shifted_image = np.abs(ifft2(shifted_image_fft))
        image_stack_registered[plane,:,:] = shifted_image.copy()
        
    return image_stack_registered, shift_coordinates

def shift_image_stack(image_stack, shift_coordinates):
    '''
    Shift an image stack based on the shifts caluclated in a previously registered stack. 
    Use this to register other channels in a multichannel stack
    '''

    # Get the shape of your stack
    N_planes, N_rows, N_cols = image_stack.shape
    
    # Create a matrix with the row and column numbers for the registered image calculation
    Nr = ifftshift(np.arange(-1 * np.fix(N_rows/2), np.ceil(N_rows/2)))
    Nc = ifftshift(np.arange(-1 * np.fix(N_cols/2), np.ceil(N_cols/2)))
    Nc, Nr = np.meshgrid(Nc, Nr)
    
    # Create an empty array to hold the registered image
    image_registered = np.zeros((N_planes, N_rows, N_cols))
    
    # register each plane based on the provided coordinates
    for plane, image in enumerate(image_stack):
        shifted_image_fft = fft2(image) * np.exp(
            1j * 2 * np.pi * (-shift_coordinates[plane,0] * Nr / N_rows - shift_coordinates[plane,1] * Nc / N_cols))
        shifted_image_fft = shifted_image_fft * np.exp(1j * shift_coordinates[plane,3])
        shifted_image = np.abs(ifft2(shifted_image_fft))
        image_registered[plane] = shifted_image.copy()
    
    return image_registered
    
def intensity_analysis(c0, c1, ROI_mask, overlay_mask, thresh, exp_details, save_figure = True):
    '''
    For calculating the average intensity in an ablation region based on the thresh percent brightest pixels

    c0            : uint16 or float32  : Imagestack that contains the images (e.g. testin) you want to search for the brightest pixels
    c1            : uint16 or float32  : Imagestack that contains a secondary channel (e.g. actin)
    ROI_mask      : Bool or uint8      : Mask of the region to search for the brightest pixels in
    overlay_mask  : Bool or uint8      : Mask of the region to search for the brightest pixels to create the spatial overlay
    thresh        : Float              : Value between 0 and 1 to represent the percent brightest pixels to search for
    exp_details   : dict               : Dictionary with the experimental details from the xml file of the experiment
    save_figure   : True/False         : Save the output figure
    '''


    # Define the number of points in the ROI mask and overlay mask
    N_pts_ROI = np.sum(ROI_mask.astype('uint8'))
    N_pts_overlay = np.sum(overlay_mask.astype('uint8'))
    # Determine the top % of points that we want to keep
    n_pts_ROI = int(N_pts_ROI * thresh)
    n_pts_overlay = int(N_pts_overlay * thresh)

    # make an empty matrix to hold your ROI_mask points
    LIM_ROI_pts = np.zeros_like(c0).astype('bool')
    # make an empty matrix to hold your ROI_mask points
    LIM_overlay_pts = np.zeros_like(c0).astype('bool')
    
    # make a reference image to compare signals in the LIM channel to
    ref_c0 = np.mean(c0[:exp_details['ablation_timepoint']], axis=0)
    ref_c1 = np.mean(c1[:exp_details['ablation_timepoint']], axis=0)
    
    # make empty lists to hold your intensities
    LIM_inten_ROI_peak, actin_inten_ROI_peak, LIM_inten_ROI_peak_std, actin_inten_ROI_peak_std = [], [], [], []
    LIM_inten_ROI_ave, actin_inten_ROI_ave, LIM_inten_ROI_ave_std, actin_inten_ROI_ave_std = [], [], [], []
    
    # for loop to find intensity in each time point in the movie
    for plane, image in enumerate(c0):
        # Find the difference in intensity in the LIM channel from the reference image and sort
        ROI_diff_masked_pts = sorted(image[ROI_mask] - ref_c0[ROI_mask])
        overlay_diff_masked_pts = sorted(image[overlay_mask] - ref_c0[overlay_mask])
        # Make a mask by finding all the points in the image above threshhold and multiply by ROI_mask or overlay_mask
        # to restrict to points in the ROI
        LIM_ROI_pts[plane] = ((image - ref_c0) > ROI_diff_masked_pts[-n_pts_ROI]) * ROI_mask
        LIM_overlay_pts[plane] = ((image - ref_c0) > overlay_diff_masked_pts[-n_pts_overlay]) * overlay_mask

        # find the average and std intensity difference from reference pixels in each channel using those points
        LIM_inten_ROI_peak.append(np.mean((c0[plane][LIM_ROI_pts[plane]] - ref_c0[LIM_ROI_pts[plane]]) / (ref_c0[LIM_ROI_pts[plane]])))
        LIM_inten_ROI_peak_std.append(np.std((c0[plane][LIM_ROI_pts[plane]] - ref_c0[LIM_ROI_pts[plane]]) / (ref_c0[LIM_ROI_pts[plane]])))
        actin_inten_ROI_peak.append(np.mean((c1[plane][LIM_ROI_pts[plane]] - ref_c1[LIM_ROI_pts[plane]]) / (ref_c1[LIM_ROI_pts[plane]])))
        actin_inten_ROI_peak_std.append(np.std((c1[plane][LIM_ROI_pts[plane]] - ref_c1[LIM_ROI_pts[plane]]) / (ref_c1[LIM_ROI_pts[plane]])))
        
        # find the average and std intensity difference from all pixels in ROI in each channel
        LIM_inten_ROI_ave.append(np.mean((c0[plane][ROI_mask] - ref_c0[ROI_mask]) / ref_c0[ROI_mask]))
        actin_inten_ROI_ave.append(np.mean((c1[plane][ROI_mask] - ref_c1[ROI_mask]) / ref_c1[ROI_mask]))
        LIM_inten_ROI_ave_std.append(np.std((c0[plane][ROI_mask] - ref_c0[ROI_mask]) / ref_c0[ROI_mask]))
        actin_inten_ROI_ave_std.append(np.std((c1[plane][ROI_mask] - ref_c1[ROI_mask]) / ref_c1[ROI_mask]))


    # find the peak in the LIM signal
    peak_pt = np.argwhere(LIM_inten_ROI_peak[exp_details['ablation_timepoint']:] == np.max(LIM_inten_ROI_peak[exp_details['ablation_timepoint']:]))
    peak_pt = peak_pt[0][0] + exp_details['ablation_timepoint']
    peak_LIM = np.max(LIM_inten_ROI_peak)
    peak_actin = actin_inten_ROI_peak[peak_pt]

    inten_fig, inten_ax = plt.subplots()
    # plot the LIM intensity in green
    inten_ax.plot(exp_details['time_pts'],LIM_inten_ROI_peak ,'g', label = 'testin ' + str(thresh) + '%')
    # inten_ax.plot(exp_details['time_pts'],LIM_inten_ROI_ave ,'-.g', label = 'testin ROI average')
    # plot the actin intensity in red
    inten_ax.plot(exp_details['time_pts'],actin_inten_ROI_peak ,'r', label = 'actin ' + str(thresh) + '%')
    # inten_ax.plot(exp_details['time_pts'],actin_inten_ROI_ave ,'-.r', label = 'actin ROI average')
    # set the title based on the file name
    inten_ax.set_title(exp_details['file'].split('/')[-1].split('_')[0])
    inten_ax.plot([exp_details['time_pts'][exp_details['ablation_timepoint']],exp_details['time_pts'][exp_details['ablation_timepoint']]], [np.min(actin_inten_ROI_peak)*.9, np.max(LIM_inten_ROI_peak)*1.1],'-.k')
    inten_ax.set_xlabel('Time (s)')
    inten_ax.set_ylabel('Norm. Fluo. Intensity \nFold Increase')
    inten_ax.set_ylim(-0.75, 3.5)
    inten_ax.legend(loc = 'upper right')
    inten_fig.show()

    if save_figure:
        # save the figure
        inten_fig.savefig(exp_details['base_folder'] + exp_details['file'] + '.png', format='png', dpi=300)

    return LIM_inten_ROI_peak, LIM_inten_ROI_ave, peak_LIM, actin_inten_ROI_peak, actin_inten_ROI_ave, peak_actin, peak_pt, LIM_ROI_pts, LIM_overlay_pts, LIM_inten_ROI_peak_std, actin_inten_ROI_peak_std, LIM_inten_ROI_ave_std, actin_inten_ROI_ave_std


def cellmask_threshold(imagename, small_object_size=50, cell_minimum_area=50000, plot_figure=False):
    # check if it's a string or a matrix and read in the image
    if isinstance(imagename, str):
        image = io.imread(imagename)
    else:
        image = imagename
           
    # Find the unique intensity values in the image
    intensity_values = np.unique(image.ravel())

    # reduce list of intensity values down to something manageable to speed up computation
    if len(intensity_values) > 300:
        slice_width = np.round(len(intensity_values)/300).astype('int')
        intensity_values = intensity_values[::slice_width]

    # Find the mean intensity value of the image
    intensity_mean = np.mean(image)

    # create a zero matrix to hold our difference values
    intensity_difference = np.zeros_like(intensity_values).astype('float')

    # for loop to compare the difference between the intensity sum of pixels above a threshold 
    # and the average image intensity of an identical number of pixels
    for i,intensity in enumerate(intensity_values):
        # make a mask of pixels about a given intensity
        mask = image > intensity

        # take the difference between the sum of thresholded pixels and the average value of those pixels
        intensity_difference[i] = np.sum(mask * image) - intensity_mean*np.sum(mask)

    # find the maximum value of the intensity_difference and set it equal to the threshold
    max_intensity = np.argwhere(intensity_difference == np.max(intensity_difference))
    threshold = intensity_values[max_intensity[0][0]]

    # make a mask at this threshold
    mask = image > threshold

    # get rid of small objects
    mask = remove_small_objects(mask, small_object_size)
    # fill any holes in the mask
    mask = morphology.binary_fill_holes(mask)
    
    # plot figure 
    if plot_figure == True:
        # plotting for data confirmation
        mask_fig, mask_axes = plt.subplots()
        mask_axes.imshow(image, cmap='Greys_r', vmin=np.min(image), vmax=np.max(image)*.8)
        mask_axes.imshow(mask, alpha=0.2)
        mask_fig.show()

    return mask

def periodic_decomposition(im):
    '''
    Break an impage into the periodic components to remove artifacts when taking the 2D FFT
    '''
    # find the number of rows and cols
    N_rows, N_cols = im.shape
    # create an zero matrix the size of the image
    v = np.zeros((N_rows,N_cols))
    # fill the edges of V with the difference between the opposite edge of the real image
    v[0,:] = im[0,:] - im[-1,:]
    v[-1,:] = -v[0,:]
    v[:,0] = v[:,0] + im[:,0] - im[:,-1]
    v[:,-1] = v[:,-1] - im[:,0] + im[:,-1]
    # calculate the frequencies of the image
    fx = matlib.repmat(np.cos(2 * np.pi * np.arange(0,N_cols) / N_cols),N_rows,1)
    fy = matlib.repmat(np.cos(2 * np.pi * np.arange(0,N_rows) / N_rows),N_cols,1).T
    # set the fx[0,0] to 0 to avoid division by zero
    fx[0,0] = 0
    # calculate the smoothed image component
    s = np.real(ifft2(fft2(v) * 0.5 / (2 - fx - fy)))
    # If you want to calculate the periodic fft directly
    # p_fft = fftshift(fft2(actin) - fft2(v) * 0.5 / (2 - fx - fy))
    p = im - s

    return p, s

def gaussian_smooth_image(im, sigma):
    shape = im.shape
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h * im

def image_norm(im):
    # Find the image norm
    im_norm = np.sqrt(np.real(im * np.conj(im)))
    return im_norm

def least_moment(image, xcoords=[], ycoords=[]):
    ''' 
    Function to find the least second moment of an image
    '''
    # get the image shape
    N_rows, N_cols = image.shape

    # check if xcoords and ycoords are passed in the function
    if len(xcoords) == 0:
        # create coordinates for the image
        xcoords, ycoords = np.meshgrid(np.arange(0,N_cols) , np.arange(0,N_rows))

    #calculate the moments
    M00 = np.sum(image)
    M10 = np.sum(image * xcoords)
    M01 = np.sum(image * ycoords)
    M11 = np.sum(image * xcoords * ycoords)
    M20 = np.sum(image * xcoords * xcoords)
    M02 = np.sum(image * ycoords * ycoords)

    # center of mass
    xave = M10 / M00
    yave = M01 / M00

    # calculate the central moments
    mu20 = M20/M00 - xave**2
    mu02 = M02/M00 - yave**2
    mu11 = M11/M00 - xave*yave

    # angle of axis
    theta = 0.5 * np.arctan2((2 * mu11),(mu20 - mu02))
    # multiply by -1 to correct for origin being in top left corner instead of bottom right
    theta = -1 * theta
    # find eigenvectors
    lambda1 = (0.5 * (mu20 + mu02)) + (0.5 * np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
    lambda2 = (0.5 * (mu20 + mu02)) - (0.5 * np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
    # calculate the eccentricity (e.g. how oblong it is)
    eccentricity = np.sqrt(1 - lambda2/lambda1)
    
    # correct for real space
    theta = theta + np.pi/2

    return theta, eccentricity

def find_ablation_mask(filename, exp_details):
    '''
    Create a mask of the line used for the ablation
    '''
    # pull out the name of the mask points file
    section = filename[filename.rfind('/')+1:]
    mask_pts_file = filename[:filename.rfind('/')+1] + 'Slide1-' + section[:section.find('_')] + '.txt'
    # read in the mask points file
    mask_pts = pd.read_csv(mask_pts_file,names=['r','c'])
    # make a zero matrix to hold the mask points
    ablation_mask = np.zeros((exp_details['N_rows'], exp_details['N_cols']))
    # populate the ablation matrix
    for index, row in mask_pts.iterrows():
        ablation_mask[row['r'],row['c']] = True
    # crop the ablation mask
    ablation_ROI = ablation_mask[(exp_details['ROI_center_row']-40):(exp_details['ROI_center_row']+41),(exp_details['ROI_center_col']-40):(exp_details['ROI_center_col']+41)]
    # calculate the angle of the line
    ablation_theta, ablation_eccentricity = least_moment(ablation_ROI)
    # calculate the rotation angle
    angle = 90 - ablation_theta*180/np.pi
    
    return ablation_mask, angle

def flat_field_correct_image(image, camera, magnification, channel, def_focus = True):
    '''
    Corrects for field flatness given the camera, laser, objective specs
    '''

    # path where the files are stored
    home = os.path.expanduser("~")
    base_path = home + "/Dropbox/Python_code/Image_correction/"
    # check if the definite focus is engaged
    if def_focus:
        # read in dark image and flat field image
        dark_image = io.imread('Flatfield_correction_images/' + 'Darkfield_' + camera + '_DF_' + channel +'.tif')
        flat_field_image = io.imread('Flatfield_correction_images/' + 'Flatfield_' + camera + '_DF_' + str(magnification) + 'X_' + channel + '.tif')
    else:
        dark_image = io.imread('Flatfield_correction_images/' + 'Darkfield_' + camera + '.tif')
        flat_field_image = io.imread('Flatfield_correction_images/' + 'Flatfield_' + camera + '_' + str(magnification) + 'X_' + channel + '.tif')
        
    # check if the image is a stack
    if len(image.shape) > 2:
        # make an empty matrix to hold the corrected image
        corrected_image = np.zeros_like(image)
        for index, plane in enumerate(image):
            # correct each image
            corrected_image[index] = (plane - dark_image) / flat_field_image
    else:
        # for single plane images
        corrected_image = (image - dark_image) / flat_field_image
        
    return corrected_image

def photobleach_correct(imagestack, time, fit_type = "single", title = '', show_figure = False):
    '''
    Correct for photobleaching by fitting the intensity sum of the image to an exponential decay (single or double)
    Best when used on image stacks that have been corrected already for darkfield contributions
    '''

    # determine whether single or double exponential fit
    if fit_type == "single":
        def photobleach_decay(t, A, B):
            return A * np.exp(-B * t)
    elif fit_type == "double":
        def photobleach_decay(t, A, B, C, D):
            return A * np.exp(-B * t) + C * np.exp(-D * t)
    else:
        raise Exception("Only accepted arguments are 'single' or 'double' ")

    # get image intensities
    image_intensity = []
    for plane in imagestack:
        image_intensity.append(np.sum(plane))

    # make parameter guesses
    if fit_type == "single":
        init_params = [image_intensity[0], .001]
    else:
        init_params = [image_intensity[0], .001, image_intensity[0]/2, 0.001]

    # fit the curve
    fit_params, fit_params_covarianve = optimize.curve_fit(photobleach_decay, time, image_intensity, init_params)

    # make the fitted curve
    if fit_type == "single":
        bleach_fit = photobleach_decay(time, fit_params[0], fit_params[1])
    else:
        bleach_fit = photobleach_decay(time, fit_params[0], fit_params[1], fit_params[2], fit_params[3])

    # make the correction curve
    bleach_correction = bleach_fit[0] / bleach_fit

    # make a new stack to hold our corrected images
    bleach_corrected_imagestack = np.zeros_like(imagestack)

    # correct each plane
    for i, plane in enumerate(imagestack):
        bleach_corrected_imagestack[i] = plane * bleach_correction[i]

    # show the bleach correction figure
    if show_figure == True:
        plt.figure()
        plt.plot(time, image_intensity,'k', label = 'original image intensity')
        plt.plot(time, bleach_fit, 'r', label = 'exponential fit')
        plt.plot(time, image_intensity * bleach_correction, 'b', label = 'corrected image')
        plt.xlabel('Time')
        plt.ylabel('Total Image Intensity')
        plt.legend(loc = 'lower left')
        plt.ylim(0, np.max(image_intensity) * 1.1)
        plt.title(title + ' photobleach correction')
        plt.show()

    return bleach_corrected_imagestack, fit_params, image_intensity, bleach_correction


