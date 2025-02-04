import numpy as np
import pydicom
import copy
from copy import deepcopy
import os
from scipy.ndimage import zoom, gaussian_filter
import matplotlib.pyplot as plt
from numpy.ma import masked_array 
import warnings
warnings.filterwarnings("ignore")

def normalize_by_top_median(img, num=10000):
    #this function normalizes an image by the median of the top num of pixels
    flattened = deepcopy(img).flatten()
    sorted_pixels = np.sort(flattened)[::-1]
    hottest = sorted_pixels[:num]
    med=np.median(hottest)
    return  img / med

def find_bb(image, bounds=[[0,-1],[0,-1]], zoom_factor=3):
    image = (copy.deepcopy(image) - np.amin(image)) / (np.amax(image) - np.amin(image))
    #first crop image to within bounds:
    img = copy.deepcopy(image)[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]]
    min_pixel = np.argmin(img)
    pixel_list = sorted(img.flatten().tolist())
    pixel_100 = pixel_list[50*zoom_factor**2]
    img[img > pixel_100] = 0

    #now find the centre of mass of the remaining pixels 
    centre_pixels = np.nonzero(img)
    centre_of_mass = np.mean(centre_pixels, axis=1)

    #centre_of_mass = np.unravel_index(min_pixel, img.shape)

 
    return [centre_of_mass[0] + bounds[0][0], centre_of_mass[1] + bounds[1][0]]




def lrfc(image_path, zoom_factor=3):
    meta = pydicom.dcmread(image_path)
    jaws_x = meta[0x3002,0x0030][0][0x300A,0x00B6][0][0x300A,0X011C].value
    jaws_y = meta[0x3002,0x0030][0][0x300A,0x00B6][1][0x300A,0X011C].value
    image = meta.pixel_array
    #this function performs lrfc test on the given image, assuming both plates are present (the light field alignment plate and the crosshair plate)
    image = normalize_by_top_median(image)
    image = zoom(image, zoom=zoom_factor, order=3)
    points = {}    #hold the bb location points 

    pixel_distance_conversion = 0.336 / zoom_factor / 1.5

    #find the centre bb location:
    centre_bb = find_bb(image, bounds=[[int(image.shape[0]*4.75/10), int(image.shape[0]*5.25/10)],[int(image.shape[1]*4.75/10), int(image.shape[1]*5.25/10)]], zoom_factor=zoom_factor)
    points["c"] = centre_bb

    #Find the location of jaw edges through the centre of the image.

    y_start = int(image.shape[0] * 3 / 8)
    y_end = int(image.shape[0] * 5 / 8)
    x_start = int(image.shape[1] * 3 / 8)
    x_end = int(image.shape[1] * 5 / 8)
 
    # Extract and average profiles over the middle 1/4 of the image.
    y1_profile = np.mean(image[int(image.shape[0]/2):, x_start:x_end], axis=1)  # Vertical, lower half
    y2_profile = np.mean(image[:int(image.shape[0]/2), x_start:x_end], axis=1)    # Vertical, upper half
    x1_profile = np.mean(image[y_start:y_end, :int(image.shape[1]/2)], axis=0)    # Horizontal, left half
    x2_profile = np.mean(image[y_start:y_end, int(image.shape[1]/2):], axis=0)  # Horizontal, right half

    # Find the location of jaw edges using the averaged profiles.
    y1_pixel = np.argmin(abs(y1_profile - 0.5)) + int(image.shape[0]/2)
    y2_pixel = np.argmin(abs(y2_profile - 0.5))
    x1_pixel = np.argmin(abs(x1_profile - 0.5))
    x2_pixel = np.argmin(abs(x2_profile - 0.5)) + int(image.shape[1]/2)


    points["y1"] = y1_pixel
    points["y2"] = y2_pixel
    points["x1"] = x1_pixel
    points["x2"] = x2_pixel

    #now need position of outside bbs in phantom

    #search in a box inside from the corners of jaws:
    box_length = int(6/pixel_distance_conversion )    #size of box used to look for bb from jaw corner

    #top left corner:
    corner_jaw = [points["y2"], points["x1"]]
    bounds = [[(corner_jaw[0]+int(1.15*box_length)), (corner_jaw[0]+int(2.15*box_length))],[(corner_jaw[1]+int(box_length*1.15)),(corner_jaw[1]+int(2.15*box_length))]]   #now take box of pixels around this region.
    top_left_bb = find_bb(image, bounds=bounds)

    # plt.imshow(image)
    # plt.show()

    #top right corner:
    corner_jaw = [points["y2"], points["x2"]]
    bounds = [[(corner_jaw[0]+int(box_length*1.15)),(corner_jaw[0]+int(2.15*box_length))], [(corner_jaw[1]-int(2.15*box_length)), (corner_jaw[1]-int(box_length*1.115))]]   #now take box of pixels around this region.
    top_right_bb = find_bb(image, bounds=bounds)

    #Bottom Left corner:
    corner_jaw = [points["y1"], points["x1"]]
    bounds = [[(corner_jaw[0]-int(2.15*box_length)), (corner_jaw[0]-int(box_length*1.15))],[(corner_jaw[1]+int(box_length*1.15)),(corner_jaw[1]+int(2.15*box_length))]]   #now take box of pixels around this region.
    bottom_left_bb = find_bb(image, bounds=bounds)

    #Bottom Right corner:
    corner_jaw = [points["y1"], points["x2"]]
    bounds = [[(corner_jaw[0]-int(2.15*box_length)), (corner_jaw[0]-int(box_length*1.15))],[(corner_jaw[1]-int(2.15*box_length)), (corner_jaw[1]-int(box_length*1.15))]]   #now take box of pixels around this region.
    bottom_right_bb = find_bb(image, bounds=bounds)

    points["bb_tl"] = top_left_bb
    points["bb_tr"] = top_right_bb
    points["bb_bl"] = bottom_left_bb
    points["bb_br"] = bottom_right_bb

    #Now want to update our jaw positions by averaging over profiles taken on each side of jaw, rather than just centre image
    y2_left_profile = image[:int(image.shape[0]/2), int(top_left_bb[1])]
    y2_right_profile = image[:int(image.shape[0]/2), int(top_right_bb[1])]
    y1_left_profile = image[int(image.shape[0]/2):, int(bottom_left_bb[1])]   
    y1_right_profile = image[int(image.shape[0]/2):, int(bottom_right_bb[1])]  
    x1_top_profile = image[int(top_left_bb[0]), :int(image.shape[1]/2)]  
    x1_bottom_profile = image[int(bottom_left_bb[0]), :int(image.shape[1]/2)]    
    x2_top_profile = image[int(top_right_bb[0]), int(image.shape[1]/2):]  
    x2_bottom_profile = image[int(bottom_right_bb[0]), int(image.shape[1]/2):]
    
    # Find the location of jaw edges using the averaged profiles.
    y1_pixel = int(0.5*(np.argmin(abs(y1_left_profile - 0.5)) + np.argmin(abs(y1_right_profile - 0.5)))+ image.shape[0]/2)
    y2_pixel = int(0.5*(np.argmin(abs(y2_left_profile - 0.5)) + np.argmin(abs(y2_right_profile - 0.5)))) 
    x1_pixel = int(0.5*(np.argmin(abs(x1_top_profile - 0.5))  +np.argmin(abs(x1_bottom_profile - 0.5))))
    x2_pixel = int(0.5*(np.argmin(abs(x2_top_profile - 0.5)) + np.argmin(abs(x2_bottom_profile - 0.5))) + image.shape[1]/2)

    #make a figure showing defined edges/BBs
    img = deepcopy(image)
    img[-20+ int(centre_bb[0]):20+int(centre_bb[0]), -20+ int(centre_bb[0]):20+int(centre_bb[0])] = -1
    img[-20 + int(top_left_bb[0]):20 + int(top_left_bb[0]), -20 + int(top_left_bb[1]): 20 + int(top_left_bb[1])] = -1
    img[-20 + int(top_right_bb[0]):20 + int(top_right_bb[0]), -20 + int(top_right_bb[1]): 20 + int(top_right_bb[1])] = -1
    img[-20 + int(bottom_left_bb[0]):20 + int(bottom_left_bb[0]), -20 + int(bottom_left_bb[1]): 20 + int(bottom_left_bb[1])] = -1
    img[-20 + int(bottom_right_bb[0]):20 + int(bottom_right_bb[0]), -20 + int(bottom_right_bb[1]): 20 + int(bottom_right_bb[1])] = -1

    img[-10 + int(y1_pixel):10 + int(y1_pixel), int(img.shape[0]/3):int(img.shape[0]*2/3)] = -1
    img[-10 + int(y2_pixel):10 + int(y2_pixel), int(img.shape[0]/3):int(img.shape[0]*2/3)] = -1
    img[int(img.shape[0]/3):int(img.shape[0]*2/3), -10 + int(x1_pixel):10 + int(x1_pixel)] = -1
    img[int(img.shape[0]/3):int(img.shape[0]*2/3), -10 + int(x2_pixel):10 + int(x2_pixel)] = -1

    
    masked_negative = masked_array(img, mask=img >= 0)



    

    #now calculate the radiation centre
    rad_centre_y = (y1_pixel + y2_pixel) / 2
    rad_centre_x = (x2_pixel + x1_pixel) / 2
    rad_centre = [rad_centre_y, rad_centre_x]

    rad_displacement = ([pixel_distance_conversion*(rad_centre[i] - centre_bb[i]) for i in range(2)]) 

    points["rad_disp"] = rad_displacement

    #now get light field crosshair displacement.
    displacement_top_bbs = (centre_bb[0] - (top_right_bb[0] + top_left_bb[0])/2) * pixel_distance_conversion
    displacement_bottom_bbs = ((bottom_left_bb[0] + bottom_right_bb[0])/2 - centre_bb[0]) * pixel_distance_conversion
    displacement_left_bbs = (centre_bb[1] - (bottom_left_bb[1] + top_left_bb[1])/2) * pixel_distance_conversion
    displacement_right_bbs = ((bottom_right_bb[1] + top_right_bb[1])/2 - centre_bb[1]) * pixel_distance_conversion

    pos_top_bbs = (top_right_bb[0] + top_left_bb[0])/2
    pos_bottom_bbs = (bottom_left_bb[0] + bottom_right_bb[0])/2
    pos_left_bbs = (bottom_left_bb[1] + top_left_bb[1])/2
    pos_right_bbs = (bottom_right_bb[1] + top_right_bb[1])/2

    field_size = 10 if int((displacement_bottom_bbs + displacement_top_bbs)/10 + 2) < 12.5 else 15    #assuming we do 10 or 15 for field size.
    points["field_size"] = field_size
    #Get light field centre and displacement from crosshair
    light_centre_y = (pos_top_bbs+pos_bottom_bbs) / 2
    light_centre_x = (pos_left_bbs + pos_right_bbs) / 2
    light_centre = [light_centre_y, light_centre_x]

    light_disp_y = (light_centre_y - centre_bb[0])*pixel_distance_conversion
    light_disp_x = (light_centre_x - centre_bb[1]) * pixel_distance_conversion
    
    light_disp = [light_disp_y, light_disp_x]

    points["light_disp"] = light_disp

    #compare light field centre with radiation centre 
    rad_light_disp = np.array(rad_displacement) - np.array(light_disp)

    points["rad_light_disp"] = rad_light_disp 


    #find jaw displacements from light field centre: 
    # x1 = abs(x1_pixel - light_centre[1])*pixel_distance_conversion
    # x2 = abs(x2_pixel - light_centre[1])*pixel_distance_conversion
    # y1 = abs(y1_pixel - light_centre[0])*pixel_distance_conversion
    # y2 = abs(y2_pixel - light_centre[0])*pixel_distance_conversion
    x1 = abs(x1_pixel - centre_bb[1])*pixel_distance_conversion
    x2 = abs(x2_pixel - centre_bb[1])*pixel_distance_conversion
    y1 = abs(y1_pixel - centre_bb[0])*pixel_distance_conversion
    y2 = abs(y2_pixel - centre_bb[0])*pixel_distance_conversion

    #express jaw positions as deviations from nominal values
    y1_disp = y1 - field_size*10/2
    y2_disp = y2 - field_size*10/2
    x1_disp = x1 - field_size*10/2
    x2_disp = x2 - field_size*10/2

    light_fs_y = (pos_bottom_bbs - pos_top_bbs)*pixel_distance_conversion + 20
    light_fs_x = (pos_right_bbs - pos_left_bbs)*pixel_distance_conversion + 20


    points["jaw_disps"] = [y1_disp, y2_disp, x1_disp, x2_disp]

    #make plot of results

    return points


if __name__ == "__main__":
    #image_path = os.path.join(os.getcwd(),"u4_lrfc_nov4","6101515106_1510-000.dcm")
    #image_path = os.path.join("U:\Profile\Desktop\Abbotsford_Physics_Residency\Projects\lrfc_winlutz_flood\lrfc_u2","-7.5_7.5_-7.5_7.5.dcm")

    # plt.imshow(image)
    # plt.show()

    for unit in range(1,5):
        print(f"Unit {unit}:")
        for folder in [os.path.join(os.getcwd(), "pipspro_comparisons", f"u{unit}_lrfc_dec24"),os.path.join(os.getcwd(), "pipspro_comparisons", f"u{unit}_lrfc_nov24")]:
            print(folder)
            for energy in sorted(os.listdir(folder)):
                if energy != "6x":
                    continue
                print(f"Energy: {energy}")
                for file in os.listdir(os.path.join(folder, energy)):
                    if "10" not in file:
                        continue
                    print(f"File: {file}")
                    image_path = os.path.join(folder, energy, file)
                    print(file)
                    points, fig, fig2 = lrfc(image_path, zoom_factor=3)
                    plt.show()
