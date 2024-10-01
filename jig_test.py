import pydicom
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import csv
import stat
import math
from scipy.ndimage import zoom, gaussian_filter

def get_bb_and_jaw_location(image, open_field=False):
    img_size = image.shape[0]
    if open_field == True:    #in this case there are two bbs in vertical column at centre axis
        #first find top bb:
        img = deepcopy(image)
        img[:,:round(4*img_size/9)] = 1
        img[:,round(5*img_size/9):] = 1
        img[:round(img_size/5),:] = 1
        img[round(4*img_size/5):,:] = 1

        top_img = deepcopy(img)
        bottom_img = deepcopy(img)

        top_img[round(img_size/2):,:] = 1
        bottom_img[:round(img_size/2),:] = 1



        pixel_list = sorted(top_img.flatten().tolist())   #define bb by minimum 500 pixels centre of mass
        pixel_500 = pixel_list[500]
        top_img[top_img > pixel_500] = 0

        pixel_list = sorted(bottom_img.flatten().tolist())
        pixel_500 = pixel_list[500]
        bottom_img[bottom_img > pixel_500] = 0

        top_location = np.mean(np.nonzero(top_img), axis=1)
        bottom_location = np.mean(np.nonzero(bottom_img), axis=1)

        # fig, ax = plt.subplots(ncols=2)
        # ax[0].imshow(top_img)
        # ax[1].imshow(bottom_img)

        # plt.show()

        return tuple((top_location, bottom_location))
    
    elif open_field == False:    #in this case there is one BB at bottom/top. Also want to return the central jaw location
        img = deepcopy(image)
        #first determine whether bb is at top or bottom:
        if np.mean(img[:round(img_size/2),:]) > np.mean(img[round(img_size/2):,:]):    #image is at top half
            #first get the jaw location
            profile = img[round(3*img_size/7):,round(img_size/2)]
            grad_profile = np.gradient(profile)
            jaw_pixel = np.argmin(grad_profile) + round(3*img_size/7)

            #now filter for finding find the bb
            img[:,:round(4*img_size/9)] = 1
            img[:,round(5*img_size/9):] = 1
            img[:round(img_size/5),:] = 1
            img[round(2*img_size/5):,:] = 1

            


        else:    #if image at bottom half
            #first get the jaw location
            profile = img[:round(4*img_size/7),round(img_size/2)]
            grad_profile = np.gradient(profile)
            jaw_pixel = np.argmax(grad_profile)
            #now filter for finding find the bb
            img[:,:round(4*img_size/9)] = 1
            img[:,round(5*img_size/9):] = 1
            img[:round(3*img_size/5),:] = 1
            img[round(4*img_size/5):,:] = 1

            

        pixel_list = sorted(img.flatten().tolist())   #define bb by minimum 500 pixels centre of mass
        pixel_500 = pixel_list[500]
        img[img > pixel_500] = 0

        bb_location = np.mean(np.nonzero(img), axis=1)

        # plt.imshow(image)
        # plt.show(block=True)

        return bb_location, jaw_pixel
    
    
        

        

def get_jaw_matching_jig_data(images):

    jaw_matching_qa_data = {
        "sc1_open": None,
        "sc2_0": None,
        "rt_cw_230": None,
        "lt_cw_310": None,
        "rt_cw_50": None,
        "lt_cw_130": None,
        "sc3_180": None

    }

    #now get the bb distances for closed fields
    for gantry in images.keys():
        for coll in images[gantry].keys():
            for blocked_field in images[gantry][coll].keys():
                if blocked_field == "open":
                    #get the two bbs in open field
                    bb_locations = get_bb_and_jaw_location(images[gantry][coll][blocked_field], open_field=True)

                    #get distance between BBs:
                    bb_dist = abs(bb_locations[1] - bb_locations[0])*0.336/2 / 1.5
                    bb_dist = np.sqrt(bb_dist[0]**2 + bb_dist[1]**2)
                    jaw_matching_qa_data["sc1_open"] = bb_dist
                else:
                    bb_location, jaw_location = get_bb_and_jaw_location(images[gantry][coll][blocked_field], open_field=False)

                    #need distance from bb to jaw
                    dist = abs(jaw_location - bb_location[0])*0.336/2 / 1.5

                    #now add to qa data dictionary 
                    for key in jaw_matching_qa_data.keys():
                        if str(gantry) in key:
                            jaw_matching_qa_data[key] = dist

    #make sure all images needed for QA were found
    for key in jaw_matching_qa_data.keys():
        if jaw_matching_qa_data[key] is None:
            raise Exception(f"{key} image not found, unable to proceed with QA.")
    #now need to calculate the junctions
    junctions = {}
    junctions["j_0_310"] = jaw_matching_qa_data["sc2_0"] + jaw_matching_qa_data["lt_cw_310"] - jaw_matching_qa_data["sc1_open"]
    junctions["j_0_130"] = jaw_matching_qa_data["sc2_0"] + jaw_matching_qa_data["lt_cw_130"] - jaw_matching_qa_data["sc1_open"]

    junctions["j_180_310"] = jaw_matching_qa_data["sc3_180"] + jaw_matching_qa_data["lt_cw_310"] - jaw_matching_qa_data["sc1_open"]
    junctions["j_180_130"] = jaw_matching_qa_data["sc3_180"] + jaw_matching_qa_data["lt_cw_130"] - jaw_matching_qa_data["sc1_open"]

    junctions["j_0_50"] = jaw_matching_qa_data["sc2_0"] + jaw_matching_qa_data["rt_cw_50"] - jaw_matching_qa_data["sc1_open"]
    junctions["j_0_230"] = jaw_matching_qa_data["sc2_0"] + jaw_matching_qa_data["rt_cw_230"] - jaw_matching_qa_data["sc1_open"]

    junctions["j_180_50"] = jaw_matching_qa_data["sc3_180"] + jaw_matching_qa_data["rt_cw_50"] - jaw_matching_qa_data["sc1_open"]
    junctions["j_180_230"] = jaw_matching_qa_data["sc3_180"] + jaw_matching_qa_data["rt_cw_230"] - jaw_matching_qa_data["sc1_open"]

    juncs = []
    for junc in junctions.keys():
        juncs.append(np.abs(junctions[junc]))
        median_junc = np.median(juncs)

    return jaw_matching_qa_data, junctions


def sort_image_dict(img_folder : str):
    #first load images into a dictionary based on gantry angle and collimator angle
    imgs = {}    #initiate the image dictionary
    if len(os.listdir(img_folder)) == 0:
        raise Exception(f"No Images found in {img_folder}")
    
    #go through the image directory and sort and store images
    for img_path in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_path)
        img_meta = pydicom.dcmread(img_path)
        img = img_meta.pixel_array
        img[:20,:] = 0
        img[-20:,:] = 0
        img[:, :20] = 0
        img[:,-20:] = 0

        img = img / np.percentile(img, 90)    #normalize image
        img = gaussian_filter(img, sigma=1, order=0)    #smoothen the image
        img = zoom(img, zoom=2, order=1)
        # jaws = img_meta[0x3002,0x0030].value[0][0x300A, 0x00B6][0]
        

        gantry_angle = round(float(img_meta[0x300A,0x011E].value)) % 360
        if gantry_angle not in imgs.keys():
            imgs[gantry_angle] = {}

        coll_angle = round(float(img_meta[0x300A, 0x0120].value)) % 360
        if coll_angle not in imgs[gantry_angle].keys():
            imgs[gantry_angle][coll_angle] = {}

        imager_location = round(float(img_meta[0x3002, 0x000D].value[2]))    #get location of EPID receptor panel


        
        #collimator positions not included in metadata, so determine closed jaw from lowest mean pixel intensity in each quarter blocked region
        y_range, x_range = img.shape
        #order of quarter regions is: [left, right, top, bottom]
        mean_blocked_pixels = [np.mean(img[:, :int(x_range/2)]), np.mean(img[:, int(x_range/2):]), np.mean(img[:int(x_range/2), :]), np.mean(img[int(x_range/2):, :])]    #C0: x1, x2, y2, y1 / C90: y2, y1, x2, x1 
        
        min_region_index = np.argmin(mean_blocked_pixels)
        blocked_field = ""

        if coll_angle != 90:
            print("Skipping image taken with collimator not at 90 degrees...")
            continue

        if coll_angle == 0:

            if min_region_index == 0:
                blocked_field = "x1"
            if min_region_index == 1:
                blocked_field = "x2"
            if min_region_index == 2:
                blocked_field = "y2"
            if min_region_index == 3:
                blocked_field = "y1"

        elif coll_angle == 90:

            if min_region_index == 0:
                blocked_field = "y2"
            if min_region_index == 1:
                blocked_field = "y1"
            if min_region_index == 2:
                blocked_field = "x2"
            if min_region_index == 3:
                blocked_field = "x1"

        #now if at gantry 0, need to store the open field image of jig separately. 
        if gantry_angle == 0:
            if np.amax(mean_blocked_pixels)/np.amin(mean_blocked_pixels) < 2:
                blocked_field = "open"

        imgs[gantry_angle][coll_angle][blocked_field] = img 

    return imgs


unit_num = 2
img_folder = "U:/Profile/Desktop/Abbotsford_Physics_Residency/Projects/Jaw Matching/Caleb/jig_tests/u2_sep27_post"


#first collect imgs:
img_dict = sort_image_dict(img_folder)

bb_dists, junctions = get_jaw_matching_jig_data(img_dict)