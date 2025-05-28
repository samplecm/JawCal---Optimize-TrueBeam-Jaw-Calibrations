import pydicom
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import csv
import stat
import math
from scipy.ndimage import zoom, gaussian_filter
import datetime
from lrfc_test import lrfc
def find_half_intensity_pixel(array):
    #This function takes a 1 or 2d array, and will find the interpolated 0.5 pixel value index along each row or column (the shortest axis)
    #and then return the average value. 
    #For example, if I have a 1d array, it'll just return the interpolated 0.5 value index. 
    #If i have a 100 x 3000 array, I will find the 0.5 index along the 100 rows and return the average value.
    im_shape = array.shape
    if len(im_shape) == 1:
        return np.interp(0.5, np.arange(len(array)), array)
    elif len(im_shape) == 2:
        if im_shape[0] < im_shape[1]:
            vals = []
            for i in range(im_shape[0]):
                ind = np.argmin(abs(array[i,:] - 0.5))
                if array[i,ind] < 0.5:
                    if array[i,ind+1] > 0.5:
                        vals.append(ind + (0.5-array[i,ind])/(array[i,ind+1]-array[i,ind])) 
                    elif array[i,ind-1] > 0.5:
                        vals.append(ind-1+(0.5-array[i,ind-1])/(array[i,ind]-array[i,ind-1]))
                elif array[i,ind] > 0.5:
                    if array[i,ind+1] < 0.5:
                        vals.append(ind+(0.5-array[i,ind])/(array[i,ind+1]-array[i,ind]))
                    elif array[i,ind-1] < 0.5:
                        vals.append(ind-1+(0.5-array[i,ind-1])/(array[i,ind]-array[i,ind-1]))
            return np.mean(vals)

        elif im_shape[1] < im_shape[0]:
            vals = []
            for i in range(im_shape[1]):
                ind = np.argmin(abs(array[:,i] - 0.5))
                if array[ind,i] < 0.5:
                    if array[ind+1,i] > 0.5:
                        vals.append(ind+(0.5-array[ind,i])/(array[ind+1,i]-array[ind,i]))
                    elif array[ind-1,i] > 0.5:
                        vals.append(ind-1+(0.5-array[ind-1,i])/(array[ind,i]-array[ind-1,i]))
                elif array[ind,i] > 0.5:
                    if array[ind+1,i] < 0.5:
                        vals.append(ind+(0.5-array[ind,i])/(array[ind+1,i]-array[ind,i]))
                    elif array[ind-1,i] < 0.5:
                        vals.append(ind-1+(0.5-array[ind-1,i])/(array[ind,i]-array[ind-1,i]))
            return np.mean(vals)
    
def define_encoder_dict(unit=2, date=None):
    #this function initializes the dictionary of jaw positions --> encoder values. 
    #it reads an encoder csv file and stores the encoder values in a dictionary with their approximate (machine - read) symmetric jaw value
    encoder_file = os.path.join(os.getcwd(), "encoder_spreadsheets", f"u{unit}_encoders_{date}.csv")
    with open(encoder_file) as fp:
        reader = csv.reader(fp)
        csv_data_list = []
        for row in reader:
            csv_data_list.append(row)
    dic = {"x1": {}, "x2": {}, "y1": {}, "y2": {}}
    for row in csv_data_list[2:]:
        dic["x1"][round(float(row[0]),1)] = {"encoder": int(row[1]), "pixel": -100}
        dic["x2"][round(float(row[0]),1)] = {"encoder": int(row[2]), "pixel": -100}
        dic["y1"][round(float(row[0]),1)] = {"encoder": int(row[3]), "pixel": -100}
        dic["y2"][round(float(row[0]),1)] = {"encoder": int(row[4]), "pixel": -100}
    
    return dic

def round_to_point_five(x):
    #rounds value to nearest 0.5
    return round(round(2*x)/2,1)

def normalize_by_top_median(img, num=20000):
    #this function normalizes an image by the median of the top num of pixels
    flattened = deepcopy(img).flatten()
    sorted_pixels = np.sort(flattened)[::-1]
    hottest = sorted_pixels[20000:20000+num]    #don't use hottest 20,000 pixels (there can be errors with broken pixels)
    med=np.median(hottest)
    return  img / med

def which_jaw_measuring(jaws_x, jaws_y):
    #this function determines which of the 4 jaws is the one being measured for encoder positions. This is determined by finding the one jaw who isn't at the default position of 12
    if round(abs(jaws_x[0])) != 120:
        return "x1"
    elif round(abs(jaws_x[1])) != 120:
        return "x2"
    if round(abs(jaws_y[0])) != 120:
        return "y1"
    elif round(abs(jaws_y[1])) != 120:
        return "y2"

def fit_encoder_vs_pixel_funcs(date, img_folder, iso_img_path, unit_num, optimal_cal,epid_position=1.086):
    #this function finds the epid pixels corresponding to each jaw position in img_dict, and then fits a curve to those pixel values with the jaw encoder readouts

    encoder_dic = define_encoder_dict(unit_num, date)   #initialize dictionary which will hold jaw positions, encoders, pixels
    #get iso img
    img_meta = pydicom.dcmread(iso_img_path)
    img = img_meta.pixel_array
    img = normalize_by_top_median(img)   #normalize image
    img = gaussian_filter(img, sigma=3, order=0)    #smoothen the image
    iso_img = zoom(img, zoom=3, order=3)
    iso = find_bead_location(iso_img, round_final=True, zoom_size=3)    #first get the pixel position of the isocentre

    for img_path in sorted(os.listdir(img_folder)):
        
        img_path = os.path.join(img_folder, img_path)
        img_meta = pydicom.dcmread(img_path)
        jaws_x = img_meta[0x3002,0x0030][0][0x300A,0x00B6][0][0x300A,0X011C].value
        jaws_y = img_meta[0x3002,0x0030][0][0x300A,0x00B6][1][0x300A,0X011C].value
        img = img_meta.pixel_array
        img = normalize_by_top_median(img)
        img = gaussian_filter(img, sigma=3, order=0)    #smoothen the image
        img = zoom(img, zoom=3, order=3)

        current_jaw = which_jaw_measuring(jaws_x, jaws_y)
        if current_jaw == "x1":
        #x1:np.mean(np.argmin(abs(x2_profile - 0.5), axis=0))
            x1_profile = deepcopy(img[iso[0]-100:iso[0]+100, 0:2250])
            # plt.plot(x1_profile[50,:])
            # plt.show()
            #determine centre as pixel with sharpest gradient
            x1_pixel = find_half_intensity_pixel(x1_profile)
            x1_displacement = round_to_point_five(round(abs(jaws_x[0])/10,1))#round((round(-4*(x1_pixel - iso[1]) * pixel_distance/2)/2),1)  #--> cm bc make negative to follow sign convention (positive if jaw crosses iso, negative if shy)     
            encoder_dic["x1"][x1_displacement]["pixel"] = x1_pixel

        elif current_jaw == "x2":
            #x2:
            x2_profile = deepcopy(img[iso[0]-100:iso[0]+100, 1500:-1])
            #determine centre as pixel with sharpest gradient
            x2_pixel = find_half_intensity_pixel(x2_profile)+1500
            x2_displacement = round_to_point_five(round(abs(jaws_x[1])/10,1))#round((round(4*(x2_pixel - iso[1]) * pixel_distance/2)/2),1)   #--> cm bc make negative to follow sign convention (positive if jaw crosses iso, negative if shy)     
            encoder_dic["x2"][x2_displacement]["pixel"] = x2_pixel

        if current_jaw == "y1":
            #y1:
            y1_profile = deepcopy(img[1500:-1, iso[1]-100:iso[1]+100])
            #determine centre as pixel with sharpest gradient
            y1_pixel = find_half_intensity_pixel(y1_profile) +1500
            y1_displacement = round_to_point_five(round(abs(jaws_y[0])/10,1))#round((round(4*(y1_pixel - iso[0]) * pixel_distance/2)/2),1)   #--> cm bc make negative to follow sign convention (positive if jaw crosses iso, negative if shy)     
            encoder_dic["y1"][y1_displacement]["pixel"] = y1_pixel

        if current_jaw == "y2":
            #y2:
            y2_profile = deepcopy(img[0:2250, iso[1]-100:iso[1]+100])
            #determine centre as pixel with sharpest gradient
            y2_pixel = find_half_intensity_pixel(y2_profile)
            y2_displacement = round_to_point_five(round(abs(jaws_y[1])/10,1))#round((round(-4*(y2_pixel - iso[0]) * pixel_distance/2)/2),1)   #--> cm bc make negative to follow sign convention (positive if jaw crosses iso, negative if shy)            
            encoder_dic["y2"][y2_displacement]["pixel"] = y2_pixel


    
    #now want to fit cubic functions of each jaws pixel vs encoder value. We will fit a curve in region around jaw=1cm, jaw=19 cm, and jaw = 5-9 cm
    fig, ax = plt.subplots(nrows=4, ncols=1)
    for j, jaw in enumerate(["x1", "x2", "y1", "y2"]):

        encoders_low = []  #encoders in the region around 1 cm 
        pixels_low = []

        encoders_mid = [] #encoders in the region around 5-9 cm
        pixels_mid = []

        encoders_high = [] #encoders in the region around 19 cm.
        pixels_high = []

        encoders = []  #list of all encoder values

        for val in encoder_dic[jaw].keys():    #go through all encoder values/jaw positions from spreadsheet and find the pixel location of jaw for each. Sort by location.
            
            if encoder_dic[jaw][val]["pixel"] == -100:   #if still at default value, image wasn't found at this position
                continue

            encoders.append(encoder_dic[jaw][val]["encoder"]) 

            if float(val) <= 2:
                encoders_low.append(encoder_dic[jaw][val]["encoder"])
                pixels_low.append(encoder_dic[jaw][val]["pixel"])

            elif float(val) <= 17:
                encoders_mid.append(encoder_dic[jaw][val]["encoder"])
                pixels_mid.append(encoder_dic[jaw][val]["pixel"])

            else: 
                encoders_high.append(encoder_dic[jaw][val]["encoder"])
                pixels_high.append(encoder_dic[jaw][val]["pixel"])

        encoders_low = np.array(encoders_low)
        pixels_low = np.array(pixels_low)

        encoders_mid = np.array(encoders_mid)
        pixels_mid = np.array(pixels_mid)

        encoders_high = np.array(encoders_high)
        pixels_high = np.array(pixels_high)

        #now want a cubic fit to the data:
        fit_low = np.polyfit(pixels_low,encoders_low,deg=3)
        fit_mid = np.polyfit(pixels_mid,encoders_mid,deg=1)
        fit_high = np.polyfit(pixels_high,encoders_high,deg=3)

        print(f"Polynomial fit coefficients low: {fit_low}")
        print(f"Polynomial fit coefficients mid: {fit_mid}")
        print(f"Polynomial fit coefficients high: {fit_high}")

        #make arrays to plot the curves with:
        pixels_fit_low = np.linspace(np.amin(pixels_low), np.amax(pixels_low),300)
        pixels_fit_mid = np.linspace(np.amin(pixels_mid), np.amax(pixels_mid),300)
        pixels_fit_high = np.linspace(np.amin(pixels_high), np.amax(pixels_high),300)

        #define the fit points for plotting.
        fit_points_low = fit_low[0]*pixels_fit_low**3 + fit_low[1]*pixels_fit_low**2 + fit_low[2]*pixels_fit_low + fit_low[3]
        fit_points_mid = fit_mid[0]*pixels_fit_mid + fit_mid[1]
        fit_points_high = fit_high[0]*pixels_fit_high**3 + fit_high[1]*pixels_fit_high**2 + fit_high[2]*pixels_fit_high + fit_high[3]

        ax[j].scatter(pixels_low, encoders_low, c="salmon")
        ax[j].plot(pixels_fit_low, fit_points_low, marker=None, c="mediumturquoise")

        ax[j].scatter(pixels_mid, encoders_mid, c="violet")
        ax[j].plot(pixels_fit_mid, fit_points_mid, marker=None, c="mediumturquoise")

        ax[j].scatter(pixels_high, encoders_high, c="lightgreen")
        ax[j].plot(pixels_fit_high, fit_points_high, marker=None, c="mediumturquoise")

        ax[j].set_xlabel("EPID Pixel")
        ax[j].set_ylabel(f"{jaw} Jaw Encoder Value")

        #also get the predicted location of the locations 1,5,9,19 using the optimal calibration point as the origin
        iso = find_bead_location(iso_img, round_final=False, zoom_size=3) #get unrounded iso
        fits = [fit_low, fit_mid, fit_high]
        p1, p5, p9, p19 = predict_opt_cal_locations(iso, jaw, optimal_cal, fits, epid_position=epid_position)
        if j == 0:
            ax[j].title.set_text(f"p1: {p1}, p9: {p9}, p19: {p19}")
        if j == 1:
            ax[j].title.set_text(f"p1: {p1}, p9: {p9}, p19: {p19}")
        if j == 2:
            ax[j].title.set_text(f"p1: {p1}, p5: {p5}, p19: {p19}")
        if j == 3:
            ax[j].title.set_text(f"p1: {p1}, p5: {p5}, p19: {p19}")

    fig.savefig(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"encoder_plots_cubic"))
    # plt.show()
        
    return

def predict_opt_cal_locations(iso, jaw, optimal_cal, fits, epid_position=1.18):
    pixel_distance = 0.336 / epid_position / 3
    #this function returns the encoder values of the jaw calibration positions (1,5,9,19) calculated relative to optimal origin
    fit_low, fit_mid, fit_high = fits
    #first get the jaw positions in pixels, relative to origin.
    if jaw == "x1":
        origin = iso[1] - optimal_cal[0]/(pixel_distance)   #optimal_cal in millimetres
        p1 = origin - (10/pixel_distance)
        p5 = origin - 50*(1/pixel_distance)
        p9 = origin - 90*(1/pixel_distance)
        p19 = origin - 190*(1/pixel_distance)
    if jaw == "x2":
        origin = iso[1] + optimal_cal[1]/(pixel_distance) 
        p1 = origin + (10/pixel_distance)
        p5 = origin + 50*(1/pixel_distance)
        p9 = origin + 90*(1/pixel_distance)
        p19 = origin + 190*(1/pixel_distance)
    if jaw == "y1":
        origin = iso[0] + optimal_cal[2]/(pixel_distance) 
        p1 = origin + (10/pixel_distance)
        p5 = origin + 50*(1/pixel_distance)
        p9 = origin + 90*(1/pixel_distance)
        p19 = origin + 190*(1/pixel_distance)
    if jaw == "y2":
        origin = iso[0] - optimal_cal[3]/(pixel_distance) 
        p1 = origin - (10/pixel_distance)
        p5 = origin - 50*(1/pixel_distance)
        p9 = origin - 90*(1/pixel_distance)
        p19 = origin - 190*(1/pixel_distance)

    p1 = fit_low[0]*p1**3 + fit_low[1]*p1**2 + fit_low[2]*p1 + fit_low[3]
    p5 = fit_mid[0]*p5 + fit_mid[1]
    p9 = fit_mid[0]*p9 + fit_mid[1]
    p19 = fit_high[0]*p19**3 + fit_high[1]*p19**2 + fit_high[2]*p19 + fit_high[3]


    return [round(p1), round(p5), round(p9), round(p19)]

def sort_junc_img_dict(img_folder : str):
    #first load images into a dictionary based on gantry angle and collimator angle
    imgs = {}    #initiate the image dictionary

    #go through the image directory and sort and store images
    for img_path in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_path)
        img_meta = pydicom.dcmread(img_path)
        img = img_meta.pixel_array
        img = normalize_by_top_median(img)
        img = gaussian_filter(img, sigma=3, order=0)    #smoothen the image
        img = zoom(img, zoom=2, order=3)

        # plt.imshow(img, cmap="bone")
        # plt.show()

        # jaws = img_meta[0x3002,0x0030].value[0][0x300A, 0x00B6][0]
        
        gantry_angle = round(float(img_meta[0x300A,0x011E].value)) % 360

        if gantry_angle not in imgs.keys():
            imgs[gantry_angle] = {}

        coll_angle = round(float(img_meta[0x300A, 0x0120].value)) % 360

        if coll_angle not in imgs[gantry_angle].keys():
            imgs[gantry_angle][coll_angle] = {}

        imager_location = round(float(img_meta[0x3002, 0x000D].value[2]))    #get location of EPID receptor panel

        #old method (when position 0 was used for other images)
        #
        #
        #if epid location is at -500, image is of the cube phantom with the bb to demark isocentre
        # if imager_location == -500:
        #     imgs[gantry_angle]["iso"] = img   #using same image for both coll rotations... isocentre defined by bb over epid
        #     continue
        #
        #
        
        #collimator positions not included in metadata, so determine closed jaw from lowest mean pixel intensity in each quarter blocked region
        y_range, x_range = img.shape
        #order of quarter regions is: [left, right, top, bottom]
        mean_blocked_pixels = [np.mean(img[:, :int(x_range/2)]), np.mean(img[:, int(x_range/2):]), np.mean(img[:int(x_range/2), :]), np.mean(img[int(x_range/2):, :])]    #C0: x1, x2, y2, y1 / C90: y2, y1, x2, x1 
        
        min_region_index = np.argmin(mean_blocked_pixels)
        
        #if field is symmetric, then it is the isocentre image (no closed jaws)
        if (np.amin(mean_blocked_pixels)/np.amax(mean_blocked_pixels)) > 0.6: 
            imgs[gantry_angle]["iso"] = img
            #find_bead_location(img)
            continue

        blocked_field = ""

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

        elif coll_angle == 270:
            if min_region_index == 0:
                blocked_field = "y1"
            if min_region_index == 1:
                blocked_field = "y2"
            if min_region_index == 2:
                blocked_field = "x1"
            if min_region_index == 3:
                blocked_field = "x2"



        imgs[gantry_angle][coll_angle][blocked_field] = img

    return imgs

def sort_jaw_img_dict(img_folder : str):

    imgs = {}    #initiate the image dictionary
    imgs["x1"] = {}
    imgs["x2"] = {}
    imgs["y1"] = {}
    imgs["y2"] = {}


    #go through the image directory and sort and store images
    for img_path in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_path)
        img_meta = pydicom.dcmread(img_path)
        jaws_x = img_meta[0x3002,0x0030][0][0x300A,0x00B6][0][0x300A,0X011C].value
        jaws_y = img_meta[0x3002,0x0030][0][0x300A,0x00B6][1][0x300A,0X011C].value
        img = img_meta.pixel_array
        img = normalize_by_top_median(img)
        img = gaussian_filter(img, sigma=3, order=0)    #smoothen the image
        img = zoom(img, zoom=2, order=3)
         
        #collimator positions not included in metadata, so determine closed jaw from lowest mean pixel intensity in each quarter blocked region
        y_range, x_range = img.shape
        #order of quarter regions is: [left, right, top, bottom]
        # mean_blocked_pixels = [np.mean(img[:, :int(x_range/2)]), np.mean(img[:, int(x_range/2):]), np.mean(img[:int(x_range/2), :]), np.mean(img[int(x_range/2):, :])]    #C0: x1, x2, y2, y1 / C90: y2, y1, x2, x1 
        # min_region_index = np.argmin(mean_blocked_pixels)

        # if min_region_index == 0:
        #     blocked_field = abs("x1")
        #     nominal_jaw = jaws_x[0]

        # if min_region_index == 1:
        #     blocked_field = "x2"
        #     nominal_jaw = abs(jaws_x[1])

        # if min_region_index == 2:
        #     blocked_field = "y2"
        #     nominal_jaw = abs(jaws_y[1])

        # if min_region_index == 3:
        #     blocked_field = "y1"
        #     nominal_jaw = abs(jaws_y[0])
        blocked_field = "x1"
        nominal_jaw = abs(jaws_x[0])
        imgs[blocked_field][round(nominal_jaw/10, 1)] = img

        blocked_field = "x2"
        nominal_jaw = abs(jaws_x[1])
        imgs[blocked_field][round(nominal_jaw/10, 1)] = img

        blocked_field = "y2"
        nominal_jaw = abs(jaws_y[1])
        imgs[blocked_field][round(nominal_jaw/10, 1)] = img

        blocked_field = "y1"
        nominal_jaw = abs(jaws_y[0])
        imgs[blocked_field][round(nominal_jaw/10, 1)] = img

    return imgs

def get_junc_offsets(img_dict, unit_num):
    #this function will determine the offset of each 1/4 blocked beam jaw with the isocentre (defined by bead in each phantom image at each gantry/coll setting)
    #values will be reported such that negative means the jaw passed over the iso, positive means it doesn't reach it. 
    
    #start by defining a dictionary of same format as img_dict that will hold the offsets.
    offset_dict = deepcopy(img_dict)

    
    for g in img_dict.keys():    #go through all gantry angles

        image = img_dict[g]["iso"]
        #start by getting the isocentre bead pixel location for each setting
        offset_dict[g]["iso"] = find_bead_location(image, round_final=False)
        
        #start with c = 0 images
        isocentre = offset_dict[g]["iso"]

        #y1 - want profile through centre y along x
        y1_profile = deepcopy(img_dict[g][0]["y1"][:, round(isocentre[1])-20:round(isocentre[1])+20])
        y1_profile[0:int(y1_profile.shape[0]/4),:] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        y1_profile[int(3*y1_profile.shape[0]/4):,:] = 1
        y1_offset = (find_half_intensity_pixel(y1_profile) - isocentre[0]) * 0.224/2   #make negative to follow sign convention (positive if jaw openy)     
        offset_dict[g][0]["y1"] = y1_offset 

        #repeat for y2 jaw
        y2_profile = deepcopy(img_dict[g][0]["y2"][:, round(isocentre[1])-20:round(isocentre[1])+20])
        y2_profile[0:int(y2_profile.shape[0]/4),:] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        y2_profile[int(3*y2_profile.shape[0]/4):,:] = 1
        y2_offset = -(find_half_intensity_pixel(y2_profile) - isocentre[0])* 0.224/2
        offset_dict[g][0]["y2"] = y2_offset 
        try:
            #repeat for x1 jaw
            x1_profile = deepcopy(img_dict[g][0]["x1"][round(isocentre[0])-20:round(isocentre[0])+20, :])
            x1_profile[:,0:int(x1_profile.shape[1]/4)] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
            x1_profile[:,int(3*x1_profile.shape[1]/4):] = 1
            x1_offset = -(find_half_intensity_pixel(x1_profile) - isocentre[1])* 0.224/2
            offset_dict[g][0]["x1"] = x1_offset  
        except:
            pass

        #repeat for x2 jaw
        try:
            x2_profile = deepcopy(img_dict[g][0]["x2"][round(isocentre[0])-20:round(isocentre[0])+20, :])
            x2_profile[:,0:int(x2_profile.shape[1]/4)] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
            x2_profile[:,int(3*x2_profile.shape[1]/4):] = 1
            x2_offset = (find_half_intensity_pixel(x2_profile) - isocentre[1])* 0.224/2
            offset_dict[g][0]["x2"] = x2_offset 
        except: pass

        # fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(17, 17))
        # ax[0,0].set_title(f"Gantry Angle: {g}$^\circ$, Collimator Angle: {0}$^\circ$")
        # ax[0,0].plot(y1_profile)
        # ax[0,1].plot(y1_profile_grad)
        # ax[0,2].imshow(img_dict[g][0]["y1"])
        # ax[0,0].text(1450, 0.8, f"Y1 Jaw offset = {round(y1_offset,3)} mm")    #pixel size 0.34 mm /2
    
        # ax[1,0].plot(y2_profile)
        # ax[1,1].plot(y2_profile_grad)
        # ax[1,2].imshow(img_dict[g][0]["y2"])
        # ax[1,0].text(0, 0.8, f"Y2 Jaw offset = {round(y2_offset,3)} mm")    #pixel size 0.34 mm /2

        # ax[2,0].plot(x1_profile)
        # ax[2,1].plot(x1_profile_grad)
        # ax[2,2].imshow(img_dict[g][0]["x1"])
        # ax[2,0].text(0, 0.8, f"X1 Jaw offset = {round(x1_offset,3)} mm")    #pixel size 0.34 mm /2

        # ax[3,0].plot(x2_profile)
        # ax[3,1].plot(x2_profile_grad)
        # ax[3,2].imshow(img_dict[g][0]["x2"])
        # ax[3,0].text(1450, 0.8, f"X2 Jaw offset = {round(x2_offset,3)} mm")    #pixel size 0.34 mm /2

        #fig.savefig(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"{g}_{0}_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"))
        #plt.show()
        #del fig

        #plt.close()

        #y1 - want profile through centre y along x
        x1_profile = deepcopy(img_dict[g][90]["x1"][:, round(isocentre[1])-20:round(isocentre[1])+20])
        x1_profile[0:int(x1_profile.shape[0]/4),:] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        x1_profile[int(3*x1_profile.shape[0]/4):,:] = 1
        x1_offset = (find_half_intensity_pixel(x1_profile) - isocentre[0])* 0.224/2   #make negative to follow sign convention (positive if jaw crosses iso, negative if shy)
        offset_dict[g][90]["x1"] = x1_offset

        #repeat for y2 jaw
        x2_profile = deepcopy(img_dict[g][90]["x2"][:, round(isocentre[1])-20:round(isocentre[1])+20])
        x2_profile[0:int(x2_profile.shape[0]/4),:] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        x2_profile[int(3*x2_profile.shape[0]/4):,:] = 1
        x2_offset = -(find_half_intensity_pixel(x2_profile) - isocentre[0])* 0.224/2
        offset_dict[g][90]["x2"] = x2_offset 

        #repeat for x1 jaw
        y2_profile = deepcopy(img_dict[g][90]["y2"][round(isocentre[0])-20:round(isocentre[0])+20, :])
        y2_profile[:, 0:int(y2_profile.shape[1]/4)] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        y2_profile[:, int(3*y2_profile.shape[1]/4):] = 1
        y2_offset = -(find_half_intensity_pixel(y2_profile) - isocentre[1])* 0.224/2
        offset_dict[g][90]["y2"] = y2_offset

        #repeat for x2 jaw
        y1_profile = deepcopy(img_dict[g][90]["y1"][round(isocentre[0])-20:round(isocentre[0])+20, :])
        y1_profile[:, 0:int(y1_profile.shape[1]/4)] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        y1_profile[:, int(3*y1_profile.shape[1]/4):] = 1
        y1_offset = (find_half_intensity_pixel(y1_profile) - isocentre[1])* 0.224/2
        offset_dict[g][90]["y1"] = y1_offset 


        # fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(17,17))
        
        # ax[0,0].set_title(f"Gantry Angle: {g}$^\circ$, Collimator Angle: {90}$^\circ$")
        # ax[0,0].plot(y1_profile)
        # ax[0,1].plot(y1_profile_grad)
        # ax[0,2].imshow(img_dict[g][90]["y1"])
        # ax[0,0].text(1450, 0.8, f"Y1 Jaw offset = {round(y1_offset,3)} mm")    #pixel size 0.34 mm /2
    
        # ax[1,0].plot(y2_profile)
        # ax[1,1].plot(y2_profile_grad)
        # ax[1,2].imshow(img_dict[g][90]["y2"])
        # ax[1,0].text(0, 0.8, f"Y2 Jaw offset = {round(y2_offset,3)} mm")    #pixel size 0.34 mm /2

        # ax[2,0].plot(x1_profile)
        # ax[2,1].plot(x1_profile_grad)
        # ax[2,2].imshow(img_dict[g][90]["x1"])
        # ax[2,0].text(1450, 0.8, f"X1 Jaw offset = {round(x1_offset,3)} mm")    #pixel size 0.34 mm /2

        # ax[3,0].plot(x2_profile)
        # ax[3,1].plot(x2_profile_grad)
        # ax[3,2].imshow(img_dict[g][90]["x2"])
        # ax[3,0].text(0, 0.8, f"X2 Jaw offset = {round(x2_offset,3)} mm")    #pixel size 0.34 mm /2
        
        # fig.savefig(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"{g}_{90}_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"))
        # del fig
        # plt.close()
        #plt.show()


        # #Now repeat for the c = 270 images from left counter clockwise - (y1, x2, y2, x1)
        #x2 - want profile through centre y along x
        # x2_profile = deepcopy(img_dict[g][270]["x2"][:, round(isocentre[1])-20:round(isocentre[1])+20])
        # x2_profile[0:int(x2_profile.shape[0]/4):] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        # x2_profile[int(3*x2_profile.shape[0]/4):,:] = 1
        # #determine centre as pixel with sharpest gradient
        # x2_offset = (np.mean(np.argmin(abs(x2_profile - 0.5), axis=0)) - isocentre[0]) * 0.224/2   #make negative to follow sign convention (positive if jaw crosses iso, negative if shy)     
        # offset_dict[g][270]["x2"] = x2_offset 
        # #repeat for x1 jaw
        # x1_profile = deepcopy(img_dict[g][270]["x1"][:, round(isocentre[1])-20:round(isocentre[1])+20])
        # x1_profile[0:int(x1_profile.shape[0]/4),:] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        # x1_profile[int(3*x1_profile.shape[0]/4):,:] = 1
        # x1_offset = -(np.mean(np.argmin(abs(x1_profile - 0.5), axis=0)) - isocentre[0])* 0.224/2
        # offset_dict[g][270]["x1"] = x1_offset 

        # #repeat for y1 jaw
        # y1_profile = deepcopy(img_dict[g][270]["y1"][round(isocentre[0])-20:round(isocentre[0])+20, :])
        # y1_profile[:,0:int(y1_profile.shape[1]/4)] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        # y1_profile[:,int(3*y1_profile.shape[1]/4):] = 1
        # y1_offset = -(np.mean(np.argmin(abs(y1_profile - 0.5), axis=1)) - isocentre[1])* 0.224/2
        # offset_dict[g][270]["y1"] = y1_offset  
 
        # #repeat for y2 jaw
        # y2_profile = deepcopy(img_dict[g][270]["y2"][round(isocentre[0])-20:round(isocentre[0])+20, :])
        # y2_profile[:,0:int(y2_profile.shape[1]/4)] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
        # y2_profile[:,int(3*y2_profile.shape[1]/4):] = 1
        # y2_offset = (np.mean(np.argmin(abs(y2_profile - 0.5), axis=1)) - isocentre[1])* 0.224/2
        # offset_dict[g][270]["y2"] = y2_offset 

        # offset_dict[g][270].pop('x2', None)
        # offset_dict[g][270].pop('x1', None)
        # offset_dict[g][270].pop('y1', None)
        # offset_dict[g][270].pop('y2', None)

        # fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(17, 17))
        # ax[0,0].set_title(f"Gantry Angle: {g}$^\circ$, Collimator Angle: {270}$^\circ$")
        # ax[0,0].plot(y1_profile)
        # ax[0,1].plot(y1_profile_grad)
        # ax[0,2].imshow(img_dict[g][270]["y1"])
        # ax[0,0].text(1450, 0.8, f"Y1 Jaw offset = {round(y1_offset,3)} mm")    #pixel size 0.34 mm /2
    
        # ax[1,0].plot(y2_profile)
        # ax[1,1].plot(y2_profile_grad)
        # ax[1,2].imshow(img_dict[g][270]["y2"])
        # ax[1,0].text(0, 0.8, f"Y2 Jaw offset = {round(y2_offset,3)} mm")    #pixel size 0.34 mm /2

        # ax[2,0].plot(x1_profile)
        # ax[2,1].plot(x1_profile_grad)
        # ax[2,2].imshow(img_dict[g][270]["x1"])
        # ax[2,0].text(0, 0.8, f"X1 Jaw offset = {round(x1_offset,3)} mm")    #pixel size 0.34 mm /2

        # ax[3,0].plot(x2_profile)
        # ax[3,1].plot(x2_profile_grad)
        # ax[3,2].imshow(img_dict[g][270]["x2"])
        # ax[3,0].text(1450, 0.8, f"X2 Jaw offset = {round(x2_offset,3)} mm")    #pixel size 0.34 mm /2

        # fig.savefig(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"{g}_{0}_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"))
        # #plt.show()
        # del fig
        # plt.close()

    #now want bar plots of offsets vs/ gantry / collimator for each angle
    #for clustered bar, need to sort data first into a new dictionary for plotting

    # for jaw in ["x1", "x2", "y1", "y2"]:
    #     gantry_angles = sorted(offset_dict.keys())
    #     plot_dic = {}
    #     for c in [90, 0, 270]:
    #         vals = []
    #         for g in sorted(offset_dict.keys()):
    #             vals.append(offset_dict[g][c][jaw])
    #         plot_dic[c] = tuple(vals)
        
        # g_range = np.arange(len(offset_dict.keys()))
        # width = 0.25
        # multiplier = 0

        # fig, ax = plt.subplots(layout='constrained')
        # colors= ["salmon", "moccasin", "skyblue"]    #colours for diff coll angles
        # for c, val in plot_dic.items():
        #     offset = width * multiplier
        #     rects = ax.bar(g_range + offset, val, width, label=f"Collimator {c}$^\circ$", color=colors[multiplier], edgecolor="black")
        #     ax.bar_label(rects, padding=3, rotation=90)
        #     multiplier += 1
        # ax.set_ylabel("Offset from isocentre (mm)")
        # ax.set_title(f"{jaw.upper()} Jaw Offset From Isocentre")
        # ax.set_xticks(g_range + width, gantry_angles)
        # ylims = ax.get_ylim()
        # ax.set_ylim([ylims[0]-abs(ylims[0])*0.15, ylims[1]+abs(ylims[1])*0.15])
        # #reset axes to fit legend nicely
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0+box.height * 0.1, box.width, box.height * 0.9])
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
        # fig.savefig(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"bar_plot_offsets_{jaw}_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"))
        # # plt.show(block=True)
        # # plt.close()
        # # plt.cla()
        # del fig
            



    return offset_dict
            
def get_jaw_offsets(img_dict,isocentre):
    #this function will determine the offset of each 1/4 blocked beam jaw with the isocentre (defined by bead in each phantom image at each gantry/coll setting)
    #values will be reported such that negative means the jaw passed over the iso, positive means it doesn't reach it. 
    
    #start by defining a dictionary of same format as img_dict that will hold the offsets.
    offset_dict = {"y1": {}, "y2": {}, "x1": {}, "x2": {}}

    for fs in [2.5, 5.0,7.5, 10.0]:
        #y1 - want profile through centre y along x
        try:
            y1_profile = deepcopy(img_dict["y1"][fs][:, round(isocentre[1])])
            y1_profile[0:int(y1_profile.size/2)] = 1  
            y1_offset = (np.argmin(abs(y1_profile - 0.5)) - isocentre[0]) * 0.224/2   #make negative to follow sign convention (positive if jaw openy)     
            offset_dict["y1"][fs] = round(y1_offset /10, 2)
        except:
            print(f"Could not find asymmetric jaw image for Y1 and fs: {fs}")

        #repeat for y2 jaw
        try:
            y2_profile = deepcopy(img_dict["y2"][fs][:, round(isocentre[1])])
            y2_profile[int(y2_profile.size/2):] = 1  
            y2_offset = -(np.argmin(abs(y2_profile - 0.5)) - isocentre[0])* 0.224/2
            offset_dict["y2"][fs] = round(y2_offset/10, 2)
        except:
            print(f"Could not find asymmetric jaw image for Y2 and fs: {fs}")

        #repeat for x1 jaw
        try:
            x1_profile = deepcopy(img_dict["x1"][fs][round(isocentre[0]), :])
            x1_profile[int(x1_profile.size/2):] = 1  
            x1_offset = -(np.argmin(abs(x1_profile - 0.5)) - isocentre[1])* 0.224/2
            offset_dict["x1"][fs] = round(x1_offset/10, 2)
        except:
            print(f"Could not find asymmetric jaw image for X1 and fs: {fs}")

        #repeat for x2 jaw
        try:
            x2_profile = deepcopy(img_dict["x2"][fs][round(isocentre[0]), :])
            x2_profile[0:int(x2_profile.size/2)] = 1    #make borders zero so that center closed jaw is properly found, and not the other jaw edge. 
            x2_offset = (np.argmin(abs(x2_profile - 0.5)) - isocentre[1])* 0.224/2
            offset_dict["x2"][fs] = round(x2_offset/10, 2)
        except:
            print(f"Could not find asymmetric jaw image for Y2 and fs: {fs}")

    return offset_dict

def find_bead_location(image: np.array, round_final=True, zoom_size=2):
    #here we simply determine the pixel location of the centre of the bead in the cubic phantom
    #img = (deepcopy(image) - np.amin(image)) / (np.amax(image) - np.amin(image))
    img = normalize_by_top_median(deepcopy(image))

    # img[img > 0.78] = 0.8   #for plotting
    # img[img < 0.68] = 0.65

    # plt.imshow(img, cmap="bone")
    # plt.show()


    #make outside borders large to make isocentre lowest value
    img[:,0:int(14*img.shape[1]/30)] = 1
    img[:,int(16*img.shape[1]/30):] = 1
    img[0:int(14*img.shape[0]/30),:] = 1
    img[int(16*img.shape[0]/30):,:] = 1


    #now keep the lowest 100 pixels (will be the bead location, and find centre of mass)
    pixel_list = sorted(img.flatten().tolist())
    pixel_100 = pixel_list[200*int(zoom_size/2)**2]
    img[img > pixel_100] = 0

    #now find the centre of mass of the remaining pixels 
    centre_pixels = np.nonzero(img)

    centre_of_mass = np.mean(centre_pixels, axis=1)

    img[round(centre_of_mass[0]), round(centre_of_mass[1])] = 1
    # plt.imshow(img)
    # plt.show()
    print(centre_of_mass)

    if round_final==False:
        return [round(centre_of_mass[0], 2), round(centre_of_mass[1],2)]
    else:
        return [round(centre_of_mass[0]), round(centre_of_mass[1])]

def calculate_cost(offsets : dict, old_offsets, use_lrfc, lrfc_vals,junction_priority=0.5, optimize_lrfc=True, optimize_junctions=True):
    #this function takes a dictionary of jaw offsets at all gantry/collimator angles and returns the cost function
    #junction_priority (default 0.8) - the fraction of the total gap "cost function" that is given to sum of junction gaps/overlaps. 
    #
    # The remaining fraction (1-junction_priority) will go towards minimizing the absolute value of all jaw offsets at each angle.

    cost_absolute = 0    #this is the portion of the cost due to the absolute offsets (not related to junctions)

    #start by computing the sum of all jaw offsets at all gantry/collimator angles (absolute positions)
    for g in offsets.keys():
        # if g == 270:
        #     continue
        for c in offsets[g].keys():
            if c == "iso":
                continue
            for jaw in ["x1","x2", "y1", "y2"]:
                if jaw in list(offsets[g][c].keys()):
                    cost_absolute += abs(offsets[g][c][jaw])
    #normalize by number of total jaw images (4 * 3 * 8)
    cost_absolute /= (4*2*6)

    ##
    ##

    #now get the cost related to junctions (specifically interested in g0c90/g180c90 x1 w/ c90 x2 at all non 0/180 gantry angles)
    if optimize_junctions == True:
        cost_junction = 0
        #first get the supraclav field junctions
        g0c90_x1 = offsets[0][90]["x1"]
        g180c90_x1 = offsets[180][90]["x1"]


        #now get the lower field junctions
        for g in [50, 130, 310, 230]:
            lower_x2 = offsets[g][90]["x2"]
            junction_gap_0 = abs(lower_x2 + g0c90_x1)  #adding together will give the total error from perfect junction (whether it's a gap or an overlap)
            junction_gap_180 = abs(lower_x2 + g180c90_x1)
            if junction_gap_0 > 0.9:
                cost_junction += 2*junction_gap_0
            else:
                cost_junction += junction_gap_0

            if junction_gap_180 > 0.9:
                cost_junction += 2*junction_gap_180
            else:
                cost_junction += junction_gap_180
        #normalize junction cost by total number of relevant junctions (8)
        cost_junction /= 8

        ##
        ##
        #lastly, need extra cost for junctions that make field colder (rather yhsn hotter --> negative)
        cost_cold_junction = 0
        for g in [50, 130, 310, 230]:
            lower_x2 = offsets[g][90]["x2"]
            junction_gap_0 = lower_x2 + g0c90_x1  #adding together will give the total error from perfect junction (whether it's a gap or an overlap)
            junction_gap_180 = lower_x2 + g180c90_x1
            if junction_gap_0 < 0:
                cost_cold_junction += abs(junction_gap_0)
            if junction_gap_180 < 0:
                cost_cold_junction += abs(junction_gap_180)

        cost_cold_junction /= 8

        #now total cost is a combination of the three values (based on the junction priority)
        cost = junction_priority*(cost_junction+cost_cold_junction) + (1-junction_priority)*cost_absolute
    else:
        cost = cost_absolute


    #Want to add cost term based on the lrfc values if use_lrfc == True
    if use_lrfc == True:
        #want to calculate the new displacement of rad/light field.
        #first calculate how much each jaw is shifted from original configuration
        disp_x1 = offsets[0][0]["x1"] - old_offsets[0][0]["x1"]
        disp_x2 = offsets[0][0]["x2"] - old_offsets[0][0]["x2"]
        disp_y1 = offsets[0][0]["y1"] - old_offsets[0][0]["y1"]
        disp_y2 = offsets[0][0]["y2"] - old_offsets[0][0]["y2"]
        #now calculate new lrfc val for all lrfc images checked

        lrfc_cost = 0
        #first calculate cost based on rad/light displacement
        if optimize_lrfc:
            for lrfc_val in lrfc_vals:
                rad_disp = lrfc_val[0]
                
                lrfc_jaw_disps = lrfc_val[1] #in order [y1_disp, y2_disp, x1_disp, x2_disp]
                new_rad_disp_y = rad_disp[0] + (disp_y1/2 + -disp_y2/2) 
                new_rad_disp_x = rad_disp[1] + (-disp_x1/2 + disp_x2/2)

                if new_rad_disp_y < 0.4:            
                        lrfc_cost += 0#abs(new_rad_light_y)
                elif new_rad_disp_y < 0.7:
                    lrfc_cost += abs(new_rad_disp_y*3)
                else:
                    lrfc_cost += abs(new_rad_disp_y*10) #huge cost, do not want an lrfc value out of action

                if new_rad_disp_x < 0.4: 
                        lrfc_cost += 0#abs(new_rad_light_x)
                elif new_rad_disp_x < 0.7:
                    lrfc_cost += abs(new_rad_disp_x*3)
                else:
                    lrfc_cost += abs(new_rad_disp_x*10) #huge cost, do not want an lrfc value out of action
            
                #now calculate cost from lrfc jaw displacements
                new_y1_disp = offsets[0][0]["y1"]#lrfc_jaw_disps[0] + disp_y1
                new_y2_disp = offsets[0][0]["y2"]#lrfc_jaw_disps[1] + disp_y2
                new_x1_disp = offsets[0][0]["x1"]#lrfc_jaw_disps[2] + disp_x1
                new_x2_disp = offsets[0][0]["x2"]#lrfc_jaw_disps[3] + disp_x2

                new_lrfc_jaw_disps = [new_y1_disp, new_y2_disp, new_x1_disp, new_x2_disp]

                for lrfc_jaw_disp in new_lrfc_jaw_disps:
                    if abs(lrfc_jaw_disp) < 0.5:
                        lrfc_cost += lrfc_jaw_disp/4    #divide by 4 because there are 4 jaws
                    elif abs(lrfc_jaw_disp) < 0.75:
                        lrfc_cost += 3*lrfc_jaw_disp/4
                    else:
                        lrfc_cost += 10*lrfc_jaw_disp/4 #don't want to consider cases with large jaw displacement errors.

            lrfc_cost /= 2*len(lrfc_vals)

            cost += lrfc_cost


    return cost


def get_opt_origin(offsets : dict, jaw_offsets, junction_priority, unit_num, lrfc_folder=None, optimize_junctions=True):
    #this function takes the offset dictionary (for each gantry angle, each collimator angle, each jaw) and computes the optimal calibration point.
    # our primary objective is to minimize the sum of gaps between g0c90, g180c90 - x2 and off axis gantry angles w/ collimator 90 and x1
    #
    #Variables:
    #offsets (dict)
    #junction_priority (default 0.8) - the fraction of the total gap "cost function" that is given to sum of junction gaps/overlaps. 
    #
    # The remaining fraction (1-junction_priority) will go towards minimizing the absolute value of all jaw offsets at each angle.

    #This function works by iterating through a set of possible calibration points (from -1 mm to 1mm across isocentre  in x/y direction) and
    #calculating the new jaw offsets each time and subsequent cost function
    #the cost function will be stored for each iteration, and finally, the cal point giving the minimum cost will be returned

    x1_iters = np.linspace(-0.49,0.49,31)
    x2_iters = np.linspace(-0.49,0.49,31)
    y1_iters = np.linspace(-0.49,0.49,21)
    y2_iters = np.linspace(-0.49,0.49,21)
    cost_vals = np.zeros((31,31,21,21))    #save as 2d grid with first dimension = x, second dimension = y

    #so for each iteration, first calculate the new offsets after shifting each jaw by respective amount
    #assume x and y vectors are in same direction as image vectors (so y1 > y2 in image - aka if calibration iso shifts by -1, then y1 would increase and y2 would decrease)
    if lrfc_folder is not None:
        use_lrfc = True
        lrfc_vals = []
        lrfc_field_sizes = []
        for file in os.listdir(lrfc_folder):
            lrfc_file = os.path.join(lrfc_folder, file)
            lrfc_points= lrfc(lrfc_file)
            lrfc_vals.append([lrfc_points["rad_disp"], lrfc_points["jaw_disps"]])
            lrfc_field_sizes.append(lrfc_points["field_size"])
    else:
        use_lrfc = False

    #in service mode, jaws must be calibrated at g = 0 and c = 0, so need to calculate the new offsets in terms of shift from original offsets at g0c0 to iso
    for x1_ind, x1 in enumerate(x1_iters):
        for y1_ind, y1 in enumerate(y1_iters):
            for y2_ind, y2 in enumerate(y2_iters):
                for x2_ind, x2 in enumerate(x2_iters):
                    new_offsets = deepcopy(offsets)
                    for g in offsets.keys():
                        for c in offsets[g].keys():
                            if c == "iso":
                                continue
                            try:
                                new_offsets[g][c]["x1"] = offsets[g][c]["x1"] - offsets[0][0]["x1"] + x1   #difference in offset from the calibration position + the cal position offset from isocentre
                            except: pass
                            try:
                                new_offsets[g][c]["x2"] = offsets[g][c]["x2"]- offsets[0][0]["x2"] + x2
                            except: pass
                            try:
                                new_offsets[g][c]["y1"] = offsets[g][c]["y1"]- offsets[0][0]["y1"] + y1
                            except: pass
                            try:
                                new_offsets[g][c]["y2"] = offsets[g][c]["y2"]- offsets[0][0]["y2"] + y2
                            except: pass

                    #now compute the cost
                    cost = calculate_cost(new_offsets, offsets, use_lrfc, lrfc_vals, junction_priority=junction_priority, optimize_junctions=optimize_junctions)
                    cost_vals[x1_ind, x2_ind, y1_ind, y2_ind] = cost


     #best cost = minimum value
    opt_offset_ind = np.argwhere(cost_vals == np.amin(cost_vals))

    opt_offset_x1 = x1_iters[opt_offset_ind[0,0]]
    opt_offset_x2 = x2_iters[opt_offset_ind[0,1]]
    opt_offset_y1 = y1_iters[opt_offset_ind[0,2]]
    opt_offset_y2 = y2_iters[opt_offset_ind[0,3]]



    #make plots of cost vs. collimator points, varying two at a time (with others at optimal values)

    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
    ax[0].imshow(cost_vals[:,:,opt_offset_ind[0,2], opt_offset_ind[0,3]], cmap='rainbow')
    x_ticks = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]
    y_ticks = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]
    x_labels = []
    y_labels = []
    ax[0].set_xticks(x_ticks)
    ax[0].set_yticks(y_ticks)

    ax[1].set_xticks(x_ticks)
    ax[1].set_yticks(y_ticks)
  
    for i in range(len(x_ticks)):
        x_labels.append(round(x1_iters[int(x_ticks[i])],1))
    for i in range(len(y_ticks)):   
        y_labels.append(round(x2_iters[int(y_ticks[i])],1))
    ax[0].set_xticklabels(x_labels)
    ax[0].set_yticklabels(y_labels)
    ax[0].set_xlabel("X1 Displacement from Iso (mm)", fontsize=16)
    ax[0].set_ylabel("X2 Displacement from Iso (mm)", fontsize=16)

    #also show the cost values with colormap levelled
    vmax = np.amax(cost_vals[:,:,opt_offset_ind[0,2], opt_offset_ind[0,3]]) - 0.95 * (np.amax(cost_vals[:,:,opt_offset_ind[0,2], opt_offset_ind[0,3]]) - np.amin(cost_vals[:,:,opt_offset_ind[0,2], opt_offset_ind[0,3]]))
    ax[1].imshow(np.log(cost_vals[:,:,opt_offset_ind[0,2], opt_offset_ind[0,3]]), cmap='rainbow')#.imshow(cost_vals[:,:,opt_offset_ind[0,2], opt_offset_ind[0,3]], cmap='rainbow', vmax=vmax)
    ax[1].set_xticklabels(x_labels)
    ax[1].set_yticklabels(y_labels)
    ax[1].set_xlabel("X1 Displacement from Iso (mm)", fontsize=16)
    ax[1].set_ylabel("X2 Displacement from Iso (mm)", fontsize=16)

    fig.savefig(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"optimal_x1_x2_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"))
    #plt.show()
    del fig

 

   #now plot y1,y2 cost
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
    ax[0].imshow(cost_vals[opt_offset_ind[0,0], opt_offset_ind[0,1],:,:], cmap='rainbow')
    x_ticks = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
    y_ticks = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
    x_labels = []
    y_labels = []
    ax[0].set_xticks(x_ticks)
    ax[0].set_yticks(y_ticks)

    ax[1].set_xticks(x_ticks)
    ax[1].set_yticks(y_ticks)

    for i in range(len(x_ticks)):
        x_labels.append(round(y1_iters[int(x_ticks[i])],1))
    for i in range(len(y_ticks)):   
        y_labels.append(round(y2_iters[int(y_ticks[i])],1))
    ax[0].set_xticklabels(x_labels)
    ax[0].set_yticklabels(y_labels)
    ax[0].set_xlabel("Y1 Displacement from Iso (mm)", fontsize=16)
    ax[0].set_ylabel("Y2 Displacement from Iso (mm)", fontsize=16)

    #also show the cost values with colormap levelled
    vmax = np.amax(cost_vals[opt_offset_ind[0,0], opt_offset_ind[0,1],:,:]) - 0.95 * (np.amax(cost_vals[opt_offset_ind[0,0], opt_offset_ind[0,1],:,:]) - np.amin(cost_vals[opt_offset_ind[0,0], opt_offset_ind[0,1],:,:]))
    ax[1].imshow(np.log(cost_vals[opt_offset_ind[0,0], opt_offset_ind[0,1],:,:]), cmap='rainbow')#.imshow(cost_vals[opt_offset_ind[0,0], opt_offset_ind[0,1],:,:], cmap='rainbow', vmax=vmax)
    ax[1].set_xticklabels(x_labels)
    ax[1].set_yticklabels(y_labels)
    ax[1].set_xlabel("Y1 Displacement from Iso (mm)", fontsize=16)
    ax[1].set_ylabel("Y2 Displacement from Iso (mm)", fontsize=16)
    fig.savefig(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"optimal_y1_y2_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}"))
    #plt.show()
    del fig



    #recalculate optimal offsets
    new_offsets = deepcopy(offsets)
    for g in offsets.keys():
        for c in [0, 90]:#offsets[g].keys(): #removed 270
            if c == "iso": 
                continue
            new_offsets[g][c]["x1"] = offsets[g][c]["x1"] - offsets[0][0]["x1"] + opt_offset_x1   #difference in offset from the calibration position + the cal position offset from isocentre
            new_offsets[g][c]["x2"] = offsets[g][c]["x2"]- offsets[0][0]["x2"] + opt_offset_x2

            new_offsets[g][c]["y1"] = offsets[g][c]["y1"]- offsets[0][0]["y1"] + opt_offset_y1
            new_offsets[g][c]["y2"] = offsets[g][c]["y2"]- offsets[0][0]["y2"] + opt_offset_y2

    disp_x1 = new_offsets[0][0]["x1"] - offsets[0][0]["x1"]
    disp_x2 = new_offsets[0][0]["x2"] - offsets[0][0]["x2"]
    disp_y1 = new_offsets[0][0]["y1"] - offsets[0][0]["y1"]
    disp_y2 = new_offsets[0][0]["y2"] - offsets[0][0]["y2"]
    #now calculate new lrfc val for all lrfc images checked
    if use_lrfc:
        new_lrfcs = []
        for lrfc_val in lrfc_vals:
            new_rad_light_y = lrfc_val[0][0] + (disp_y1/2 + -disp_y2/2) 
            new_rad_light_x = lrfc_val[0][1] + (-disp_x1/2 + disp_x2/2)

            new_y1_disp = new_offsets[0][0]["y1"]#lrfc_val[1][0] + disp_y1
            new_y2_disp = new_offsets[0][0]["y2"]#lrfc_val[1][1] + disp_y2
            new_x1_disp = new_offsets[0][0]["x1"]#lrfc_val[1][2] + disp_x1
            new_x2_disp = new_offsets[0][0]["x2"]#lrfc_val[1][3] + disp_x2
            
            new_lrfcs.append([[new_rad_light_y, new_rad_light_x],[new_y1_disp, new_y2_disp, new_x1_disp, new_x2_disp]])
    #want to write this data to a csv:

    with open(os.path.join(os.getcwd(), f"U{unit_num}_Output", f"jaws_and_junctions_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}.csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([f"Offsets"])
        writer.writerow(["", f"Current", "Optimal"])
        writer.writerow(["x1", str(offsets[0][0]["x1"]), str(opt_offset_x1)])
        writer.writerow(["x2", str(offsets[0][0]["x2"]), str(opt_offset_x2)])
        writer.writerow(["y1", str(offsets[0][0]["y1"]), str(opt_offset_y1)])
        writer.writerow(["y2", str(offsets[0][0]["y2"]), str(opt_offset_y2)])
        writer.writerow(["Junction", "Original Offset", "Final Offset"])

        #calculate the junctions
        g0c90_x1 = offsets[0][90]["x1"]
        g180c90_x1 = offsets[180][90]["x1"]

        g0c90_x1_final = new_offsets[0][90]["x1"]
        g180c90_x1_final = new_offsets[180][90]["x1"]
        for g in [50, 130, 310, 230]:    #just the breast tangent angles for calculating junctions
            lower_x2 = offsets[g][90]["x2"]
            junction_gap_0 = lower_x2 + g0c90_x1  #adding together will give the total error from perfect junction (whether it's a gap or an overlap)
            junction_gap_180 = lower_x2 + g180c90_x1

            lower_x2_final = new_offsets[g][90]["x2"]
            junction_gap_0_final = lower_x2_final + g0c90_x1_final  #adding together will give the total error from perfect junction (whether it's a gap or an overlap)
            junction_gap_180_final = lower_x2_final + g180c90_x1_final

            #now add these junctions to the csv
            writer.writerow([f"g0c90_x1 and g{g}c90_x2", junction_gap_0, junction_gap_0_final])
            writer.writerow([f"g180c90_x1 and g{g}c90_x2", junction_gap_180, junction_gap_180_final])
        #now that junctions have been included, add all the offset values
        if use_lrfc:
            writer.writerow(["","",""])
            for f,field_size in enumerate(lrfc_field_sizes):
                writer.writerow([f"Radiation Light Field Coincidence ({field_size}X{field_size})", "Original", "Final"])
                writer.writerow(["Y", lrfc_vals[f][0][0],new_lrfcs[f][0][0]])
                writer.writerow(["X", lrfc_vals[f][0][1],new_lrfcs[f][0][1]])

                writer.writerow([f"Radiation Jaw Displacements ({field_size}X{field_size})", "Original", "Final"])
                writer.writerow(["Y1", lrfc_vals[f][1][0],new_lrfcs[f][1][0]])
                writer.writerow(["Y2", lrfc_vals[f][1][1],new_lrfcs[f][1][1]])
                writer.writerow(["X1", lrfc_vals[f][1][2],new_lrfcs[f][1][2]])
                writer.writerow(["X2", lrfc_vals[f][1][3],new_lrfcs[f][1][3]])

        writer.writerow(["","",""])
        writer.writerow(["Jaw Displacements from Isocentre"])
        writer.writerow(["","",""])



        writer.writerow(["Gantry Angle","Collimator Angle","X1", "", "X2", "", "Y1", "", "Y2", ""])
        writer.writerow(["","","Original", "Final","Original", "Final","Original", "Final","Original", "Final"])
        for g in offsets.keys():
            for c in [0,90]:#offsets[g].keys(): taking out 270
                if c == "iso":
                    continue
                writer.writerow([g,c,offsets[g][c]["x1"],new_offsets[g][c]["x1"], offsets[g][c]["x2"],new_offsets[g][c]["x2"], offsets[g][c]["y1"],new_offsets[g][c]["y1"], offsets[g][c]["y2"],new_offsets[g][c]["y2"]])
        if jaw_offsets is not None:
            writer.writerow(["","",""])
            writer.writerow(["Asymmetric Jaw Measurements"])
            writer.writerow(["","",""])
            writer.writerow(["X1"])
            for pos in jaw_offsets["x1"].keys():
                writer.writerow([pos, jaw_offsets["x1"][pos]])
            writer.writerow(["","",""])

            writer.writerow(["X2"])
            for pos in jaw_offsets["x2"].keys():
                writer.writerow([pos, jaw_offsets["x2"][pos]])
            writer.writerow(["","",""])

            writer.writerow(["Y1"])
            for pos in jaw_offsets["y1"].keys():
                writer.writerow([pos, jaw_offsets["y1"][pos]])
            writer.writerow(["","",""])

            writer.writerow(["Y2"])
            for pos in jaw_offsets["y2"].keys():
                writer.writerow([pos, jaw_offsets["y2"][pos]])
            

    return tuple((opt_offset_x1, opt_offset_x2, opt_offset_y1, opt_offset_y2)), new_offsets

def predict_optimal_encoders(date, unit_num, junction_priority, img_folder, jaw_pos_folder, enc_img_folder, enc_iso_img_path, lrfc_folder, optimize_junctions=True, epid_position=1.086):

    if not os.path.exists(os.path.join(os.getcwd(), f"U{unit_num}_Output")):
        os.mkdir(os.path.join(os.getcwd(), f"U{unit_num}_Output"))

    #first collect imgs for closed jaws:
    junc_img_dict = sort_junc_img_dict(img_folder)
    #also get images for asymmetric jaw positions
    if jaw_pos_folder is not None:
        jaw_img_dict = sort_jaw_img_dict(jaw_pos_folder)


    #fit_encoder_vs_pixel_funcs(enc_img_folder, enc_iso_img_path, unit_num=unit_num, optimal_cal=[0.1, 0.1, -0.5, -0.3])
    # #now want to define the offset of each 1/4 blocked beam's jaw from isocentre at each gantry/collimator combination
    junc_offsets = get_junc_offsets(junc_img_dict, unit_num)
    if jaw_pos_folder is not None:
        isocentre = junc_offsets[0]["iso"]
        jaw_offsets = get_jaw_offsets(jaw_img_dict, isocentre)
    else:
        jaw_offsets = None

    # #now find the optimal calibration point (relative to g = 0, c = 0 isocentre image) to be used for calibration
    optimal_cal, new_offsets = get_opt_origin(junc_offsets, jaw_offsets, junction_priority, unit_num, lrfc_folder=lrfc_folder, optimize_junctions=optimize_junctions)    #x1,x2,y1,y2
    print(f"Optimal Calibration Shift: {optimal_cal}")
    # optimal_cal = [0.5,1,-0.5,-1]

    #now get jaw images to use for encoder-jaw correlations

    fit_encoder_vs_pixel_funcs(date, enc_img_folder, enc_iso_img_path, unit_num=unit_num, optimal_cal=optimal_cal, epid_position=epid_position)

# import random
# a = np.ones((1000,100))
# for i in range(100):
#     a[i*10,i] = random.random()*0.5
# vals = find_half_intensity_pixel(a)
def main():
    print("////////////////////////////////////////////")
    print("JawCal - TrueBeam Jaw Calibration Optimizer.")
    print("////////////////////////////////////////////")
    print("To begin optimizing your jaw calibration positions, please follow the instructions in the README file to properly name and sort your image data.")
    print("Carefully respond to the following prompts:")
    unit_num=input("Please enter the unit number, matching the one used when naming data folders. (eg. 'U1_jaws_post_feb22' --> enter '1')\n")
    junction_priority=input("Please enter the junction priority to use in the cost function (between 0 and 1, larger = more emphasis on junction optimization, smaller = more emphasis on individual jaw positions.)\n")
    date=input("Please enter the date, matching the one used when naming data folders. (eg. 'U1_jaws_post_feb22' --> enter 'feb22')\n")
    pre_or_post = input("Please enter 'pre' or 'post', matching the one used when naming data folders. (eg. 'U1_jaws_post_feb22' --> enter 'post')\n")
    epid_position = input("Please enter, as an absolute number, the position of the epid used when acquiring encoder correlation images. (eg. I've found ~8.6 aligns the isocentre cube at the isocentre.)\n")#1.086
    epid_position += 100
    epid_position /= 100
    optimize_junctions = True

    img_folder = os.path.join(os.getcwd(), "Images", f"U{unit_num}_{pre_or_post}_{date}")
    lrfc_folder = os.path.join(os.getcwd(), "Images", f"U{unit_num}_lrfc_{pre_or_post}_{date}")

    enc_img_folder = os.path.join(os.getcwd(), "Images", f"U{unit_num}_encoders_{date}")
    enc_iso_img_path = os.path.join(os.getcwd(), "Images", f"U{unit_num}_iso_encoder_{date}.dcm")

    jaw_pos_folder = os.path.join(os.getcwd(), "Images", f"U{unit_num}_jaws_{pre_or_post}_{date}")

    predict_optimal_encoders(date, unit_num, junction_priority, img_folder, jaw_pos_folder, enc_img_folder, enc_iso_img_path, lrfc_folder, optimize_junctions=optimize_junctions, epid_position=epid_position)

    print("Program Finished Successfully")

if __name__ == "__main__":
    main()