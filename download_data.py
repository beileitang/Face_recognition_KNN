#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:42:07 2019
COMPLETE RESIZE SHIT TONIGHT
@author: gloria
"""

from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import cv2
import os
import urllib
import constant
import glob



#act = list(set([a.split("\t")[0] for a in open(FACES_TXT).readlines()]))

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
    
def mkdir_if_dne(dir_path):
    '''
    #INPUT:
    # dir_path: path to the directory you want to examine
    '''
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('Made dir: %s \n' % dir_path)
    else:
        print('dir %s already exists ^o^' % dir_path)

## =============================================================================
## WILL BE DEPRECATED process data
## =============================================================================
#def resize_image(img,coord):
#    '''
#    #INPUT:
#    # file_path: path to image 
#    # coord: [x1,x2,y1,y2]
#    
#    where (x1,y1) is the coordinate of the top-left corner of the bounding box 
#    and (x2,y2) is that of the bottom-right corner, 
#    with (0,0) as the top-left corner of the image.
#    
#    #OUTPUT: 
#    # resized_img: cropped,resized image in grayscale
#    '''
#    x1 = coord[0]
#    y1 = coord[1]
#    x2 = coord[2]
#    y2 = coord[3]
#    # crop img
#    cropped_img = img[y1:y2, x1:x2]
#    # convert to gray scale
#    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
#    # resize to 32*32
#    resized_img = cv2.resize(gray, (32, 32))
#    return resized_img
#
#
#def process_data_df(actor_list, df, uncropped_dir):
#    '''
#    #INPUT:
#    # actor_list: list of names of actors/acctresses 
#    # df: dataframe containing URLs of images with faces, 
#        as well as the bounding boxes of the faces.
#    # uncropped_dir: all the downloaded .jpg files will be saved here
#    #OUTPUT:
#    # df: with new row containing where each file is saved
#    __note__: all the files are saved to 
#    '''
#    # sub-select the actors who are in the list
#    df = df[df[0].isin(actor_list)]
#    df = df.reindex()
#    df[4] = df[4].str.split(',').apply(lambda s: list(map(int, s)))
#    last_name = df[0].str.split().map(lambda x: x[1])
#    last_name = last_name.str.lower()
#    occurence = df.groupby([0]).cumcount().astype('str')
#    file_name = last_name + occurence + '.jpg'
#    df[6] = uncropped_dir + file_name
#    return df
#
#   
#def download(actor_list, df):
#    '''
#    #INPUT:
#    # actor_list: list of names of actors/acctresses 
#    # df: dataframe containing URLs of images with faces, 
#        as well as the bounding boxes of the faces.
#    #OUTPUT:
#    # files_timedout: files that are not downloaded
#    __note__: all the files are saved to 
#    '''
#    print('start downloading.... \n this might take a few minutes >O<')
#    start_time = time.time()
#    testfile = urllib.request
#    files_timedout = []
#    for i in range(df.shape[0]):
#        URL = df.iloc[i,3]
#        save_as = df.iloc[i,6]
#        timeout(testfile.urlretrieve, (URL, save_as), {}, 30)
#        if not os.path.exists(save_as):
#            print('file %s timedout' % save_as)
#            files_timedout.append(save_as)
#        else:
#            print('file %s downloaded' % save_as)
#    print('time spent: %s seconds' % (time.time()-start_time))
#    return files_timedout
#
#def get_processed(df, file_paths, cropped_dir):
#    '''
#    #INPUT:
#    #  df: info df 
#    #  file_paths: the actual files downloaded
#    #OUTPUT:
#    # nothing~~ just saving processed_imgaes to cropped_dir
#    '''
#    #process and save image
#    df = df[df[6].isin(file_paths)]
#    empty_file = []
#    error_file = []
#    for i in range(df.shape[0]):
#        coord = df.iloc[i,4]
#        file_path = df.iloc[i,6]
#        #read img
#        img = cv2.imread(file_path)
#        if not img is None:
#            filename = os.path.basename(file_path)
#            processed_img = resize_image(img,coord)
#            if cropped_dir:
#                try:
#                    processed_filepath = cropped_dir + filename
#                    cv2.imwrite(processed_filepath,processed_img)
#                    print("saved processed file @: %s" % processed_filepath)
#                except:
#                    print('(╬ﾟдﾟ)▄︻┻┳═一')
#                    print(file_path)
#                    error_file.append(file_path)
#        else:
#            print('(／‵Д′)／~ ╧╧')
#            print('%s is empty! ' % file_path)
#            empty_file.append(file_path)
#    return empty_file, error_file
#        
# =============================================================================
#  preprocess and read in file
# =============================================================================
def get_img_types(dir_path, types = ['*.jpg', '*.png','*.jpeg']):
    globbed = []
    types_upper = [x.upper() for x in types]
    types = types + types_upper
    for file_type in types:
        type_path = os.path.join(dir_path,file_type)
        globbed.extend(glob.glob(type_path))
    return globbed

def get_actor_img(dir_path, actor_name):
    if not os.path.exists(dir_path):
        print("%s doesnt exist"% dir_path)
    if 'uncropped' not in dir_path.lower():
        actor_dir = os.path.join( dir_path, '_'.join(actor_name.split()))
    else:
        actor_dir = os.path.join(dir_path, actor_name)
    files = get_img_types(actor_dir)
    return files

def process_cropped_image(path, save= True, save_dir = constant.RESIZED_GRAY_IMG_DIR):
    file_name = os.path.basename(path)
    save_path = save_dir + file_name
    #load image
    img = cv2.imread(path)
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize to 32*32
    resized_img = cv2.resize(gray, (32, 32))
    if save:
        sucess_save = cv2.imwrite(save_path,resized_img)
        if sucess_save:
            print('image saved: %s' % save_path)
        else:
            print('error saving...')
    return resized_img, save_path

def read_in_and_flattened_images(path):
    img = cv2.imread(path)
    if not img.any():
        return path
    else:
        return img.flatten()

def read_in_img(path):
    return cv2.imread(path)

#%%

# =============================================================================
# # USE THE TIMEOUT FUNCTION TO DOWNLOAD
#        
# =============================================================================
#UNCROPPED_DIR = constant.UNCROPPED_DIR_M
#FACES_TXT = "lib/subset_actors.txt"
#act = ['Gerard Butler', 'Daniel Radcliffe',
#       'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
#df = pd.read_table(FACES_TXT,header=None)
#df = df[df[0].isin(act)]
#
#
#testfile = urllib.request 
## CHECK IF UNCROPPED_DIR EXISTS
#download_data.mkdir_if_dne(constant.UNCROPPED_DIR_M)
#download_data.mkdir_if_dne(constant.CROPPED_DIR_M)
#
#
#for a in act:
#    name = a.split()[1].lower()
#    i = 0
#    for index,row in df.iterrows():
#        if a in row[0]:
#            filename = name + str(i)+'.jpg'
#            coord = row[4].split(',')
#            coord = list(map(int, coord))
#            save_as = constant.UNCROPPED_DIR_M + filename
#            #A version without timeout (uncomment in case you need to 
#            #unsupress exceptions, which timeout() does)
#            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
#            #timeout is used to stop downloading images which take too long to download
#            timeout(testfile.urlretrieve, (row[3], save_as), {}, 30)
#            if not os.path.isfile(save_as + filename):
#                continue
#            processed_img = process_image(save_as,coord)
#            cv2.imwrite(constant.CROPPED_DIR_M + filename ,processed_img)
#            print(filename)
#            i += 1
        
