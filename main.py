#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:43:25 2019

@author: gloria & beilei
"""

import constant
import download_data

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from knn_class import *

np.random.seed(constant.RANDOM_SEED)
#%%
# =============================================================================
# OLD PREPROCESSING... WILL BE DEPRECATED
# =============================================================================
if __name__ == "__main__":
    
#    # check if directories exist
#    print('Cheking directories...')
#    download_data.mkdir_if_dne(constant.UNCROPPED_DIR_M)
#    download_data.mkdir_if_dne(constant.CROPPED_DIR_M)
#    download_data.mkdir_if_dne(constant.UNCROPPED_DIR_F)
#    download_data.mkdir_if_dne(constant.CROPPED_DIR_F)
#    
#    
#    # process dataframe to make sure which files to download
#    df_M = download_data.process_data_df(constant.act,constant.MALE_DF, constant.UNCROPPED_DIR_M)
#    df_F = download_data.process_data_df(constant.act,constant.FEMALE_DF, constant.UNCROPPED_DIR_F)
#    
#    #download files...
#    files_timedout_M = download_data.download(constant.act, df_M)
#    files_timedout_F = download_data.download(constant.act, df_F)
#    
#    # process downloaded imsges and save them 
#    file_paths_M = glob.glob(constant.UNCROPPED_DIR_M + '*.jpg') # i got 467 male images
#    empty_file_M,error_file_M = download_data.get_processed(df_M, file_paths_M, constant.CROPPED_DIR_M) 
#    '''
#    Daniel Radcliffe  171  
#    Gerard Butler     146  
#    Michael Vartan    150  
#    '''
#    #plz confirm 43 of male images is empty
#    # i.e. len(empty_file) ==43
#    file_paths_F = glob.glob(constant.UNCROPPED_DIR_F + '*.jpg') # i got 467 images
#    #NOTE BROKEN FILE
#    BROKEN_FILE = 'harmon33.jpg' # THIS ONE DOESNT EXIST ON FLICKR ANYMORE
#    file_paths_F = [f for f in file_paths_F if BROKEN_FILE not in f]
#    empty_file_F, error_file_F = download_data.get_processed(df_F, file_paths_F, constant.CROPPED_DIR_F) 
#    #plz confirm 43 of them is empty
#    # i.e. len(empty_file) ==43
#   =============================================================================
#   OLD PREPROCESSING... WILL BE DEPRECATED
#   this is how to process them
#   =============================================================================
    print('make resized dir if dir doesnt exist')
    download_data.mkdir_if_dne(constant.RESIZED_GRAY_IMG_DIR)
    download_data.mkdir_if_dne(constant.FIGURE_DIR)
    
    uncropped = []
    for actor in constant.ACTOR_LIST:
        files = download_data.get_actor_img(constant.UNCROPPED_IMG_DIR, actor)
        uncropped.extend(files)
    
    resized = glob.glob(constant.RESIZED_GRAY_IMG_DIR + "*")
    if not resized:  
        # trun cropped images into grayscale and resize
        print('%s is empty! start processing images' % constant.RESIZED_GRAY_IMG_DIR)
        for actor in constant.ACTOR_LIST:
            cropped_files = download_data.get_actor_img(constant.CROPPED_IMG_DIR, actor)
            resized_images = [download_data.process_cropped_image(file) for file in cropped_files]
            resized.extend(resized_images)
    else:
        print('loaded resized img paths')
    
    uncropped_df = pd.DataFrame({'uncropped_path':uncropped})
    uncropped_df['img_id'] = uncropped_df.uncropped_path.str.extract(r'(\d+)')
    uncropped_df['actor'] = uncropped_df['uncropped_path'].str.extract(r'uncropped_images/(.*)/')

    
    
    df = pd.DataFrame({'resized_path':resized})
    df['actor'] = df['resized_path'].str.extract(r'resized_images/(.*)_[0-9]')
    df['img_id'] = df["resized_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    df = df[['actor', 'img_id', 'resized_path']]
    df.sort_values('actor',axis=0,inplace = True)
    df = df.reset_index(drop = True)
    underline_actor = ["_".join(x.split()) for x in constant.ACTOR_LIST]
    df['Gender'] = np.where(df.actor.isin(underline_actor[:3]), 'M', 'F')
    
    df3 = pd.merge(df, uncropped_df, how='inner', on=['img_id'])
    df3.rename(columns={'actor_x':'actor'}, inplace=True)
 
    
#    uncropped_images = [download_data.read_in_and_flattened_images(path) for path in df3.uncropped_path.values]
#    cropped_images = [download_data.read_in_and_flattened_images(path) for path in df3.resized_path.values]



    #%% RANDOMLY CHOOSE 3 SAMPLES
    
    '''
    Part 1 (10%)
    Describe the dataset of faces. 
    In particular, provide at least three examples of the images in the dataset,
     as well as at least three examples of cropped out faces. 
    Comment on the quality of the annotation of the dataset: are the bounding boxes accurate? 
    Can the cropped-out faces be aligned with each other?
    '''
    import shutil
        
    download_data.mkdir_if_dne(constant.PART1_SAMPLE_DIR)
    
    # randomly choose 3 sample images
    part1_samples = df3.sample(3,random_state = constant.RANDOM_SEED)
    
    # find cropped file names
    cropped_samples = constant.CROPPED_IMG_DIR + part1_samples.actor + '/' + part1_samples.actor + \
    '_'+ part1_samples.img_id + '.png'
    cropped_samples = cropped_samples.tolist()
    
    # find uncropped file names
    uncropped_samples = constant.UNCROPPED_IMG_DIR + part1_samples.actor.str.replace("_", " ")  + '/'+\
    part1_samples.actor.str.replace("_", " ") + '_'+ part1_samples.img_id + '.jpg'
    uncropped_samples = uncropped_samples.tolist()
    
    # copy file to folder
    def copy_to_dir(list_of_files):
        for i in range(len(list_of_files)):
            file_name = os.path.basename(list_of_files[i])
            to_dir = constant.PART1_SAMPLE_DIR + file_name
            shutil.copyfile(list_of_files[i], to_dir)
    
    copy_to_dir(cropped_samples)
    copy_to_dir(uncropped_samples)
    
    #%%
    '''
    Part 2 (5%)
    Separate the dataset into three non-overlapping parts: 
        the training set (100 face images per actor), 
        the validation set (10 face images per actor), and
        the test set (10 face images per actor).
    For the report, describe how you did that. (Any method is fine).
    The training set will contain faces whose labels you assume you know. 
    The test set and the validation set will contain faces whose labels you pretend to not know and will attempt to determine using the data in the training set. 
    You will use the performance on the validation set to tune the K, and you will then report the final performance on the test set.
    '''
    sample_df = df3.groupby('actor').apply(lambda s: s.sample(120,random_state = constant.RANDOM_SEED))
    sample_df = sample_df.reset_index(drop=True)
    
    training_df= sample_df.groupby('actor').apply(lambda t: t[:100])
    training_df = training_df.reset_index(drop=True)
    training_resized_paths = training_df.resized_path.tolist()
    training_resized_imgs_flatten = [download_data.read_in_and_flattened_images(path) for path in training_resized_paths]
    training_resized_imgs_flatten = np.asarray(training_resized_imgs_flatten)
    training_resized_imgs = [download_data.read_in_img(path) for path in training_resized_paths]
    training_labels = training_df.actor.values
    training_gender_lables=training_df.Gender.values
    
    validation_df = sample_df.groupby('actor').apply(lambda t: t[101:111])
    validation_df = validation_df.reset_index(drop=True)
    validation_resized_paths = validation_df.resized_path.tolist()
    validation_resized_imgs_flatten = [download_data.read_in_and_flattened_images(path) for path in validation_resized_paths]
    validation_resized_imgs_flatten = np.asarray(validation_resized_imgs_flatten)
    validation_resized_imgs = [download_data.read_in_img(path) for path in validation_resized_paths]
    validation_labels = validation_df.actor.values
    validation_gender_lables=validation_df.Gender.values
    
    testing_df = sample_df.groupby('actor').apply(lambda t: t[-10:])
    testing_df = testing_df.reset_index(drop=True)
    testing_resized_paths = testing_df.resized_path.tolist()
    testing_resized_imgs_flatten = [download_data.read_in_and_flattened_images(path) for path in testing_resized_paths]
    testing_resized_imgs_flatten = np.asarray(testing_resized_imgs_flatten)
    testing_resized_imgs = [download_data.read_in_img(path) for path in testing_resized_paths]
    testing_labels = testing_df.actor.values
    testing_gender_lables=testing_df.Gender.values
    
    
    
    
    
   #%%
    '''
    Part 3 (45%)
    Write code to recognize faces from the set of faces act using K-Nearest Neighbours (note that face recognition is a classification task!). Determine the K using the validation set, and the report the performance on the test set. Here and elsewhere in the project, use the L2 distance between the flattened images of cropped-out faces in order to find the nearest neighbours.
    
    Note: the reason you need to use both a validation set and a test set is that picking the best k by using the validation set and then reporting the performance on the same set makes it so that you report artificially high performance. A better measure is obtained by picking the parameters of the algorithm using a validation set, and then measuring the performance on a separate test set.
    
    The performance you obtain will likely be around 50% or a little higher
    
    In addition to the performance on the test set, in your report, display 5 failure cases (where the majority of the k nearest neighbours of a test face are not the same person as the test person). Display both the face to be recognized and its 5 nearest neighbours.
    '''
    from knn_class import *
    
    """
    #OUTPUT:
    the distance bewteen all traning img and new predict image 
    """
    total_val_err = []
    for k in range(1,11):
        errors = np.zeros(len(validation_resized_imgs_flatten))
        for i in range(len(validation_resized_imgs_flatten)):
            neighbors = cal_neighbours(training_resized_imgs_flatten,training_labels,
                                   validation_resized_imgs_flatten[i],k)
            neighbors_labels_test = [i[2] for i in neighbors]
            predicted_val_label = winner_vote(neighbors_labels_test)
            if predicted_val_label[0][0] != validation_labels[i]:
                errors[i] = 1
        total_val_err.append(errors.mean())
    best_k = np.where(total_val_err==min(total_val_err))[0][0] + 1

#%%
    total_test_err = []
    for k in range(1,11):
        test_errors = np.zeros(len(testing_resized_imgs_flatten))
        for i in range(len(testing_resized_imgs_flatten)):
            neighbors_test = cal_neighbours(training_resized_imgs_flatten,training_labels,
                                   testing_resized_imgs_flatten[i],k)
            neighbors_test_label = [i[2] for i in neighbors_test]
            predicted_test_label = winner_vote(neighbors_test_label)
            if predicted_test_label[0][0] != testing_labels[i]:
                test_errors[i] = 1
        total_test_err.append(test_errors.mean())
        
    test_errors
    testing_df['test_errors'] = test_errors 
   #%% 
    total_training_err= []
    for k in range(1,11):
        training_errors = np.zeros(len(training_resized_imgs_flatten))
        for i in range(len(training_resized_imgs_flatten)):
            neighbors_training = cal_neighbours(training_resized_imgs_flatten,training_labels,
                                   training_resized_imgs_flatten[i],k)
            neighbors_training_label = [i[2] for i in neighbors_training]
            predicted_training_label = winner_vote(neighbors_training_label)
            if predicted_training_label[0][0] != training_labels[i]:
                training_errors[i] = 1
        total_training_err.append(training_errors.mean())
    #neighbors_labels = [n[2] for n in neighbors]
    #validation_labels = validation_df.actor.tolist()
    #winner_vote(neighbors_labels)
    
    training_errors

    
    #%% geting 5 neighbors for testing set 
 
    #total_failed=testing_df
    #total_failed=total_failed[total_failed.test_errors.isin([1])
    testing_df = testing_df.reset_index(drop=True)  # reset index to numbers 

    total_fialed_5nearest_neighbors = []  
    total_files_to_copy = []
    
    total_failed_neighbors_true_label= []

    for i, value in testing_df['test_errors'].iteritems():
        if value == 1: # if test erro is 1 then we put int neibours function to get 5 nearest labours!
                failed_neighbors_true_label = testing_df.iloc[i,0]
                failed_neighbors_testing, failed_neighbors_loc = cal_neighbours(training_resized_imgs_flatten,
                                                          training_labels,testing_resized_imgs_flatten[i],5, True)
                files_to_copy = [training_df.iloc[j,4] for j in failed_neighbors_loc]
                failed_neighbors_testing_label = [i[2] for i in failed_neighbors_testing]
                total_fialed_5nearest_neighbors.append(failed_neighbors_testing)
                total_files_to_copy.append(files_to_copy)
                total_failed_neighbors_true_label.append(failed_neighbors_true_label)
                
#%%
    # 5 faied case with 5 nearest neighbours
    download_data.mkdir_if_dne(constant.PART3_RESULT_DIR)
    
    def copy_to_dir_part3(list_of_files, save_dir):
        for i in range(len(list_of_files)):
            file_name = os.path.basename(list_of_files[i])
            to_dir = save_dir + file_name
            shutil.copyfile(list_of_files[i], to_dir)
    
    for i in range(5):
        print("-----------------------------------------")
        print("The failed case is:")
        print(total_failed_neighbors_true_label[i])
        print("fThe nearest 5 neigbours are:")
        print(total_fialed_5nearest_neighbors[i])
        save_dir = constant.PART3_RESULT_DIR + '/group%s/'%i
        download_data.mkdir_if_dne(save_dir)
        copy_to_dir_part3(total_files_to_copy[i],save_dir)
        print("-----------------------------------------")

    #%%
    '''
    Part 4 (10%)
    Plot the performance on the test, train, and validation sets of 
    the K-Nearest-Neighbours algorithm for various K. Make sure the axes of 
    your graphs are correctly labelled. Explain why the plots look the way they do.
    '''
    x=[1,2,3,4,5,6,7,8,9,10]

    
    plt.plot(x,total_training_err,'g',label='traning errors')
    plt.plot(x,total_test_err,'b',label='test errors')
    plt.plot(x,total_val_err,'r',label='validation errors')
    plt.axis([0, 11, 0, 1])
    plt.title('performance')
    plt.xlabel('The vaule of k')
    plt.ylabel('Mean of the error')
    plt.legend()
    plt.savefig(constant.FIGURE_DIR + 'performence of face recognition.jpg')
    
    
    
    
    #%%
    '''
    Part 5 (20%)
    Write code to determine, for every face in a set of faces, the gender of the person using K-Nearest-Neighbours. 
    This should work in the same way as face recognition, except instead of each face being assigned a name, 
    each face is considered to be simply either male or female. Again, use your validation set to select a k for the best performance 
    (i.e., the proportion of faces whose gender was classified correctly), report the performance for the different k's for the validation set,
    and then report the results on the test set for the k that works best for the validation set. 
    For this part, you should still use the set of actors act for both the test and the validation set.
    '''
    
    
    total_gender_val_err = []
    for k in range(1,11):
        gender_errors = np.zeros(len(validation_resized_imgs_flatten))
        for i in range(len(validation_resized_imgs_flatten)):
            gender_neighbors = cal_neighbours(training_resized_imgs_flatten,training_gender_lables,
                                   validation_resized_imgs_flatten[i],k)
            gender_neighbors_labels_test = [i[2] for i in gender_neighbors]
            gender_predicted_val_label = winner_vote(gender_neighbors_labels_test)
            if gender_predicted_val_label[0][0] != validation_gender_lables[i]:
                gender_errors[i] = 1
        total_gender_val_err.append(gender_errors.mean())
        
    best_k_gender = np.where(total_gender_val_err==min(total_gender_val_err))[0][0] + 1

#%%
    total_gender_test_err = []
    for k in range(1,11):
        gender_test_errors = np.zeros(len(testing_resized_imgs_flatten))
        for i in range(len(testing_resized_imgs_flatten)):
            gender_neighbors_test = cal_neighbours(training_resized_imgs_flatten,training_gender_lables,
                                   testing_resized_imgs_flatten[i],8)
            gender_neighbors_test_label = [i[2] for i in gender_neighbors_test]
            gender_predicted_test_label = winner_vote(gender_neighbors_test_label)
            if gender_predicted_test_label[0][0] != testing_gender_lables[i]:
                gender_test_errors[i] = 1
        mean_error_test=gender_test_errors.mean()
        
        gender_test_errors
        testing_df['gender_test_errors'] = gender_test_errors 
#%%
'''
    total_gender_train_err = []
    for k in range(1,11):
        gender_training_errors = np.zeros(len(training_resized_imgs_flatten))
        for i in range(len(training_resized_imgs_flatten)):
            gender_neighbors_training = cal_neighbours(training_resized_imgs_flatten,training_gender_lables,
                                   training_resized_imgs_flatten[i],k)
            gender_neighbors_training_label = [i[2] for i in gender_neighbors_training]
            gender_predicted_training_label = winner_vote(gender_neighbors_training_label)
            if gender_predicted_training_label[0][0] != training_gender_lables[i]:
                gender_training_errors[i] = 1
        total_gender_train_err.append(gender_training_errors.mean())
        
   # gender_training_errors
   '''
#%%
    #performence of gender prediction
    x=[1,2,3,4,5,6,7,8,9,10]

    #plt.plot(x,total_gender_train_err,'g',label='traning errors')
    plt.plot(8,mean_error_test,'bo',label='test errors')
    plt.plot(x,total_gender_val_err,'r',label='validation errors')
    plt.axis([0, 11, 0, 1])
    plt.title('The peformance of the gender prediction ')
    plt.xlabel('The number of k')
    plt.ylabel('Mean of the error')
    plt.legend()
    #plt.save(constant.FIGURE_DIR + 'performence of gender prediction.jpg')
    

    #%%
    '''
    Part 6 (10%)
    Now, evaluate the implementation in Part 5 on faces of actors who are not in the training set 
    (i.e., actors other than the people in the set act.)
    Discuss the difference between the performance 
    in Part 5 and the performance in Part 6. 
    '''
    actor_list = pd.read_table(constant.SUBSET_ACTORS,header=None)[0]
    actor_to_download = list(set([x for x in actor_list if x not in constant.ACTOR_LIST]))
    actress_list = pd.read_table(constant.SUBSET_ACTRESSES,header=None)[0].tolist()
    actress_to_download = list(set([x for x in actress_list if x not in constant.ACTOR_LIST]))
    
    ## ORGANIZE DATA DOWNLOADED
    download_data.mkdir_if_dne(constant.NEW_RESIZED_GRAY_IMG_DIR)
        
    new_uncropped = []
    for actor in (actor_to_download + actress_to_download):
        files = download_data.get_actor_img(constant.UNCROPPED_IMG_DIR, actor)
        new_uncropped.extend(files)
    
    new_resized = glob.glob(constant.NEW_RESIZED_GRAY_IMG_DIR + "*")
    
    if not new_resized:  
        # trun cropped images into grayscale and resize
        print('%s is empty! start processing images' % constant.RESIZED_GRAY_IMG_DIR)
        for actor in (actor_to_download + actress_to_download):
            new_cropped_files = download_data.get_actor_img(constant.CROPPED_IMG_DIR, actor)
            new_resized_images = [download_data.process_cropped_image(file, True, constant.NEW_RESIZED_GRAY_IMG_DIR)\
                                  for file in new_cropped_files]
            new_resized.extend(new_resized_images)
    else:
        print('loaded new resized img paths')
    
    uncropped_df_new = pd.DataFrame({'uncropped_path':new_uncropped})
    uncropped_df_new['img_id'] = uncropped_df_new.uncropped_path.str.extract(r'(\d+)')
    uncropped_df_new['actor'] = uncropped_df_new['uncropped_path'].str.extract(r'uncropped_images/(.*)/')

    df_new = pd.DataFrame({'resized_path':new_resized})
    df_new['actor'] = df_new['resized_path'].str.extract(r'resized_images/(.*)_[0-9]')
    df_new['img_id'] = df_new["resized_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    df_new = df_new[['actor', 'img_id', 'resized_path']]
    df_new.sort_values('actor',axis=0,inplace = True)
    df_new = df_new.reset_index(drop = True)
    underline_actor_new = ["_".join(x.split()) for x in actor_to_download]
    df_new['Gender'] = np.where(df_new.actor.isin(underline_actor_new), 'M', 'F')
    
    df_new2 = pd.merge(df_new, uncropped_df_new, how='inner', on=['img_id'])
    df_new2.rename(columns={'actor_x':'actor'}, inplace=True)
    df_new2 = df_new2.drop(columns = ['actor_y'])
    
#    df_new2 = df_new2.iloc[0:1000,:]
    new_resized_paths = df_new2.resized_path.tolist()
    new_testing_resized_imgs_flatten = [download_data.read_in_and_flattened_images(path) for path in new_resized_paths]
    new_testing_resized_imgs_flatten = np.asarray(new_testing_resized_imgs_flatten)
    new_testing_resized_imgs = [download_data.read_in_img(path) for path in new_resized_paths]
#    new_testing_labels = df_new2.actor.values
    new_testing_gender_labels = df_new2.Gender.values

    ## TESTING ERROR
    
    print('WARNING: THIS PART TAKES FOREVER!')
    print('run it on a happy day :3')
    new_total_test_error = []
    new_test_errors = np.zeros(len(new_testing_resized_imgs_flatten))
    k = 8
    for i in range(len(new_testing_resized_imgs_flatten)):
        neighbors_test = cal_neighbours(new_testing_resized_imgs_flatten,new_testing_gender_labels,
                               new_testing_resized_imgs_flatten[i],k)
        neighbors_test_label = [i[2] for i in neighbors_test]
        predicted_test_label = winner_vote(neighbors_test_label, False)
        if predicted_test_label[0][0] != new_testing_gender_labels[i]:
            new_test_errors[i] = 1
        if (i%100==0):
            print(i)    
    np.savetxt('testing_error_k8.csv', new_test_errors)
#    test_errors
#    df_new2['test_errors'] = test_errors 

