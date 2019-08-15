```
---
title: Face Recognition and Gender Classification with K-Nearest Neighbours (KNN)
layout: post
---

**Codes for this project are joint work with @yuhsienliu ** 

# Introduction

We constructed facial recognition programs with KNN algorithm to classify the names and gendors of actors/actresses in a subset of the [FaceScrub](https://megaface.cs.washington.edu/) dataset. This project is done in Python on MacOS.

## K-nearest-neighbours explained

The k-nearest-neighbours (KNN) classification algoriths, is one of the simplest supervised Machine Learning algorithms. $K$ is interpreted as the number of nearest neighbors desired to examine.

Let's say we have a dataset consiting of data pairs. That is, our data is a collection of $(x_1, y1), (x_2, y_2), …, (x_N, y_N)$. We can collect all the datapoints and call them our feature  $Xtrain= [datapoint1, …, datapointN]$ and our target labels $Ytrain = [label1, …, labelN]$. Given an unseen datapoint $Xtest_i$, we are interested in knowing the label for this unseen point $Ytest_i$. If we have want to estimate the true label for a point $y_i$ with KNN, it calculates the Euclidean distances between that point and all the points in $Xtrain$ and returns the lables for $k$ points that are closest to $y_i$. The majority of the labels for closest points would be the estimated label for this new observation $y_i$.

Here in Figure 1 (please bear with my hand drawing (〃∀〃) ), we assume there are 2 kinds of lables, $\{ +,-\}  $. And the dots denote the datapoints. We set $K$, the number of nearest neighbours, to be 3. And we endup with 3 points being in blue circle to be taken into consideration. Since the majority, two, of them have the label $-$, the estimated label would be $-$ as well.

# Data 

## Dataset

In our study of name classification, we used a subset of 6 actors, including Angie Harmon, Peri Gilpin, Lorraine Bracco, Micheal Vartan, Daniel Radcliffe, and Gerad Butler. The first 3 actresses are female and the latter 3 are male. From the file paths of the cropped and uncropped images, we collected and saved information, such as actor name and gender, in a Python pandas dataframe.

For the gender classification, we trained KNN classifier on the above 6 actors/acctresses. And then we test the performance of this classifier on other 24 actors/acctresses (12 are male, 12 are female.)

##  Download

The full FaceScrub data set is 16.4 GB and contains uncropped, cropped images  and their bounding boxes sorted in folders of the names of actors/acctresses. 

First of all, we unzip each of the actor's folder containing the cropped images set on terminal 


## Matching 

Then,  we checked if each cropped image has their corresponding uncropped image.  We examine if output directories (one for the figures shown in this paper and another for the resized images) exist or not, then we load the file paths for each cropped and uncropped image in a pandas dataframe.


In order to choose an optimzal number of nearest neighbor ($k_{best}$), images are divided into three sets: training, validation and testing set. The training set is used to calculate relevant distances; validation set is used to make sure we choose the $k$ with smallest mean $0-1$ error; testing set is used to examine the performance.

## Matching results and examples

We have most of the uncropped pictures aligned with their cropped correspondence. However, a few files seem to be broken. After matching up cropped and uncropped images, we end up with the number of images shown in **TABLE**. We then re-sized the coloured cropped images into $32\times 32-$pixel images and transformed them into black and white pictures.

<p style="text-align: center;"><img src="figures/PART1_sample.jpg"  height="300"><br>
  <b>Figure 2</b>: Examples of cropped and uncropped images. <br>left: uncropped. middle: cropped. right: re-sized</p>

## 

## Sampling

For each actor, $120$ images are randomly chosen with the Pandas data frame sample function. The first $100$ entries in the $120$ samples form the training set; $101-111$ entries are taken as validation set; The last $111-120$ images are used as the testing set. 

## Image processing

We loaded  and resized the matched images again with OpenCV-Python. The loaded images are converted into grayscale and r esized  to size of $32\times32\times3$(representing BGR form images) Python NumPy matrices. These images can further be flattened into $1-$dimensional array with the numpy.flatten() method. 

# KNN from Scratch

## 1. Euclidean distance


## 2. cal_neighbours

We sorted the distances between X_train and the given testing image, X_test, and return the locations where the closest neighbors are.

## 3. majority vote

# Results

## Face recognition

We used $10$ different values for $k$ from $1$ to $10$ on validation data set, and the best performance is the value of $k$ equals to $4$. Then, by using the KNN algorithm with the best value of $k$ ($k=4$) on test set , we get the mean of error which is $0.267$

## Gender recognition

We tested different values for $k$ from $1$ to $10$ on validation set to get the best value of k. And we found the best performing value for $k$ is $8$ on the validation set with mean of testing error to be $0.083$

