#!/usr/bin/env python
# coding:utf-8
# ------Author:Chenxi Li--------
import os
import operator
from numpy import *
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.misc
from src.LDA import *

def pca(data,k):
    data = float32(mat(data))
    # Get the row and col value of the data
    row,col = data.shape
    # Calculate the mean value of each column
    col_mean = mean(data,0)
    # expend the total mean (by using numpy function tile)
    real_mean = tile(col_mean,(row,1))
    # PCA first step: Data - mean
    real_data = data - real_mean

    # 1. Calculate the Grim matrix
    Cov_Matrix = real_data * real_data.T
    # 2. Calculate the eigenvector and eigenvalue of the covariance matrix
    Value, Vector = linalg.eig(Cov_Matrix)
    # 3. Select the most significant k eigenvectors
    Real_Vector = Vector[:, 0:k]
    Real_Vector = real_data.T * Real_Vector
    # Normalize selected eigenvector
    for i in range(k):
        L = linalg.norm(Real_Vector[:, i])
        Real_Vector[:, i] = Real_Vector[:, i] / L
    new_data = real_data * Real_Vector
    #reconMat = (new_data * Real_Vector.T) + real_mean
    return new_data, Real_Vector, real_mean

def img2vector(filename):
    # Convert image to grayscale
    img = cv2.imread(filename,0)
    # Get the row and col value of the image
    row,col = img.shape
    # Convert the image to vector
    imgVector = reshape(img,(1,row*col))
    return imgVector

def loadData(k, gender):
    print("--Starting Load the Data Set---")

    dataDir = "/Users/ChenxiLi/Desktop/Face_Recognition_Project/data/gender/" + gender
    files = os.listdir(dataDir)
    trained_data = zeros((k, 350*350))
    count = 0
    for file in files:
        if not os.path.isdir(file):
            if file.endswith(".jpg"):
                img_vector = img2vector(dataDir+"/"+file)
                trained_data[count, :] = img_vector
                count = count + 1
    print("--Data Pre-processing Done!--")
    return trained_data, count

def start_lda():
    trained_data = loadData(227, "male")[0]
    trained_data2 = loadData(227, "female")[0]
    all_data = np.vstack((trained_data, trained_data2))
    eig_v= pca(all_data, 225)[1]
    return eig_v
