#!/usr/bin/env python
# coding:utf-8
# ------Author:Chenxi Li--------
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
from LDA import *

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

def loadData(gender):
    print("--Starting Load the Data Set---")
    files = glob.glob("../data/gender/%s/*" % gender)
    random.shuffle(files)
    k = int(len(files))
    trained_data = zeros((k, 128 * 128))
    count = 0
    for file in files:
        # if not os.path.isdir(file):
            # if file.endswith(".jpg"):
                img_vector = img2vector(file)
                trained_data[count, :] = img_vector
                count = count + 1
    print("--Data Pre-processing Done!--")
    return trained_data, count

def writeData(data,count,mean,gender,):
    print("--Start to Write Out Data--")
    dataDir = "/Users/ChenxiLi/Desktop/Face_Recognition_Project/data/gender/pca_" + gender
    for i in range(count):
        reconstruct_data = double(data[i,:])
        reconstruct_data = reshape(reconstruct_data,(128,128))
        print(reconstruct_data)
        filename = dataDir+"/"+ str(i) + ".jpg"
        scipy.misc.imsave(filename, reconstruct_data)
    print("--You are all set--")

if __name__ == '__main__':
    [trained_data, countM] = loadData("male")
    [trained_data2, countF] = loadData("female")
    print(countM)
    print(countF)
    all_data = np.vstack((trained_data, trained_data2))
    [all, eig_v, real_mean]= pca(all_data, 2457)
    row, col = all.shape
    male = all[0:countM,:]
    female = all[countM:row,:]
    [w, testMale, testFemale] = lda(male, female,2457)
    np.save("/Users/ChenxiLi/Desktop/male.npy", testMale)
    np.save("/Users/ChenxiLi/Desktop/female.npy", testFemale)
    np.save("/Users/ChenxiLi/Desktop/lda_vector.npy", w)
    print("---Training Done!---")
