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
import numpy as np
from face_detection.RealDetect.PCA import *


def class_mean(dataSet):
    means = mean(dataSet, axis=0)
    mean_vectors = mat(means)
    return mean_vectors


def within_class_S(dataSet):
    m = shape(dataSet[1])[1]
    class_S = mat(zeros((m, m)))
    mean = class_mean(dataSet)
    for line in dataSet:
        x = line - mean
        class_S += x.T * x
    return class_S

def lda(male, female, dimension):
    m_mean = class_mean(male)
    f_mean = class_mean(female)
    m_s = within_class_S(male)
    f_s = within_class_S(female)
    SW = m_s + f_s
    # 2-class, rank is 1
    # SB = (m_mean - f_mean) * (m_mean - f_mean).T
    # real_male = zeros((227,dimension))
    # real_female = zeros((227, dimension))
    # for i in range(227):
    #     real_male[i, :] = male[i, :] - m_mean
    #     real_female[i, :] = female[i, :] - f_mean
    #
    # SW = zeros((dimension, dimension))
    # for i in range(227):
    #     SW = SW + real_male[i, :].T * real_male[i, :]
    #     SW = SW + real_female[i, :].T * real_female[i, :]
    #
    # print(SW)

    # Since this is 2-class problem, vector can be easily got as follow:

    w = np.linalg.inv(SW)*(m_mean - f_mean).T
    w = w / linalg.norm(w)

    Male = w.T * male.T
    Female = w.T * female.T
    # avg_Male = zeros((30, 1))
    # avg_Female = zeros((30,1))
    # for  i in range(227):
    #     avg_Male = avg_Male + Male[:,i]
    #     avg_Female = avg_Female + Female[:,i]
    avg_Male = mean(Male,0)
    avg_Female = mean(Female,0)
    # np.save("/Users/ChenxiLi/Desktop/male.npy", avg_Male)
    # np.save("/Users/ChenxiLi/Desktop/female.npy", avg_Female)
    np.save("/Users/ChenxiLi/Desktop/lda_vector.npy",w)
    return w, avg_Male, avg_Female

def fitLDA(test,w, avg_male, avg_female):
    transform = w.T * test.T
    print(mean(transform - avg_male))
    print(mean(transform - avg_female))
    if abs(mean(transform - avg_male)) < abs(mean(transform - avg_female)):
        print("It's a male!")
    else:
        print("It's a female!")



