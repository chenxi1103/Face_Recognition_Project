# Face_Recognition_Project
Final Project for CMU 18794: Pattern Recognition Theory

# 1 Introduction
In this research, the approaches using Fisherfaces and Convolutional
Neural Network (CNN) are proposed for recognizing gender, race and
emotion based on detected facial images. The effectivity and performance
of different algorithms are compared based on the real experiments. Finally,
a demonstration and feedback learning system was designed and realized by
web-based application development. The ultimate results verified that the
proposed CNN approaches are effective solutions for multi-facial attributes
recognition.

# 2 Methodology
Complete algorithm for multi-facial attributes can be divided into three parts:
1. Face detection and extraction
2. Training classification model by specific algorithms (PCA, LDA, CNN)
and Testing accuracy by utilizing trained models
3. Result demonstration realized by development of web application

### 2.1 Face detection and extraction
Before actually classifying the facial attributes, the precise facial landmark detection
is essential for valid and effective result. Face detection is one of the
most important tasks of any facial classification method system. Specifically, the
main face detection methods can be classified into 4 categories: feature-based,
appearance-based, knowledge-based, and template matching [1]. Among these
four, appearance-based method has the best performance which can also be used
for feature extraction for face recognition. The appearance-based model further
divided into sub-methods like ‘Eigenface-based’, ‘Distribution-based’, ‘Neural
Network’ and so on [2]. Practically, there are mainly two classifiers provided
by OpenCV - “Haar Cascade Classifier” and “LBP Cascade Classifier”. Haar
classifiers provides higher accuracy while LBP is computationally simpler and
fast [3].Haar Cascade Classifier is more suitable in this case since the dataset is
relatively small.

