# Facial Multi-Attribute Detection based on CNN and PCA/LDA
Chenxi Li Electrical & Computer Eng.Dept, Carnegie Mellon University Pittsburgh, PA 15213 chenxili@andrew.cmu.edu
Jiayue Bao Electrical & Computer Eng.Dept, Carnegie Mellon University Pittsburgh, PA 15213 jiayueb@andrew.cmu.edu
Lishang Chang Electrical & Computer Eng.Dept, Carnegie Mellon University Pittsburgh, PA 15213 lishangc@andrew.cmu.edu
# Abstract
In this research, the approaches using Fisherfaces and Convolutional
Neural Network (CNN) are proposed for recognizing gender, race and
emotion based on detected facial images. The effectivity and performance
of different algorithms are compared based on the real experiments. Finally,
a demonstration and feedback learning system was designed and realized by
web-based application development. The ultimate results verified that the
proposed CNN approaches are effective solutions for multi-facial attributes
recognition.

# 1 Introduction
Facial attributes detection has become an increasingly popular research topic and has signiﬁcant applicability in various areas such as identity veriﬁcation, monitoring in public areas, photo classiﬁcation and so on. With rapid development of related algorithms and open-source tools, it has become possible for realization of accurate classiﬁcation of facial attributes like gender, race and emotion. In this project, the main goal is to study facial attributes recognition among various person with different genders, races, and emotions through establishing classiﬁcation models based on major pattern recognition methods.

# 2 Methodology
Complete algorithm for multi-facial attributes can be divided into three parts:
1. Face detection and extraction
2. Training classification model by specific algorithms (PCA, LDA, CNN)
and Testing accuracy by utilizing trained models
3. Result demonstration realized by development of web application

## 2.1 Face detection and extraction
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
<img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture1.png">
<p align="center">Figure 1: Face Detection by Haar Cascade Classiﬁer</p><br>
For gender and race recognition, in order to guarantee the eﬃciency and eﬀect of the model training, the detected face would be extracted, and resized to be 64 px * 64 px grayscale image which is prepared for the further training. This kind of data preprocessing and cleaning would fasten the training process and neglect the unnecessary information in the image eﬀectively.
<img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture2.png">
<p align="center">Figure 2: Female Face Images Example: Data Preprocessing and Cleaning</p><br>

For emotion detection, we did not use image format (.jpg/.png) data to train the model. Instead, we will use fer2013.csv (will be described in detail in the following section) as our training data. We still need to detect and extract the face from a image when using our training data to recognize emotions. Here, the detected faces will be resize to 48 px * 48 px in order to ﬁt the format in training model.

## 2.2 Training classiﬁcation model by speciﬁc algorithms (PCA, LDA, CNN)and Testing accuracy by utilizing trained models
### 2.2.1 PCA + LDA (Fisherfaces) - For gender classiﬁcation
Among three target attributes the project intends to research on, gender is the simplest one since it is a binary-classiﬁcation problem which requires less eﬀort to identify the features. According to the existing researches related to gender classiﬁcation, there are mainly two ways to accomplish this task: “EigenFaces”, which is more focused on PCA, and “FisherFaces”, which is more focused on LDA [4]. For classifying gender based on human faces, The latter one is more appropriate since “EigenFaces” is optimal for representation rather than discrimination. In this case, the best solution might be the combination of PCA and LDA, which guarantees both of the dimensionality reduction and better classiﬁcation. The project ﬁnally decided to construct “FisherFaces” by combining PCA and LDA to do the gender classiﬁcation.

First, PCA algorithm was developed for dimensionality reduction and major features extracting. Another important objective for PCA is to prepare data for LDA to accomplish “FisherFace” construction. Since when handling highdimensional data, the parameter “within-class-scatter matrix” S w would possibly to be singular. For a singular matrix, it is not invertible, which prevents the further calculation of projection vector w. As S w still has at most N-C (N is the total number of samples from all classes, C is the number of classes) non-zero eigenvalues, PCA should keep at most N-C eigenvectors to guarantee that Sw is full rank, and then do LDA to compute the C − 1 projections in this lower-dimensional N-C subspace. After projection, two classes “male” and “female” would reach the greatest degree of separation. When giving another input image for testing, the image would be ﬁrst projected on the same projection vector and compare the distances between these two classes. If it is closer to “male” class, it would be classiﬁed as “male” and vice versa.

### 2.2.2 Convolutional Neural Network (CNN) - For gender, race and emotion classiﬁcation
CNN is a kind of neural network made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single diﬀerentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer citeref5.

CNN is used in model training for gender, race and emotion detection implementation. The gender classiﬁcation model was re-trained by CNN for comparison of eﬀectivity between Fisherfaces and CNN. CNN was chosen for race and emotion classiﬁcation instead of PCA/LDA as the latter is not able to perform well in multiple classiﬁcations like emotions.

CNN is typically made up of three layers: convolutional layer, pooling layer and fully-connected layer. We followed a [INPUT - CONV - RELU - POOL - FC] architecture in the training model. Taking CNN model for emotion classiﬁcation as an example:

1. INPUT[48*48*1] layer is the raw grayscale image data (1 channel)

2. CONV layer will use 3*3 kernel to computer dot product. We used 3 convolution layers of 128, 64, 32 ﬁlters respectively

3. RELU layer will apply an elementwise activation function max(0,x) thresholding at zero. The size of the volume will remain unchanged

4. POOL layer will perform a downsampling operation and we chose a [1,2,2,1] downsampling to reduce the spatial dimension to half

5. FC layer, which is the fully connected layer will computer the class score, resulting in the size of [1*1*7], where 7 is the class number of emotions. Softmax is used here

The detailed setting for hyper parameters will be listed in the next section.

## 2.3 Result demonstration realized by development of web application
For better demonstration of the results and further model optimization by expanding training dataset, a web application deployed on the cloud server was developed which allows anyone to visit, test trained classiﬁcation models, and provide training data. The frontend was mainly developed by HTML, CSS and JavaScript. In order to better demonstrate the percent data of race and emotion classiﬁcation, the UI design also developed with D3.js for data visualization.
<img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture3.png">
<p align='center'>Figure 3: Data Visualization by D3.js</p>

The application also allows user to train the classiﬁers by providing feedback to classiﬁcation result. For example, if a user upload an image for gender classiﬁcation and the classiﬁcation result is “male”, he can give the feedback to indicate the result is correct by clicking the “correct” button. By doing this, this image would be stored in the male training data in the database for future training. If this result is not correct, the user can also click the “wrong” button to put this image in the female training data, which realized an actively and continuously learning by interacting with real users.

The backend of the web application was developed by Django, a Python based web framework, which allows easy implementation with TensorFlow, Keras, and OpenCV because of the same programming language. PostgreSQL is selected to be the database for storing the potential training data. Finally, the application was deployed on the cloud server by using DigitalOcean as cloud platform, Gunicorn as WSGI HTTP Server, and Nginx as reverse proxy server. The general structure of this demonstration and feedback learning system is shown as follow:
<p align='center'><img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture4.png"></p>
<p align='center'>Figure 4: User Can Provide Feedback Based on Result to Help Better Improve the Models</p><br>

<p align='center'><img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture5.png"></p>
<p align='center'>Figure 5: Structure of Demonstration and Feedback Learning System</p>

# 3 Experiment
## 3.1 Experiment Procedure
### 3.1.1 Programming Languages and libraries
For this project, we selected Python as the programming language and worked on JetBrains PyCharm. OpenCV-python library is used in detecting human faces, as well as converting RGB images into gray scaled data for training and testing. For CNN model, we explored both tensorﬂow and Keras. Tensorﬂow is used in emotion detection and Keras is used in race and gender classiﬁcation. Pandas and Numpy are the libraries that we used to preprocess the training data including data clean, normalization and hot spot.

### 3.1.2 Training and testing dataset resources
Face Place data - For gender and race classiﬁcation “Face Place” facial data is originally from Tarrlab in Carnegie Mellon University, which contains multiple images for over 200 individuals of many diﬀerent races [6]. Speciﬁcally, after data cleaning and preprocessing, it contains 252 front-face images of Hispanic people, 1971 images of Caucasian, 701 images of Asian and 330 images of African. The data for two genders is somewhat evenly distributed while data for diﬀerent races is not, which is one of the potential obstacles for further optimizing the race classiﬁcation model.

Fer2013 - For emotion classiﬁcation We investigated various human facial datasets with emotion labels, including the Japanese Female Facial Expression (JAFFE) Database, the Extended Cohn-Kanade Dataset(CK+), fer2013, GEMEP-FERA 2011, etc. We ﬁnally chose fer2013 dataset as our training dataset after considering its superior advantages over other datasets:<br>
<p align='center'><img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture6.png"></p>
<p align='center'>Figure 6: Facial image recovered from fer2013 dataset</p><br>

1. Fer2013 is one of the most widely used open source dataset in facial emotion recognition, ﬁrst created for an ongoing project by Pierre-Luc Carrier and Aaron Courville, then shared publicly for a Kaggle competition. It is easily to be downloaded and is completely for free.

2. Fer2013 is one of the largest facial emotion datasets which consists of 35887 grayscale, 48x48 sized face images with 7 emotions all labeled:

0: -4593 images- Angry 1: -547 images- Disgust 2: -5121 images- Fear 3: -8989 images- Happy
4: -6077 images- Sad 5: -4002 images- Surprise 6: -6198 images- Neutral

3. This dataset is in the form of a single “fer2013.csv” ﬁle with 35888 rows and 2305=48*48+1 columns where each row represents a face (the ﬁrst row is the header), each column records a pixel value (0 255) and the ﬁrst column is the emotion label. We do not need to spare great eﬀort in image preprocessing work and can focus more on training stage.


### 3.1.3 Parameter Conﬁguration
Fisherfaces Architecture Design 2459 images are used for training and 1000 eigenvectors are remained after PCA dimensionality reduction.

Convolutional Neural Network Architecture Design Three CNN models’ conﬁguration are listed below
<p align='center'><img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture7.png" width=400px></p>
<p align='center'><img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture8.png" width=400px></p>

## 3.2 Result and Analysis
<p align='center'><img src="https://github.com/chenxi1103/Face_Recognition_Project/blob/master/images/Picture9.png" width=500px></p>

# 4 Conclusion
In this project, we focused on three facial attributes (gender, emotion and gender) classiﬁcation and explored both PCA/LDA and CNN’s performance in solving binary and multi-classiﬁcation problems. We used tensorﬂow and Keras for CNN and chose different datasets to train the three models. The training data format includes both pure RGB image format (.png/.jpg) and .csv format. OpenCV is used in image preprocessing and Pandas and Numpy are used in data preprocessing.

We obtained perfect results in gender classiﬁcation and acceptable results in emotion and race classiﬁcation. We can improve our model in the future by adding more layers and better tuning the parameters. We can also try different training dataset such as AffectNet for emotion detection.

Our innovation for this project is that we also implemented an online web application deployed on the cloud where people can not only visit, test trained classiﬁcation models, but also provide us with their correctly classiﬁed image as the training data to improve our model.

# 5 Acknowledgements
This work would not have been possible without the academic support from Dr. Marios Savvides, Raied Aljadaany, Chenchen Zhu and Dipan Pal. Thanks for the people who actively contribute to the open source community and pattern recogition to make this happen. Thanks for the version control tool like github to make remote cooperation possible.

The main source code is available via this link: https://github.com/chenxi1103/Face_ Recognition_Project. The project demonstration web application is available via this link: http://157.230.5.57.

# 6 Reference
[1] Rizvi, Dr Qaim Mehdi. (2011). A Review on Face Detection Methods. Journal of Management Development and Information Technology. 11.

[2] Divyansh Dwivedi. Face Detection for Beginners. Towards Data Science. https:// towardsdatascience.com/face-detection-for-beginners-e58e8f21aad9

[3] Ramiz Raja (2017). Face Detection Using OpenCV and Python: A Beginner’s Guide. Super Datascience. https://www.superdatascience.com/opencv-face-detection/

[4] T. Bissoon and S. Viriri, "Gender classiﬁcation using face recognition," 2013 International Conference on Adaptive Science and Technology, Pretoria, 2013, pp. 1-4.

[5] CS231n: Convolutional Neural Networks for Visual Recognition, Stanford University, http:

//cs231n.github.io/convolutional-networks/

[6] Michael J. Tarr. Face-Place Face Database, Center for the Neural Basis of Cognition and Department of Psychology, Carnegie Mellon University, http://wiki.cnbc.cmu.edu/Face_ Place

[7] “Challenges in representation learning: a report on three machine learning contests.” I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah. https://www.kaggle.com/deadskull7/fer2013
