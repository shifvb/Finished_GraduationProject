# GraduationProject
大学本科毕业设计

Design and Implementation of Liver Segmentation and Lymphoma Detection System Based on TensorFlow

Feature extraction is an important prerequisite for image analysis and computer-aided diagnosis. It is hard to extract the hidden feature in medical images, which are complex and difficult to represent. With the rapid development of deep learning technology, it is important to use deep learning methods to proceed automatic medical image feature extraction.
Firstly, linear spectral clustering algorithm is applied in this paper to extract multi-scale superpixels. Deep convolutional neural network (DCNN) based on TensorFlow is integrated to extract features from superpixels. Secondly, the uptake of fludeoxyglucose (FDG) of liver is high in PET images, which impact on the false positive percentage of lymphoma diagnosis. Considering this issue, in order to reduce the false positive, a stacked autoencoder (SAE) is applied to identify the liver region and lymphoma area. The feature extraction and classification are implemented as web services by using Flask, which is a web microframework. The design of the system is based on MVC architecture in order to improve the portability of the system. Moreover, a distributed storage mechanism is implemented to maintain data integrity and improve the storage performance.
In order to evaluate the effect of feature extraction, this paper also implement a stacked autoencoder based on TensorFlow, which is used to classify the feature vectors extracted in the system. Totally 51,535 superpixels are used as train set and test set, half of them are the train set, and the rest of them are test set. In the case the of three-layered SAE, the liver 
recognition accuracy is 88.9%, and the lymphoma region accuracy is 96.2%. The result shows the system has good reliability and extensibility, which is of great significance for optimizing the recognition effect of medical images.

Key words: TensorFlow, CNN, Stacked Autoencoder, Feature Extraction, Medical Image Processing  

