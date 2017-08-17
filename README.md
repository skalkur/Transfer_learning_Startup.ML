#  Transfer Learning using Inception V3 Model and Convolutional Autoencoder on STL-10 Dataset

## Introduction

 Deep Learning is becoming one of the most sought after skills in the current industry. Many companies are adopting AI into their produccts, emerging markets are found in Self-driving vehicles, image based product recognition and recommendation, face recognition for Snapchat filters, and many more. For all of these applications, the one major requirement is "Data"- Lots of it! To be precise, the AI models designed need Labeled data for training them. The hiccups in getting this dataset are below:
 
1. Labeled data is hard to obtain due to the limited amount available
2. Labeled data is expensive to acquire as a human (Amazon Mechanical Turk) must label them manually
    
Even if the labeled dataset were to be obtained, training a sophisticated model to the job would take a lot of time, money, and resources. The workarounds are simple.

1. **To overcome the data scarcity**: Obtaining unlabeled images of almost any class from the internet is easy and the amount is abundant. Once the unlabeled dataset is obtained of similar, correlated classes, building and training a model just to learn features (not classify) from the images is the next step. After learning the weights of the model, the limited number of labeled dataset can then be fed to the model to classify the images.

2. **To overcome the limitation of resources and time**: Pre-trained models such as Google's Inception model, and VGG16 Model are trained extensively on high power computers for weeks on ImageNet dataset and are capable of predicting 1000 classes given an image. Most of the predictions done by these models are accurate due to the depth of the model. These models can be modified slightly (discussed below) and can be made to classify images suitable for custom purposes with a far higher accuracy, with as little computation possible, and in a short amount of time. Again, this method also makes up for limited labeled dataset as the model does not need to learn weights of the images.

The above listed methods are called "Transfer Learning". Transfer learning is helpful in mitigating the above two scenarios. 

Transfer Learning with the pre-trained network always work best if the pre-trained network is fed correlated data. 

The dataset used in this project is Stanford's STL-10 Dataset (Link below). Let us take a minute to understand STL-10's intention to exist- Emphasize training semi-supervised! The dataset has 3 datasets:

1. Train X and Train Y (5000 examples)
2. Test X and Test Y (8000 examples)
3. Unlabeled X (100000 examples)

This is close to a real-world scenario. Getting unlabeled dataset from the internet is not hard. However, obtaining that many labeled data is hard, getting the exact data as the labeled data as unlabeled might be hard too. We need to make the best of what we have. Hence, STL-10.

To give a brief information about how the dataset is advised to be used (Transfer Learning POV):

* Perform Feature extraction on the Unlabeled dataset and learn weights. The unlabeled dataset does not contain the same images or same class of images as that of labeled (be it train or test) but it contains similar data
* Perform supervised learning on this model with the labeled dataset.

This notebook deals with Transfer Learning on Stanford's STL-10 dataset using Google Inception Model and also the designing a Convolutional Autoencoder (CAE) and training it learn the features of the Unlabeled dataset and then perform supervised learning on the Transfer Learning.

The dataset can be found here: https://cs.stanford.edu/~acoates/stl10/

Required Libraries: TensorFlow (v.1.2.1) and Numpy
