# Semantic-Segmentation-With-Cambridge-Labeled-Objects-In-Video
Implement and compare three deep convolutional neural network architectures – FCN, Deeplab, and UNet – for semantic segmentation on the Cambridge labeled objects in video dataset.

## Goals
Implement and compare 3 convolutional neural network architectures: FCN, Deeplab, UNet.
Achieve over 80% accuracy on the test set.
Optimise models to reduce overfitting.

## Results
1. achieved the highest test accuracy of 86% with the UNet model.
2. Reduced overfitting through data augmentation and early stopping.
3. Compared 3 architectures – UNet was most efficient and accurate.
4.Faced computational challenges due to limited GPU access: optimised code and model size to improve training time.

## Problem Statement
Semantic segmentation of images is an important task in computer vision, but it remains challenging especially for complex scenes with occluded objects. Existing methods can be computationally expensive and may lack accuracy. There is a need for efficient deep learning architectures that can accurately perform semantic segmentation.

## Introduction
Semantic image segmentation is the task of classifying each pixel in an image from a predefined set of classes i.e., a computer vision operation aiming to assign semantic labels to each pixel in an image. An arduous task that requires the model to learn to identify and segment objects in images, even when they are partially occluded or have similar appearances.
There are several different approaches to semantic segmentation, but one of the most popular is to use deep convolutional neural networks (CNNs). CNNs can learn hierarchical features from images, which can be used to identify and segment objects. Some of the most popular CNN architectures for semantic segmentation include FCN, DeeplabV3 and UNet and they we have utilised these 3 in this project.
In this project, we have utilised the Cambridge labelled objects in video dataset. This is a dataset of videos that have been annotated with semantic labels for each pixel. It consists of 700 images and 32 classes, including cars, pedestrians, and buildings. The images in this unique dataset were captured from a car driving through the iconic town of Cambridge in the United Kingdom. The dataset is popularly used for training and evaluating semantic segmentation algorithms. Below is a summary of the dataset:

1. The images are 360x480 pixels in size.
2. The frame rate is 30 frames per second.
3. The images are labeled with 32 classes, including cars, pedestrians, and buildings.
4. The dataset is split into 367 training images, 101 validation images, and 233 test images.

## Methodology
In this project, we first create a function to get all image directories, read images and masks in separate tensors. It takes the images in the dataset, separates the files into two lists, sorts
the lists of frames and masks, creates file paths for each frame and mask using the os.path.join(), converts them to tensors and stores them in respectively named folders.
The folder effectively splits the training dataset into train and validation data in a ratio of (90:10). This is followed by extracting the class definitions for a text file containing them. This is done by a function that parses through all elements in the text file and returns a tuple of 2 elements.
([(64, 128, 64), (192, 0, 128), (0, 128, 192), (0, 128, 64), (128, 0, 0)],
['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building']) Then 2 functions, rgb_to_onehot() and onehot_to_rgb() takes an RGB image as input and returns a one-hot encoded image and vice versa respectively. The one-hot encoded image is a tensor that represents the class of each pixel in the image. The classes are represented by a vector of numbers, where each number represents a different class.
To prevent overfitting, we create functions for data augmentation. The functions are used to generate batches of training and validation data for the semantic segmentation model. The following random transformations are applied to the data:
1. The functions load the training and validation images and masks from the DATA_PATH directory.
2. The functions resize the images and masks to 224x224 pixels.
3. The functions apply random augmentations to the training images, such as rotation, cropping, and flipping.
4. The functions one-hot encode the training masks.
5. The functions yield a batch of training images and masks.
6. The functions repeat steps 4-5 until all of the training images and masks have been processed.
7. The functions yield a batch of validation images and masks.
8. The functions repeat steps 6-7 until all of the validation images and masks have been processed.

## Architectures

# FCN - 
METHODOLOGY
In this project, we first create a function to get all image directories, read images and masks in separate tensors. It takes the images in the dataset, separates the files into two lists, sorts
the lists of frames and masks, creates file paths for each frame and mask using the os.path.join(), converts them to tensors and stores them in respectively named folders.
The folder effectively splits the training dataset into train and validation data in a ratio of (90:10). This is followed by extracting the class definitions for a text file containing them. This is done by a function that parses through all elements in the text file and returns a tuple of 2 elements.
([(64, 128, 64), (192, 0, 128), (0, 128, 192), (0, 128, 64), (128, 0, 0)],
['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building']) Then 2 functions, rgb_to_onehot() and onehot_to_rgb() takes an RGB image as input and returns a one-hot encoded image and vice versa respectively. The one-hot encoded image is a tensor that represents the class of each pixel in the image. The classes are represented by a vector of numbers, where each number represents a different class.
To prevent overfitting, we create functions for data augmentation. The functions are used to generate batches of training and validation data for the semantic segmentation model. The following random transformations are applied to the data:
1. The functions load the training and validation images and masks from the DATA_PATH directory.
2. The functions resize the images and masks to 224x224 pixels.
3. The functions apply random augmentations to the training images, such as rotation, cropping, and flipping.
4. The functions one-hot encode the training masks.
5. The functions yield a batch of training images and masks.
6. The functions repeat steps 4-5 until all of the training images and masks have been processed.
7. The functions yield a batch of validation images and masks.
8. The functions repeat steps 6-7 until all of the validation images and masks have been processed.

# Deeplab - 
DeepLabv3 is an extension of FCN that was introduced in the paper "DeepLabv3: Semantic Image Segmentation with Deep Convolutional Nets" by Chen et al. (2017). DeepLabv3 uses
several techniques to improve the performance of FCN, including atrous convolution, spatial pyramid pooling and residual connections.

# UNet - 
UNet stands for universal network. UNet is a convolutional neural network architecture that was introduced in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al. (2015). Originally designed for biomedical image segmentation, it has also been used for other types of image segmentation, such as natural scene segmentation.
UNet is a symmetric network that consists of an encoder and a decoder. The encoder is responsible for extracting features from the input image, while the decoder is responsible for upsampling the features and generating a pixel-wise prediction.
Our UNet function takes three inputs. The number of filters to use in the convolutional layers, a boolean flag that indicates whether to use batch normalization the dilation rate to use in the convolutional layers and then outputs a model object.
It starts by defining the input layer of the model. The input layer has a batch size of 5 and a shape of (224, 224, 3). The function then defines the encoder part of the model. The encoder part of the model consists of five convolutional blocks. Each block consists of two convolutional layers with a kernel size of 3 and a padding of same. The first convolutional layer in each block has a stride of 2, which downsamples the feature map by half. The second convolutional layer in each block is followed by a batch normalization layer and a ReLU activation layer.

## Results
The three models FCN, Deeplab, and UNet were trained with the deep learning effective optimizer Adam, the cross-entropy loss function, and metrics such as dice coefficient, accuracy, and MeanIOU.
Looking at the results of the three models, the training loss generally decreases as the training progresses. This indicates that the models are learning and improving their ability to make accurate predictions.
The FCN model has the highest training loss among the three models, with a final loss of 0.5610. The Deeplab model has a lower training loss of 0.2484, while the UNet model has the lowest training loss of 0.1023.
FCN achieved a mean IoU of 0.4881, a dice coefficient of 0.8883, and an accuracy of 85% during training. The model’s training time was 1233.36674094 seconds. However, its accuracy on the test set was only approximately 49%, which indicates that the model may be overfitting on the training set. This low accuracy on the test set suggests that FCN needs to be further optimized and regularized.
Deeplab achieved a mean IoU of 0.5325, a dice coefficient of 0.6699, and an accuracy of 85.8% during training. Its accuracy on the test set was 69%, which is higher than FCN. This model’s training time was 950.465908766 seconds. This indicates that Deeplab has better generalization capability and is less prone to overfitting. However, the model may still require further tuning to achieve higher accuracy.
UNet achieved the highest accuracy of the three models on the test set, with 86%. During training, UNet achieved a mean IoU of 0.4848, a dice coefficient of 0.8350, and an accuracy of 86.8%. The model's training time of 436.090950966 seconds was also better than the other two models, which suggests it is the most computationally efficient.
Overall, the results suggest that UNet is the most effective model among the three, achieving the highest accuracy on the test set. Deeplab also showed promising results, with better generalization performance than FCN. Further fine-tuning and optimization may be necessary to improve the performance of all three models.

## Conclusion
In conclusion, semantic segmentation is an essential task in computer vision that allows us to identify and classify each pixel in an image from a predefined set of classes. In this project, we used three popular architectures: FCN, Deeplab_v3, and UNet, to perform semantic segmentation on the Cambridge labelled objects in video dataset. We created functions for data preprocessing, data augmentation, and for converting RGB images to one-hot encoded images and vice versa. We also created a pipeline to generate batches of training and validation data for the models.
The results showed that all three models achieved promising performance on the validation set. However, further improvements can be made by tuning the hyperparameters especially with the availability of a gpu and time to explore many techniques. We also identified some of the challenges that need to be addressed for semantic segmentation, such as its high computational cost and the need to improve accuracy for images with complex scenes or partially occluded objects.
