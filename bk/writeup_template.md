#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram1.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image2.5]: ./examples/normalized.png "normalized"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./output_images/01-05.png "Traffic Sign 1--5"
[image5]: ./output_images/bar_charts.jpg "bar_charts"
[image6]: ./test_german_traffic_sign/03.jpg "Traffic Sign 3"
[image7]: ./test_german_traffic_sign/04.jpg "Traffic Sign 4"
[image8]: ./test_german_traffic_sign/05.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

The entire code are mainly listed in `Traffic_Sign_Classifier.ipynb`.

**1. Load the data**

Download the required data, and extract it into train/valid/test part. Load the data using pickle.load() function and check the shape of them. The corresponding code are listed in `cell 1` and `cell 3` in `Traffic_Sign_Classifier.ipynb`.

**2. Explore, summarize and visualize the data set**

I use shape property and np.unique() function to get basic summary of dataset, which is listed in `cell 5`. 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

And then, I perform an exploratory visualization of the dataset. First, I plot a histogram showing the inputs per class using np.bincount() and display one instance for each class (0-42). The following histogram diagram is an exploratory visualization of the data set, which shows the number of instance input for each classes.

![alt text][image1]

The corresponding code are listed in `cell 6`. Then, by using sorting function, I observer the smaller 15 classes are class [ 0 37 19 32 27 41 42 24 29 39 21 40 20 36 22] `(cell 7)`.


**3. Design, train and test a model architecture**

3.1 Preprocess the image data

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.
For this traffic sign classification project, color information is essential for accurate classification. Thus, I did not convert the data into grayscale. I use the orginal RGB images.

While the range of intensity has obvious impacton optimization operation in training step. Normalization step is desired.
Thus, I normalize the image data using cv2.normalize() function to obtain zero mean and equal variance images `(cell 8 -- 10)`. The comparison between the orginal and normalized one is shown in the following figure:

![alt text][image2.5]

As the very early step, I already shuffle the training data by shuffle() in `cell 2`.

```
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
```

3.2 Define model architecture
As the problem (traffic sign classfication) is a typical image recognition problem, it's the best to try CNN-based network model. As the problem task only involes 43 classes and around 30,000 training images, very deep CNN models such as VGG/Inception/ResNet are not suitable. For simplicity and effectiveness, I choose the model based on LeNet 5.
My final architecture involves a convolutional neural network (CNN) similar to that of LeNet 5, but with several important updates/changes. In general, the architecture incorporates two convolution layers (conv1 and conv2) followed by three fully connected layers (fc1, fc2 and fc3, fc3 is used for logit output). To help to reduce the overfitting effect, I apply dropout method: append dropout layer after fc1 and fc2 respectivey, which take the majority part of parameters in the whole network.

General Parameters:

Number of conv layers = 2
Number of fully-connected layers = 3
Size of patches for the convolutions = 5x5
Size of patches for the pooling = 2x2

1st Convolutional Layer (Input = 32x32x3. Output = 14x14x6)

The first layer is fed with 32x32x3 color image. This image is put through a 2-dimensional convolution with a stride of 1. Next, the result of the convolution is added with a bias vector and their sum is processed using the tf.nn.relu activation operator. Then, the result of this activation is put through a max pooling operator using kernal of [1,2,2,1] and a stride of [1,2,2,1]. Finally, the result of this max pooling is passed to succeeding covolution layer.

2nd Convolutional Layer  (Input = 14x14x6. Output = 5x5x16)

The second convolutional layer is identical to the first, with one main exceptions. The second layer is fed the output of the first convolutional layer. 

1st Fully Connected Layer (Input = 400. Output = 120)

The output of the second convolutional layer is flattened into 5x5x16=400 first and multiplied by a weight matrix. The result of this multiplication is added to a bias vector, and that summation is passed through the tf.nn.relu activation function. To reduce overfitting effect, dropout layer (prob.=0.5) is added. 

2nd Fully Connected Layer (Input = 120. Output = 84)
The second fully connected layer is almost identical to the first fully connected, except the input and output dimention. 

3rd Fully Connected Layer (Input = 84. Output = 43)
The network concludes by multiplying the result of the 2nd fully connected layer with a weight matrix, adding a bias, and returning the result for the logits operation to provide the final classification.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input = 400. Output = 120			|
| RELU					|												|
| Dropout					|				Prob. = 0.5								|
| Fully connected		| Input = 120. Output = 84						|
| RELU					|												|
| Dropout					|				Prob. = 0.5								|
| Fully connected(Logits)				| Input = 84. Output = 43   									|


The details of the network structures is listed in `cell 12`

3.3 Training the model
I take mini-batch (BATCH_SIZE = 128) SGD approach to train the network. The number of epochs is 100. The probability used in dropout is 0.5. I use Adam method to optimize the loss function (cross entropy). Learning rate is 0.001.
At each epoch, the accuracy in training and validation are printed out. We could see the accuracy is continually increasing in most of the epoches. At the last epoch(EPOCH 100), Training Accuracy and Validation Accuracy reach into 1.000 and 0.964 respectively, which is a quite satisfactory result. The corresponding code is listed in `cell 17`.

3.4 Validation and testing
During the training, I check the validation accuracy to prevent overfitting. The Validation Accuracy reached into 0.969 at last epoch (100), and during the whole epoches in training, we could see the Validation Accuracy continually increasing.
For testing, the accuracy is 0.949, which is quite satisfactory. The corresponding code is listed in `cell 18`.

In summary, my final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.9% 
* test set accuracy of 94.9%

From the obtained training/validation/testing accuracy,  the chosen model (LeNet5 with extenstion) is a suitable one in this project.

**4. Use the model to make predictions on new images**
Then, I test the model on new German traffic sign images downloaded from internet. The following 5 pictures are the test images. (code: `cell 19`)

![alt text][image4]

As these test images are captured from real scenario with diverse background, so it is good to use them for testing the generalization capability of my trained network model. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)      		| Speed limit (50km/h)   									| 
| Yield					| Yield											|
| Stop Sign      		| Stop sign   									| 
| Children crossing     			| Children crossing 										|
| Pedestrians	      		| Pedestrians					 				|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This keeps the consistency with the accuracy on the test set of GT-SRB, which reaches into more than 94 % in testing accuracy. (code: `cell 20-22, 24`)

To check the certainties in each prediction, bar charts are used. The corresponding code and the numerical probability values are located in `cell 26`.
The following bar charts are used to visualize the softmax probabilities for each prediction. In each bar chart, the corresponding top 5 softmax probabilities for each image along with the sign type of each probability are shown.

![alt text][image5]

We can see the top one probabilities in each prediction is very high against other top-5 probabilities, which shows that my model is very certain about its output.
