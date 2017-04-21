#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

The entire code are mainly listed in `Traffic_Sign_Classifier.ipynb`.

**1. Load the data**

Download the required data, and extract it into train/valid/test part. Load the data using pickle.load() function and check the shape of them. The corresponding code are listed in `cell 1` and `cell 2` in `Traffic_Sign_Classifier.ipynb`.

**2. Explore, summarize and visualize the data set**

I use shape property and np.unique() function to get basic summary of dataset, which is listed in `cell 4`. 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

And then, I perform an exploratory visualization of the dataset. First, I plot a histogram showing the inputs per class using np.bincount() and display one instance for each class (0-42). The following histogram diagram is an exploratory visualization of the data set, which shows the number of instance input for each classes.

![alt text][image1]

The corresponding code are listed in `cell 5`. Then, by using sorting function, I observer the smaller 15 classes are class [ 0 37 19 32 27 41 42 24 29 39 21 40 20 36 22] `(cell 6)`.


**3. Design, train and test a model architecture**

3.1 Preprocess the image data

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I normalize the image data using cv2.normalize() function to obtain zero mean and equal variance images `(cell 8 & 9)`. The comparison between the orginal and normalized one is shown in the following figure:

![alt text][image2.5]

As a last step, I shuffle the image data by

```
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
```



I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following 

3.2 Define model architecture

My final architecture involves a convolutional neural network (CNN) similar to that of LeNet, but with several important updates/changes. In general, the architecture incorporates two convolution layers (conv1 and conv2) followed by three fully connected layers (fc1, fc2 and fc3, fc3 is used for logit output). To help to reduce the overfitting effect, I apply dropout method: append dropout layer after fc1 and fc2 respectivey, which take the majority part of parameters in the whole network.

General Parameters:

Number of conv layers = 2
Number of fully-connected layers = 3
Size of patches for the convolutions = 5x5
Size of patches for the pooling = 2x2

1st Convolutional Layer (Input = 32x32x3. Output = 14x14x6)

The first layer is fed the 32x32x3 color image. This image is put through a 2-dimensional convolution with a stride of 1. Next, the result of the convolution is added with a bias vector and their sum is processed using the tf.nn.relu activation operator. Then, the result of this activation is put through a max pooling operator using kernal of [1,2,2,1] and a stride of [1,2,2,1]. Finally, the result of this max pooling is passed to succeeding covolution layer.

2nd Convolutional Layer  (Input = 14x14x6. Output = 5x5x16)

The second convolutional layer is identical to the first, with one main exceptions. The second layer is fed the output of the first convolutional layer. 

1st Fully Connected Layer (Input = 400. Output = 120)

The output of the second convolutional layer is flattened into 5x5x16=400 first and multiplied by a weight matrix. The result of this multiplication is added to a bias vector, and that summation is passed through the tf.nn.relu activation function.

2nd Fully Connected Layer (Input = 120. Output = 84)
The second fully connected layer is almost identical to the first fully connected, except the output dimention.

3rd Fully Connected Layer (Input = 84. Output = 43)
The network concludes by multiplying the result of the 2nd fully connected layer with a weight matrix, adding a bias, and returning the result for the logits operation to provide the final classification.

The details of the network structures is listed in `cell 12`

3.3 Training the model
I take mini-batch (BATCH_SIZE = 128) SGD approach to train the network. The number of epochs is 100. The probability in dropout is 0.5. I use Adam method to optimize the loss function (cross entropy).
At each epoch, the accuracy in training and validation are printed out. We could see the accuracy is continually increasing in most of the epoches. At the last epoch, (EPOCH 100) Training Accuracy and Validation Accuracy reach into 1.000 and 0.964 respectively, which is a quite satisfactory result. The corresponding code is listed in `cell 17`.

3.4 Validation and testing
During the training, I check the validation accuracy to prevent overfitting. The Validation Accuracy reached into 0.964 at last epoch (100), and during the whole epoches in training, we could see the Validation Accuracy continually increasing.
For testing, the accuracy is 0.936, which is quite satisfactory. The corresponding code is listed in `cell 18`

**4. Use the model to make predictions on new images**

**5. Analyze the softmax probabilities of the new images**

**6. Summarize the results with a written report**


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)






####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


