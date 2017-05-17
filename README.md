
---

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/fahrrad_fussgaengerweg.jpg "cycle and pedastrian path sign"
[image5]: ./examples/give_way.jpeg "yield Sign"
[image6]: ./examples/halte_verbot.jpg "No stopping sign"
[image7]: ./examples/speed_limit70.jpg "Speed limit 70 Sign 4"
[image8]: ./examples/stop.jpg "Stop Sign 5"

---
# Traffic Sign Classification with Deep Neural Networks
## Aim

This project aims to build a road sign classifier based on a neural network. The [...] dataset of german road signs was used to train the network. The minimal main requirement for this project was to reach a minimum of 93% validation accuracy on a provided test set but I personally wanted to achieve at least 96% testing accuracy. It should be noted that this may or may not be a good approach to build safe self driving cars depending on the driving behaviour the model will cause. E.g. misinterpreted right of way regulating signs could have catastrophic consequences whereas detecting a speed limit of 70 instead of 50 might cause a speeding ticket but in general can be recovered from more easily and will likely be offset by general awareness of the cars surroundings and the car would behave based on these factors as well.
However this model is not going to be plugged into any car in this form and serves the purpose of learning to build neural networks with tensorflow and deploy them to a specific problem (image classification).

## Project Source Code and Setup

The project source code has been commited to Github and can be found in my personal repository. [project code](https://github.com/FlorianGraef/traffic-sign-classifier)
The entire code is located in the Traffic_Sign_Classifier.ipynb Jupyter notebook.
The training dataset used is the MNST data set (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) of german traffic signs.  It was downloaded from Udacity (https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) and already split in pickle files for train, validation and test datasat.

## Data Set Summary & Exploration

### 1. Basic data set summary

As mentioned above, the data set was provided by udacity in pickle files and using basic python revealed the following properties of the dataset:
* training set - 34799 images
* validation set - is 4410 images
* test set - 12630 images
* Each traffic signs image has a resolution of 32 by 32 pixels and 3 colour channels (RGB)
* There are 43 different types of traffic signs in the data set. This means that the classifier needs to assign each image one of 43 labels.

### 2. Exploratory dataset visualization

To get an understanding for the dataset I looked at a sample of images from the dataset. I noticed that, in the unshuffled dataset subsequent images show a series of the same traffic sign extracted from videos. The images show the road signs fairly well centered and zoomed in on the road sign. This means that the scope of this project is limited to classification only. The first step prior to the classification, identifying the traffic sign object in the, not just proverbial, bigger picture, has already been taken care of. 
For better characterization of the dataset the distribution of samples per class (traffic sign) were plotted in a bar chart depicted below.

![alt text][image1]
The bar chart shows that the distribution of samples per class is very uneven. It ranges from 180 images for sign 0 (Speed limit 20km/h),  19 (Dangerous curve to the left), 37 (Go straight or left) to 2010 images of the 50 km/h speed limit sign with many.

## Design and Test a Model Architecture

### 1. Pre-processing
In first classifaction runs with the LeNet architecture I achieved ~93% validation accuracy using the images as provided. It was mentioned in the neural network lectures that normalizing the images helped the gradient descent to converge. As well having a zero mean of input values helps to keep activation functions from near zero means early in the learning process.
Consequently the input values were centered around zero within a range of +/- 0.5. In addition historgram nromalization was applied on each colour-channel. The histogram normalaization causes the colour spectrum more widely to be facilitated. This typically yields more contrast and better detail in images (http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html).
These two techniques combined increased the validation accuracy with LeNet to almost 95%.

### 2. Data Augmentation
To increase the performance further after establishing my own network architecture the data was augmented by adding random rotations and randomly altering the brightness of the images. The idea behind this is to add noise to the data to make the model more robust and generalize better. Furthermore, more training data typically allows more accurate models. 
To balance the dataset data augmentation could have applied selectively but the assumption was made that the validation and testing dataset possessed a distribution simliar to the training set. It could as well be argued that the MNST dataset represents the distribution of traffic signs realistically and does hence not need to be adjusted. Henceforth all data augmentation was applied uniform. In a road sign classifier deployed in a self driving car the classifier could be optimized so that signs that have the most dangerous consequences when not classified correctly get augmented more than other classes. However this was not a goal laid out for this project.
This resulted in a training set of 313191 (nine times the size of the initial training set). Here is an example of a normalized and augmented image.

![alt text][image2]

### 3. Neural Network Architecture
After first steps with LeNet and reaching about 95% Validation accuracy with preprocessing I ventured off to build my own network inspired by the VGG architecture (https://arxiv.org/pdf/1409.1556.pdf). I chose the VGG architecture because it was highlighted in the lectures how variable it is and often serves well as a starting point for image classification problems. Since traffic sign classification is a special case of image classification it was thought to be well suited for this problem.
The basic structure of VGG consists of, almost exclusively, 3x3 convolutions and repeats a pattern of two convolutional layers with the same filter count, starting with 32 and doubling every two or three convolutional layer. Due to GPU memory constraints (2GB) I was limited to 4 convolutional layers  followed by 3 fully connected layers and started with 32 filters instead of 64 filters in the first two convolutional layers. This was considered to be acceptable because the input data here was much smaller (32x32x3) compared to the original VGG input of 224x224x3. It was realized later that the 4th convolutional layer was never used in my code. Upon training with the 4th layer included accuracy dropped slightly. The 4th convolutional layer was then replaced by just max pooling, as it was the case in the architecture prior to discovery of this mishap.  
I decided to pass the output of the marked layers, flattened and concatenated, into three fully connected layers that would return the label predictions. 
The final model is described below.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, 32 filters	|
| RELU					|												|
| Max pooling*	      	| 2x2 pooling, valid padding, 1x1 stride		|
| Dropout				| keep probability 0.8							|
| Convolution 3x3	    | 1x1 stride, valid padding, 32 filters   		|
| RELU					|             									|
| Max pooling*			| 2x2 pooling, valid padding, 2x2 stride 		|
| Convolution 5x5		| 1x1 stride, valid padding, 64 filters			|
| RELU					|												|
| Max pooling*			| 2x2 pooling, valid padding, 1x1 stride		|
| Max pooling*			| 2x2 pooling, valid padding, 2x2 stride 		|
| flatten + concat		| Flatten and concat output of marked (*) layers|
| Fully connected		| 700 nodes										|
| RELU					|												|
  Fully connected		| 300 nodes										|
| RELU					|												|
| Fully connected		| 43 nodes										|



### 4. Training the network
The model was trained for 130 epochs with a batch size of 256 and a learning rate of 0.001. The learning rate was taken as a starting point from the LeNet lab. 0.0001 was tested as well but was found to extend training time by a lot. The number of epochs was determined by training the network until the validation accuracy did no longer improve. This was used to avoid overfitting and cut training time. The batch size of 256 was chosen because it allowed a high GPU facilitation and improved training time.
The Adam optimizer was used as it was said to be a good overall choice in the udacity course material as well as in this more exhaustive comparison (http://sebastianruder.com/optimizing-gradient-descent/index.html#whichoptimizertochoose).

### 5. Developing the model
The approach taken was to first start with the LeNet architecture, as it is a powerful yet simple image classification architecture, to get a starting point and develop the skills building neural networks. Together with image normalization this lead to 95% validation accuracy. This enabled further experiments with other constructs in neural networks. Testing was done with inception modules, based on GoogLenet, but did not yield significantly better results but required much longer training times increasing the iteration time which slowed down overall evolution of the model. Furthermore graphics memory was quickly exhausted. 
This lead to a switch to a loosely VGG inspired model with output of layers not only passed into the next layer but as well passed directly into the first fully connected layer combined with data augmentation as described above this resulted in 98% validation accuracy and 97% testing accuracy in the final model.
Convolutional layers were used as core component, as a lot of successfull image classification networks do. There weights shareing cross the location of the images helps to detect features in the image regardless of where it is located. This follows the reasoning that a certain feature most of the times has the same meaning no matter where it is located and helps the classifer to generalize. 
The dropout layer implemented early in the model as well serves the purpose of making the model generalize well by randomly switching nodes to zero. The intention of this is for the model to not rely on any single feature too much. This was implemented in an early layer to facilitate a trickle down effect for later layers. In addition this avoids overfitting.
Overall the final outcome of ~98% validation accuracy and 97% test accuracy are fairly close indicating that overfitting did not take place. However looking at the validation accuracy hovering around 98% for some time suggest that even earlier termination can be considered.

## Testing the Model on New Images

To test the model on completely new images of traffic signs I downloaded five images from the web found through the google image search. Images that were not included in the 43 classes of the dataset were included as well to see how the model does on unknown images. The images below were then resized, disregarding the original aspect ratio, to 32x32 pixel to comply with network input layer. This resulted in varying degrees of distortion.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and third image are challenging because they are not on the list and show two signs each. The second sign could be a bit tricky because it differs from a square and will get distorted during resizing. The stop and 70km/h speed limit should be easy to classify.

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| pedastrian/bike way	| Speed limit 30 km/h   						| 
| Yield     			| Yield 										|
| No stopping			| Priority Road									|
| Speed limited 70km/h	| General Caution								|
| Stop					| Stop			      							|

The model correctly predicted the stop and yield sign. That is two out of five equating 40% accuracy. It was not expected to classify the pedastrian/bike way and the no stopping sign as they are not contained in the set of classes. The 70km/h speed limit sign however was expected to be classified correctly. This shows significant underperforming compared to the validation and test accuracy on the MNST dataset but is expected due to two traffic signs which are not covered by the MNST data set.
Looking at the top 5 softmax probabilities for each predictions reveals that the model is 100% certain about each prediction to be the first choice. For the first traffic sign (pedastrian/ bike way) a very minuscule 6.40968491e-39 probability is reported for the sign to be a 70km/h speed limit. With the 1.0 probability of the first guess this seems to exceed 100% but may be attributable to floating point calculation or rounding issues.
