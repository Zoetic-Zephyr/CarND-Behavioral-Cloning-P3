## Behavioral Cloning

### Zheng(Jack) Zhang jack.zhang@nyu.edu

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./writeup_images/nn.png "Model Visualization"
[image2]: ./writeup_images/normal.png "Normal Image"
[image3]: ./writeup_images/flipped.png "Flipped Image"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* models/model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py models/model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I adopted [NIVDIA's end-to-end CNN for Self Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The model consists of a three-part convolution neural network with:

1. 5x5 filter sizes, depths between 24, 36 and 48 and stride of 2x2 (model.py lines 61-63)
2. 3x3 filter sizes, depth of 64 and stride of 1x1 (model.py lines 64-65)
3. 4 fully connected dense layers of depth 100, 50, 10, and 1 (model.py lines 68-71)

The model includes RELU layers to introduce nonlinearity, and the data is normalized, and then cropped in the model using a Keras Lambda and Cropping2D layers (code line 59-60). 

#### 2. Attempts to reduce overfitting in the model

Dropout layers in order to reduce overfitting seems unecessary in this model. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21-47). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and vertically inverted images (to deal with counter-clockwize only data).

The final data set used in training and validation consists of:

|      IMAGE      | STEERING_ANGLE  |
| :-------------: | :-------------: |
|     center      |     center      |
|      left       |      left       |
|      right      |      right      |
| inverted_center | inverted_center |
|  inverted_left  |  inverted_left  |
| inverted_right  | inverted_right  |



And below are the image data before and after np.fliplr():

![alt text][image2]

![alt text][image3]

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture is Transfer Learning, a.k.a. start with a robust existing model and tailor/fine-tune it for a specific task. 

My first step was to use a convolution neural network model similar to the LeNet-5 used in the Traffic Sign Classifier project. I wasn't really expecting it to be effective, but just want to see if I can successfully train a model using the remote workspace and test on my local machine. I do find there is a version mismatching problem, and I have send an [issue](https://github.com/udacity/CarND-Term1-Starter-Kit/issues/114) to the [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) repo.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set (20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I swicthed to the NVIDIA model, and found that not only did my loss decrease tenfold, and overfitting was neglegible.

The final step was to run the simulator to see how well the car was driving around track one. Using the pre-recorded data, there were absolutely no spots where the vehicle fell off the track, even if I manually force the vehicle to drive in the opposite direction. The vehicle seems to be able to stay perfectly in the center of the road. 

#### 2. Final Model Architecture

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The project uses pre-recorded data provided by Udacity. Recording data manually tuns out to be extremly tricky without a joystick and therefore it is really hard to record high-quality training data for the model to learn. 

I utilize the center, left,  and right image data and augumented them by flipping vertically. The flipping operation is used to counter the one-way driving bias.

After the collection process, I had 8,036x3x2=48,216 number of data points. I then preprocessed this data by normalization and cropping.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as the model starts to overfits at around epoch=4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
