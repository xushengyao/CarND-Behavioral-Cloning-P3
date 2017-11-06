# **Behavioral Cloning**

## Writeup Report
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Nvidia_model.png "Model Visualization"
[image2]: ./images/center_example.jpg "example"
[image3]: ./images/center_1.jpg "Recovery Image"
[image4]: ./images/center_2.jpg "Recovery Image"
[image5]: ./images/center_3.jpg "Recovery Image"
[image6]: ./images/center_reverse.jpg "reverse Image"
[image7]: ./images/before_flip.png "Flipped Image"
[image8]: ./images/after_flip.png "Flipped Image"
[image9]: ./images/MSE.png "MSE"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My [project](https://github.com/xushengyao/CarND-Behavioral-Cloning-P3) includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network
* [writeup_report.md](writeup_report.md) summarizing the results
* [video.mp4](video.mp4) video recoding of the vehcile driving autonomously for two laos around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and [drive.py](drive.py) file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The base model I am using is the [Nvidia](https://arxiv.org/pdf/1604.07316.pdf) model, which consists of 9 layers, including a normalization layer, 5 convolution layers and 3 fully connected layers.

![Model Visualization][image1]

Here, the data was normalized in the model using a Keras lambda layer. Also, in order to choose an area of interest, Cropping2D layer of Keras was used to exclude the sky and the hood of the car. For the convolution layers, I followed the Nvidia's model and used strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

Actually, before trying Nvidia model, I used LeNet but got a bad result. Then I switched to Nvidia model and found out it's relatively simple and the resulted MSE is very low.

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers (keep_prob = 0.7) in order to reduce overfitting (code line 102 and 104)

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (code line 109).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because it has been used for

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model and added two dropout layers after the Flatten layer.

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (code lines 94-107) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

|      Layer      |                Description                 |
|:---------------:|:------------------------------------------:|
|      Input      |            160x320x3 RGB image             |
|     Lambda      |               normalization                |
|   Cropping2D    |     cropping=(70,25), outputs 65x320x3     |
|  Convolution2D  | 2x2 stride, 5×5 kernel,  outputs 31x158x24 |
|  Convolution2D  |  2x2 stride, 5×5 kernel, outputs 14x77x36  |
|  Convolution2D  |  2x2 stride, 5×5 kernel, outputs 5x37x48   |
|  Convolution2D  |      3x3 non-stride, outputs 3x35x64       |
|  Convolution2D  |      3x3 non-stride, outputs 1x33x64       |
|     Flatten     |                outputs 2112                |
|     Dropout     |              keep_prob = 0.7               |
| Fully connected |                outputs 100                 |
|     Dropout     |              keep_prob = 0.7               |
| Fully connected |                 outputs 50                 |
| Fully connected |                 outputs 10                 |
| Fully connected |                 outputs 1                  |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![example][image2]

Then I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to get back to the center of the road if it gets off to the side.  These images show what a recovery looks like starting from completely off the side, partly off the side, back to the center:

![Recovery Image"][image3]
![Recovery Image"][image4]
![Recovery Image"][image5]

Then I recorded one counter-clockwise lap around the track to help the model to combat the bias which is towards left turns. An example image for the reverse driving is shown below:

![reverse image][image6]

To augment the data sat, I also flipped images and angles thinking that this would increase the size of the training set and helping with the left turn bias. For example, here is an image that has then been flipped:

![flipped image][image7]
![flipped image][image8]

Also, I randomly changed the brightness and of the image to

I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.The final MSE result for training sets and validations sets is shown below:

![MSE][image9]
