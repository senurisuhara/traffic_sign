# Traffic Sign Classifier
Trained model Base on LeNet-5 Architecture to identify traffic signs

The goals / steps of this project are the following:

* Load the data set 
* Explore, summarize and visualize the data set 
* Design, train and test a model architecture 
* Use the model to make predictions on new images 
* Analyze the SoftMax probabilities of the new images 
* Summarize the results with a written report

## Dataset and Repository

This lab requires:

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.

2. Clone the project, which contains the Ipython notebook and the writeup template.

```sh git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project cd CarND-Traffic-Sign-Classifier-Project jupyter notebook Traffic_Sign_Classifier.ipynb ``` Or

3. !git clone https://bitbucket.org/jadslim/german-traffic-signs

Test data was obtained by following URLs:

• https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg

• https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg

• https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg

• https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg

• https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg 

## 1. Data Summary

Number of training examples are 34799 and number of testing examples are12630. Image shape was 32*32 with 3 color channels. And we have got 43 different classes.

## 2. Preprocessing Mainly there was 2 preprocessing techniques are used

• Convert BGR to Grayscale 

This conversion mostly ruled out unwanted feature maps that are above to create by using 3 color spaces. Because of using grayscale, we are able to conserve all the necessary features (shapes, edges, corners) without loosing much detailed. Also, it really speeds up the convolution process/training

• Histogram Equalization

Not all the images are same even from the same class. They were under different lighting conditions and that was one of the main problems. By using histogram equalization, we were able to overcome this matter successfully

## 3.Model Architecture

Final model that is created is a result of 3 attempts. First, we tried the pure LenNet-5 architecture as it is. But the model tends to overfits. Then we increase the number of convolutional layers and number of kernels by adding two additional convolutional layers with increased number of kernels. This almost done the job, but only gives around 60-80% accuracy for new images, with validation accuracy of around 96%. So, by using trial and error we are able to overcome this matter by simply removing a dropout layer from the original model. Then it has given more than 99.8% validation accuracy and successfully identified all the test images/new images from the internet.

Final Model summary is as follows

![](/README/model.png)

## Please refer the TrafficSigns.ipynb notebook for better analysis of past models and their error rates

## 4. Data Augmentation

Of course, in order to increase the accuracy and to increase the size of training data we use data augmentations techniques like rotation, shear and zoom the image.

## 5. Model Training

Model training is done by using Google Colab and its GPU. This has increased the parallel computing time very much and model was trained within few minutes. We use 'Adam' optimizer for the training with 0.001 learning rate (this is of course by trial and error result) and we use ‘categorical cross entropy’ as the loss function. Finally, with batch size of 50 and with simply just using 20 epochs were able to beat the challenge that Mr. Sebastian Thrun has
given at the beginning

Here are the graphs for the final trained model

![](/README/Picture1.png)

![](/README/Picture2.png)

## 6. Acquiring New Images By using above URLs, 
We test over trained model and it has finally given 100% accuracy for all 5 images

These are some images after they preprocessed, before fed into the model for classification:

!["Image 1"](/README/Picture3.png)

## img 2

!["Image 2"](/README/Picture4.png)

These are the SoftMax probabilities and predictions that are obtained for these 2 specific images: 

Probabilities: 

(k=5) 

Top 5 softmax probabilities of imag1 are:  [9.744509e-01 1.789298e-02 5.325369e-03 1.5164 02e-03 5.202094e-04]

Top 5 softmax probabilities of imag1 are:  [9.9999523e-01 4.7153135e-06 1.4311459e-08 6.8975176e-10 1.9309242e-10]
 
 
Predicted Classes: 

(k=5)

Top 5 softmax predictions of imag2 are:  [34 38 40 12 35] 

Top 5 softmax predictions of imag2 are:  [1 2 5 7 4]
 
## 7.	Test a Model on New Images 
The images that are used in training are 3 channel,32*32 sized images. (model input shape is 32,32,1) While the test images that has taken from the internet has higher resolution and detailed than the trained images (1300*960*3 etc.). So, when these images are fed into the model, there are resolution errors. So, after we do gray scaling and histogram equalization, the resize and reshape the new images to the original input size of the model is must. (which is (32,32,1) Because of our new test images are high resolution, when it’s shrunk to (32,32,1), there are no significant detail loss. But again, the input size doesn’t matter if we trained using Fully Convoluted Neural Network. (Because of no dense layers except the output) 
 
## 8.	Top softmax probabilities 
 
As it can see the top certainty levels of img1 to img5 are more than 97% percent for all the 5 images. The second prediction, for an example, in image 3 is only 0.0000065031809. In percentage it is 0. 00065031809%. So, we can see how well the model is predicted the correct class. Also, in the above example the img3 was indeed indicates a slippery road. That was the model’s first choice for this image and the second choice was Bicycle crossing. But as mentioned above the probability was very small compared to the first probability. The 3 rd. (softmax) prediction was wild animal crossing with probability of 0.00000000000053416538. Rest of two probabilities are negligible compared to the statistics of the original output. So, by comparing we could say that the certainty of this model is reasonably accurate. 
