# WHOSE IS VANGOGH
A deep learning project to submit for project showcase challenge in Facebook Pytorch Scholarship Challenge. I make this project to test my knowledge on what I learned and to satisfy my enthusiast of bringing what I learn into solving some real problems.

# Introduction
In the art lessons at secondary high school, the teachers usually showed me pictures from famous artists in the past to teach us about style at that time as well as to give me the inspiration. To be honest I love these painting but always have trouble telling which paintings belong to whom artist. However there are two artists I can immediately identify their style are Van Gogh and Picasso, I guess you know the reason.

When taking Introduction to Deep Learning with Pytorch on Udacity, there's a lesson about tranfering style from a painting to a photos. For me this is a good lesson but the application is not interesting enough, instead I remember what struggle me in the past (categorizing artist's painting) and wanna build a model to solve it, using convolutional neural network.

To start with this project, I want to begin with only two categories: Van Gogh and Non-Van Gogh. The model can be extended to more categories (more artists) later on.

# Deep learning framework used
Pytorch with Nvidia CUDA

# Image dataset
## Getting the data

I train my deep learning model on three differents dataset. Why 3? Because after training the dataset with my self-built model, I figured out that there might be a problem with the dataset itself that affects the model accuracy. The first dataset was taken from Kaggle which contains paintings from Van Gogh and other artists. After training, my model gave high accuracy on identifying Van Gogh paintings but low on other artists' paintings. Checking the dataset, I realized that non-Van Gogh's paintings came from different artists so they do not have a unified style. Therefore the model might find it hard to generalize.

Link to first dataset:
```sh

```

Because of this, I went on to create a new dataset which uses the same Van Gogh's paintings from the first one, but replace all non-Van Gogh paintings by ones from a single artist. The artist that I chose is Aja Kusick (https://www.instagram.com/sagittariusgallery/?hl=en). Her paintings adopt Van Gogh's style and mix it with modern objects like characters from cartoons or movies (like Starwar). With this dataset, I predict that the accuracy will be higher.

Link to the second dataset:
```sh

```

How about the third one ? Well, I'm a curiosity one so I would like to see what if there's less variance in the non-Van Gogh dataset, will the model perform better compared to its performance over the first one. This time I chose Picasso as I figured out his paintings belong to two styles, according to me. 

Link to the third dataset:
```sh

```

## Spliting data for training and testing
All three datasets are split into training and testing at 80/20 ratio.

## Augumentation
I apply only one method of augumentation which is Random Resize Crop

## Normalization
[0.5, 0.5, 0.5] is used for both mean and std

## Loading dataset
The image dataset is loaded using dataloader with bath_size=32

```python
transform
```

# The models
I built two models to solve the challenge. One is a self-built convolutional neural network and the other one reused a pretrained neural network: Resnet18. I train both models on three datasets to compare the performance.

## Self-built neural network
This model is built with three convolutional layers and two fully connected layers. Each convolutional layers has a padding of 1 and stride of 1. Kernel size is 3 for all three convolutional layers. After each layer, I apply a ReLU activation function and insert a maxpooling layer (4x4) to reduce the dimension size and increase the depth. Through each layers, the input depth increases from 3 (RGB image) to 16, 32 and then 64.

As the input is 512, the output volume of these convolutional neural networks is 64*8*8 (512/4/4/4 = 8). This output serves as input for the next two fully connected layers, which will output class score for vg (Van Gogh) or nvg (non-Van Gogh) paintings. 

```python
```

## Resnet18
A pre-trained Resnet18 model is downloaded using torchvision. I freeze all the convolutional layers and adjust the fully connected layer to fit with my output.

```python
```

# Training the network
In this Github repositories, I provide the code for both self-built and resnet model. As the self-built one is pretty simple, I don't provide the checkpoint file as it takes only 10 to 15 minutes to train the model from scratch. With Google Colab one, I provide both the code with output and the checkpoint file to quickly reproduce the result.

# Result and key learning
I train the self-built neural network with 10 epochs for each dataset. The trainning is done on CPU as my GPU does not have enough memory to run. 

With first dataset, the accuracy varies from 65% to 70% on the test set. The model performs pretty well on identifying Van Gogh's paintings with 90% accuracy but struggles to spot non-Van Gogh ones. As dicussed above, I guess the problem stays with variance in styles of the non-Van Gogh paintings. Of course there could also be issues with how I setup the deep learning model but at this stage I cannot confirm the later theory. 

With second dataset, as expected, the result is much better. In general the accuracy varies from 70% to 80%. Final, with the third dataset, the accuracy once again reduced to 65% to 70%.

I went on to train my resnet18 model


With Resnet18, I employed Google Colab and train the model on GPU with 30 epochs



