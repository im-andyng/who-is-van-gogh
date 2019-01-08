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
https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/5870145/vgdb_2016.zip
```

Because of this, I went on to create a new dataset which uses the same Van Gogh's paintings from the first one, but replace all non-Van Gogh paintings by ones from a single artist. The artist that I chose is Aja Kusick (https://www.instagram.com/sagittariusgallery/?hl=en). Her paintings adopt Van Gogh's style and mix it with modern objects like characters from cartoons or movies (like Starwar). With this dataset, I predict that the accuracy will be higher.

How about the third one ? Well, I'm a curiosity one so I would like to see what if there's less variance in the non-Van Gogh dataset, will the model perform better compared to its performance over the first one. This time I chose Picasso as I figured out his paintings belong to two styles, according to me. 

The third and second dataset is available in this Github repo in RAR archieve.

## Spliting data for training and testing
All three datasets are split into training and testing at 80/20 ratio.

## Augumentation
I apply only one method of augumentation which is Random Resize Crop

## Normalization
[0.5, 0.5, 0.5] is used for both mean and std

## Loading dataset
The image dataset is loaded using dataloader with bath_size=32

```python
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

data_dir = "aja"

train_data = datasets.ImageFolder(data_dir+"/train", transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

valid_data = datasets.ImageFolder(data_dir+"/test", transform=valid_transforms)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
```

# The models
I built two models to solve the challenge. One is a self-built convolutional neural network and the other one reused a pretrained neural network: Resnet18. I train both models on three datasets to compare the performance.

## Self-built neural network
This model is built with three convolutional layers and two fully connected layers. Each convolutional layers has a padding of 1 and stride of 1. Kernel size is 3 for all three convolutional layers. After each layer, I apply a ReLU activation function and insert a maxpooling layer (4x4) to reduce the dimension size and increase the depth. Through each layers, the input depth increases from 3 (RGB image) to 16, 32 and then 64.

As the input is 512, the output volume of these convolutional neural networks is 64*8*8 (512/4/4/4 = 8). This output serves as input for the next two fully connected layers, which will output class score for vg (Van Gogh) or nvg (non-Van Gogh) paintings. 

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(4, 4)
        # fully-connected layer
        self.fc1 = nn.Linear(64*8*8, 2048)
        self.fc2 = nn.Linear(2048,2)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64*8*8)
        
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
```

## Resnet18
A pre-trained Resnet18 model is downloaded using torchvision. I freeze all the convolutional layers and adjust the fully connected layer to fit with my output.

```python
model = models.resnet18(pretrained=True)

for param in model.parameters():
  param.requires_grad = False
  
model.fc = nn.Linear(512, 2, bias=True)
```

# Loss function and optimizer
With self-built neural network, I use Cross Entropy Loss function (nn.CrossEntropyLoss()) and SGD optimizer. For resnet18, I also use Cross Entropy Loss but with Adam optimizer.

# Training the network
In this Github repositories, I provide the code for both self-built and resnet model. As the self-built one is pretty simple, I don't provide the checkpoint file as it takes only 10 to 15 minutes to train the model from scratch. With Google Colab one, I provide both the code with output and the checkpoint file to quickly reproduce the result.

<b>Note: </b>
- If you use Aja Kusnick or Picasso dataset to train the model, remember to copy vg folder in test and train folder of vgdb_2016 dataset to test and train folder in these dataset. 
- Remember to change the folder name/path to the dataset
- If you load the checkpoint, make sure it matches the file name.

# Result and key learning
I train the self-built neural network with 10 epochs for each dataset. The trainning is done on CPU as my GPU does not have enough memory to run. 

With first dataset, the accuracy varies from 65% to 70% on the test set. The model performs pretty well on identifying Van Gogh's paintings with 90% accuracy but struggles to spot non-Van Gogh ones. As dicussed above, I guess the problem stays with variance in styles of the non-Van Gogh paintings. Of course there could also be issues with how I setup the deep learning model but at this stage I cannot confirm the later theory. 

With second dataset, as expected, the result is much better. In general the accuracy varies from 70% to 80%. Final, with the third dataset, the accuracy once again reduced to 65% to 70%.

I went on to train my resnet18 model using GPU (on Google Colab) with 30 epochs. Resnet18 model performed much better than my model with 
86% accuracy on my first dataset. With more convolutional layers being stacked upon each other. Resnet18 seems to dealt very well with variance of style in non-Van Gogh dataset. I may infer from this that the model will beat 90% or even more accuracy on the other two datasets. And it's true. With second dataset resnet18 model archive 100% accuracy, this is a little bit unreal and there might be some bias. With the third one it achieved 96% accuracy.

From this experiment, I learned that both data quality and model quality affects the accuracy of the prediction. Second, if a state-of-art model suits the problem you are trying to solve, then use it will save lots of time and unecessary effort.

# Questions:
This final section will list out questions that I haven't got time to see if I can answer it
- If I apply the model that was trained on vgdb_2016 dataset, how will it perform on the other dataset ?
- Besides resnet18, what other models can be applied to improve the accuracy ?
- Because this seems like a binary classification, will using different loss function improve the performance ?


