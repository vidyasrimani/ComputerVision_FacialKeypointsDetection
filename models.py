## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torch.autograd import Variable#added import

'''
Taking reference from AlexNet
-AlexNet famously won the 2012 ImageNet LSVRC-2012 competition by a large margin (15.3% VS 26.2% (second place) error rates). 

Implementation
-Use Relu instead of Tanh to add non-linearity. It accelerates the speed by 6 times at the same accuracy.
-Use dropout instead of regularisation to deal with overfitting. However the training time is doubled with the dropout rate of 0.5.
-Overlap pooling to reduce the size of network. It reduces the top-1 and top-5 error rates by 0.4% and 0.3%, repectively

Experiment
-Try different dropout probabilities
-try different activation functions

'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        #Note to self:
            #NN layers inclue :
                #Convolutional layers
                #Maxpooling layers
                #Fully-connected (linear) layers
                #Also maybe include drop out layer(to avoid over fitting) 
  
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        #Note to self:Syntax--> self.conv2 = nn.Conv2d(in_channels=__, out_channels=__, kernel_size=__, stride=__, padding=__)
        #Note to self : stride and padding need not be included but is apparently a good practice to include
        self.conv1 = nn.Conv2d(1,10,3)#kernel of size 3
        self.conv2 = nn.Conv2d(10,32,3)
        self.conv3 = nn.Conv2d(32,64,3)
        
        # maxpool that uses a square window of kernel_size=2, stride=0
        # maxpool layer
        # pool with kernel_size=2, stride=0--
        #Note to self:Syntax-->self.pool = nn.MaxPool2d(kernel_size=__, stride=__, padding=__)
        self.pool = nn.MaxPool2d(2, 2)

        
        #Drop out layers
        #Probability values randomly chosen. Any ethod to pick optimal values for p in dropout layers?Note to self:--> check
        self.dropout=nn.Dropout(p=0.3)


        '''
        self.fc1=nn.Linear(6400,3200)       
        self.fc2=nn.Linear(3200,1600)
        self.fc3=nn.Linear(1600,136)
        
        
        Note to self: RuntimeError: DataLoader worker (pid 323) is killed by signal: Bus error
        '''
        
        #fully connected layers
        #Note to self: check dimentions
        self.fc1 = nn.Linear(in_features=43264, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=136)

        
        '''
        Note to self: Error: RuntimeError: size mismatch, m1: [10 x 43264], m2: [6400 x 1000] at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/TH/generic/THTensorMath.c:2033
        WHere is m1: [10 x 43264] coming from ? Need to check
        
        self.fc1=nn.Linear(6400,1000)       
        self.fc2=nn.Linear(1000,500)
        self.fc3=nn.Linear(500,136)
     


        
        #let's try batch normalization
        
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        '''
        
        
        #Customize weights

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = x.view(x.size(0),-1)#-->like reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.dropout(x)
        
        return x
