"""
MNIST CNN model adapted from:
https://github.com/ChawDoe/LeNet5-MNIST-PyTorch

Licence:
MIT Licence, Copyright (c) 2022 ChawDoe.
A copy of the licence is included in:
projectcodes/MNIST/licences/ChawDoe_LeNet5_MNIST_MIT_LICENSE.txt

Original source note:
The source repository implements a LeNet-5-style CNN for MNIST using
convolutional layers, ReLU activations, max pooling, and fully connected
layers.

Adaptation made in this project:
The original model applied a ReLU activation after the final linear layer:

    y = self.fc3(y)
    y = self.relu5(y)
    return y

For this project, that final ReLU was removed so that the model returns
raw class scores/logits:

    y = self.fc3(y)
    return y

Justification:
1. CrossEntropyLoss expects raw logits as input.
2. The dissertation methodology defines the class-gradient explanation as
   g_c(x) = ∇_x S_c(x), where S_c(x) is the class score.
3. The raw output of the final linear layer fc3 is the appropriate class
   score S_c(x).
4. A final ReLU would clip negative class scores to zero, making the output
   less suitable for class-gradient saliency analysis.
5. As further updates we replace any ReLu with Softplus activation function because ReLU second order derivative can vanish, which 
can be avoided by Softplus function...and second order derivatives are important for calculating differences in explanations.
"""

from torch.nn import Module
from torch import nn


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.softplus1 = nn.Softplus()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.softplus2 = nn.Softplus()
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.softplus3 = nn.Softplus()
        self.fc2 = nn.Linear(120, 84)
        self.softplus4 = nn.Softplus()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = self.softplus1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.softplus2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.softplus3(y)
        y = self.fc2(y)
        y = self.softplus4(y)
        y = self.fc3(y)
        return y
