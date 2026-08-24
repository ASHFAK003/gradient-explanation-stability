"""
Training script adapted from:
https://github.com/ChawDoe/LeNet5-MNIST-PyTorch

Licence:
MIT Licence, Copyright (c) 2022 ChawDoe.
A copy of the licence is included in:
projectcodes/MNIST/licences/ChawDoe_LeNet5_MNIST_MIT_LICENSE.txt

Minimal adaptations:
- import changed from model.py to mnist_model.py
- MNIST download=True added
- model saving changed to state_dict for later loading in the saliency experiment

Update for the smooth-network experiment:
- The architecture in mnist_model.py now uses Softplus activations and
  average-pooling (instead of ReLU and max-pooling). This script does not
  need to change for that: it imports Model from mnist_model.py and therefore
  trains whatever architecture is defined there.
- The only change here is the OUTPUT CHECKPOINT NAME. The retrained smooth
  model is saved to a NEW file so the original ReLU checkpoint
  (models/mnist_lenet5_state_dict.pt) is preserved for the
  ReLU-vs-Softplus comparison reported in the dissertation.
"""

from mnist_model import Model
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# New checkpoint name for the retrained Softplus + AvgPool model.
# (Original ReLU model lived at models/mnist_lenet5_state_dict.pt and is kept.)
MODEL_SAVE_PATH = 'models/mnist_lenet5_softplus_state_dict.pt'

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    prev_acc = 0
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()

        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('epoch {}  accuracy: {:.3f}'.format(current_epoch, acc), flush=True)
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc
    print("Model finished training. Saved to:", MODEL_SAVE_PATH)
