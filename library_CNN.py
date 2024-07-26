import time
import numpy as np

import torch
import torch.nn as nn

from typing import Tuple, Dict, Any, List
from torchvision import datasets, transforms

import matplotlib

import matplotlib.pyplot as plt

import mlx as ml
import mlx.core as ml # Adding support for Apple Silicon
import mlx.nn.layers as nn_mlx

'''
Notes by Adrian Ramos:
We are using MPS: Metal Performance Shaders to accelerrate operations by using Apple Silicon M3 GPU 
in a nutshell refactor torch.cuda by torch.mps and so. CUDA will only be available at Cluster

Also we use mlx.core as library for hardware accelerated GPU operations. May it also Neral Unit accelerated? Check! Would be amazing
'''

ml.metal.clear_cache()

import time

import torch
import torch.nn as nn

from typing import Tuple, Dict, Any, List
from torchvision import datasets, transforms

start = time.time()

device = torch.device('mps') if ml.metal.is_available() else torch.device('cpu')

cnn_hyperparams = {
'lon',
'lat',
'eastward_model_wind',
'northward_model_wind',
'model_speed',
'model_dir',
'se_model_wind_curl',
'se_model_wind_divergence',
'msl',
'air_temperature',
'q',
'sst',
'sst_dx',
'sst_dy',
'uo',
'vo'
}

'''
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Dataset initializations

mnist_trainset = datasets.MNIST(
    root='data', 
    train=True, 
    download=True,
    transform=transforms
)
mnist_testset = datasets.MNIST(
    root='data', 
    train=False, 
    download=True,
    transform=transforms
)

# Dataloders initialization

train_loader = torch.utils.data.DataLoader(
    dataset=mnist_trainset,
    batch_size=hparams['batch_size'],
    shuffle=True,
    drop_last=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=mnist_testset,
    batch_size=hparams['test_batch_size'],
    shuffle=False,
    drop_last=True,
)

'''

'''
Reminder of the core function of the CNNs: The convolution. In the end
they are dilations, erosions, maybe even masking! We should discuss
about designing a kernel function that ignores NULL data, in the case
this one based on MINST do not work as expected.
conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,3),
                 padding=1,device=mps_device)
'''

NUM_BITS_FLOAT32 = 32

"""
Let's define a class that encapsulates a collection of layers we pass in
for each forwarded layer, it retains the amount of consumed memory for
the returned feature map. It also displays the total amount used after 
all blocks are ran.
"""
class CNNRegressor(nn.Module):

    def __init__(self, layers: nn.Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> Tuple[float, List[int]]:
        tot_mbytes = 0
        spat_res = []
        for layer in self.layers:
            h = layer(x)
            mem_h_bytes = np.cumprod(h.shape)[-1] * NUM_BITS_FLOAT32 // 8
            mem_h_mb = mem_h_bytes / 1e6
            print('-' * 30)
            print(f'New feature map of shape: {h.shape}')
            print(f'Mem usage: {mem_h_mb} MB')
            x = h
            if isinstance(layer, nn.Conv2d):
                # keep track of the current spatial width for conv layers
                spat_res.append(h.shape[-1])
            tot_mbytes += mem_h_mb
        print('=' * 30)
        print('Total used memory: {:.2f} MB'.format(tot_mbytes))
        return tot_mbytes, spat_res
    
    '''
    Perfect conv without compression is 

    conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,1),device=device)

    
    '''

cnn = CNNRegressor(
    nn.ModuleList([
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), device=device),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), device=device),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), device=device),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), device=device),
        nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3,3), device=device),
    ])
)

# Let's work with a realistic 16x512x512 image size
# Also, keep track of time to make forward
beg_t = time.perf_counter()
nopool_mbytes, nopool_res = cnn(torch.randn(1, 1, 512, 512))

# https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html
# Waits for all kernels in all streams on a CUDA device to complete.
torch.cuda.synchronize(device=device)

end_t = time.perf_counter()
nopool_time = end_t - beg_t
print('Total inference time for non-pooled CNN: {:.2f} s'.format(nopool_time))

class ConvBlock(nn.Module):

    def __init__(
            self, 
            num_inp_channels: int, 
            num_out_fmaps: int,
            kernel_size: int, 
            pool_size: int=2) -> None:

        super().__init__()

        # Define the 3 modules. First one apply a simple Conv2d storing the data into the 
        # Apple MPS. The module do all the malloc & sync. 
        self.conv = nn.Conv2d(in_channels=num_inp_channels, out_channels=num_out_fmaps, kernel_size=kernel_size,device=device)
        # The ReLu do nothing. Is a linear transfoermation
        self.relu = nn.ReLU()
        # Now maxpool picks only the maximum value in all the image
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.conv(x)))



model = ConvBlock(
    num_inp_channels=16, # 16 variables per image
    num_out_fmaps=6, # 2 variables to compare with the ground truth: u and v
    kernel_size=5, 
    pool_size=2)

# run forward pass
x = torch.randn(16, 16, 32, 32,device=device) #TODO: MAKE ACCORDING THE MAP DIMENSIONS. MANAGE NULL DATA
y = model(x)

assert y.shape[1] == 6, 'The amount of feature maps is not correct!'
assert y.shape[2] == 14 and y.shape[3] == 14, 'The spatial dimensions are not correct!'
print(f'Input shape: {x.shape}')
print(f'ConvBlock output shape (S2 level in Figure): {y.shape}')

class PseudoLeNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # TODO: Define the zero-padding
        self.pad = torch.nn.ConstantPad2d(2, 0)

        self.conv1 = ConvBlock(num_inp_channels=1, num_out_fmaps=6, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=16, kernel_size=5)

        # TODO: Define the MLP at the deepest layers
        self.mlp = nn.Sequential(nn.Linear(400, 120),
                                 nn.ReLU(),
                                 nn.Linear(120, 84),
                                 nn.ReLU(),
                                 nn.Linear(84,10),
                                 nn.LogSoftmax(dim = 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # Obtain the parameters of the tensor in terms of:
        # 1) batch size
        # 2) number of channels
        # 3) spatial "height"
        # 4) spatial "width"
        bsz, nch, height, width = x.shape
        # TODO: Flatten the feature map with the reshape() operator
        # within each batch sample
        x = x.view(bsz,-1)

        y = self.mlp(x)
        return y

        # Let's forward a toy example emulating the MNIST image size

plenet = PseudoLeNet()
y = plenet(torch.randn(1, 1, 28, 28))
print(f"Output shape: {y.shape}")

def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> float:
    """
    Define the Accuracy metric in the function below by:
      (1) obtain the maximum for each predicted element in the batch to get the
        class (it is the maximum index of the num_classes array per batch sample)
        (look at torch.argmax in the PyTorch documentation)
      (2) compare the predicted class index with the index in its corresponding
        neighbor within label_batch
      (3) sum up the number of affirmative comparisons and return the summation

    Parameters:
    -----------
    predicted_batch: torch.Tensor shape: [BATCH_SIZE, N_CLASSES]
        Batch of predictions
    label_batch: torch.Tensor shape: [BATCH_SIZE, 1]
        Batch of labels / ground truths.
    """
    pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum

def train_epoch(
        train_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        log_interval: int,
        ) -> Tuple[float, float]:

    # Activate the train=True flag inside the model
    network.train()

    train_loss = []
    acc = 0.
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set network gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the network
        output = network(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the network
        optimizer.step()

        # Compute metrics
        acc += compute_accuracy(output, target)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader.dataset)

    return np.mean(train_loss), avg_acc
    

@torch.no_grad() # decorator: avoid computing gradients
def test_epoch(
        test_loader: torch.utils.data.DataLoader,
        network: torch.nn.Module,
        ) -> Tuple[float, float]:

    # Dectivate the train=True flag inside the model
    network.eval()
    
    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = network(data)

        # Apply the loss criterion and accumulate the loss
        test_loss.append(criterion(output, target).item())

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    # Average accuracy across all correct predictions batches now
    test_acc = 100. * acc / len(test_loader.dataset)
    test_loss = np.mean(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc

    train_losses = []
test_losses = []
train_accs = []
test_accs = []
network = PseudoLeNet()
network.to(device)

optimizer = torch.optim.RMSprop(network.parameters(), lr=hparams['learning_rate'])
criterion = nn.NLLLoss(reduction='mean')

for epoch in range(hparams['num_epochs']):

    # Compute & save the average training loss for the current epoch
    train_loss, train_acc = train_epoch(train_loader, network, optimizer, criterion, hparams["log_interval"])
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # TODO: Compute & save the average test loss & accuracy for the current epoch
    # HELP: Review the functions previously defined to implement the train/test epochs
    test_loss, test_accuracy = test_epoch(test_loader, network)

    test_losses.append(test_loss)
    test_accs.append(test_accuracy)

    # Plot the plots of the learning curves
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.subplot(2,1,2)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy [%]')
plt.plot(train_accs, label='train')
plt.plot(test_accs, label='test')