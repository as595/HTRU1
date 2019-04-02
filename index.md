# HTRU1

The [HRTU1 Batched Dataset](https://raw.githubusercontent.com/as595/HTRU1/master/htru1-batches-py.tar.gz) is a subset of the HTRU Medlat Training Data, a collection of labeled pulsar candidates from the intermediate galactic latitude part of the HTRU survey. It was assembled to train the SPINN pulsar classifier described in:

*SPINN: a straightforward machine learning solution to the pulsar candidate selection problem*
V. Morello, E.D. Barr, M. Bailes, C.M. Flynn, E.F. Keane and W. van Straten [arXiv:1406:3627](http://arxiv.org/abs/1406.3627)


The full HTRU dataset is available [here](https://archive.ics.uci.edu/ml/datasets/HTRU2#). If you use these data please cite:

*The High Time Resolution Universe Pulsar Survey - I. System Configuration and Initial Discoveries* 
M. J. Keith et al., 2010, Monthly Notices of the Royal Astronomical Society, vol. 409, pp. 619-627. DOI: 10.1111/j.1365-2966.2010.17325.x 

## The HTRU1 Batched Dataset

The HTRU1-BD consists of 60000 32x32 colour images in 2 classes: pulsar & non-pulsar. There are 50000 training images and 10000 test images. The HTRU1-BD is inspired by the [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html).

The dataset is divided into five training batches and one test batch. Each batch contains 10000 images. These are in random order, but each batch contains the same balance of pulsar and non-pulsar images. Between them, the six batches contain 1196 true pulsars and 58804 non-pulsars. 

This is an *imbalanced dataset*.

Pulsar: ![pulsar1](https://github.com/as595/HTRU1/blob/master/media/pulsar_0000.jpg) ![pulsar2](https://github.com/as595/HTRU1/blob/master/media/pulsar_0001.jpg) ![pulsar3](https://github.com/as595/HTRU1/blob/master/media/pulsar_0002.jpg) ![pulsar4](https://github.com/as595/HTRU1/blob/master/media/pulsar_0003.jpg) ![pulsar5](https://github.com/as595/HTRU1/blob/master/media/pulsar_0004.jpg) ![pulsar6](https://github.com/as595/HTRU1/blob/master/media/pulsar_0005.jpg) ![pulsar7](https://github.com/as595/HTRU1/blob/master/media/pulsar_0006.jpg) ![pulsar8](https://github.com/as595/HTRU1/blob/master/media/pulsar_0007.jpg) ![pulsar9](https://github.com/as595/HTRU1/blob/master/media/pulsar_0008.jpg) ![pulsar10](https://github.com/as595/HTRU1/blob/master/media/pulsar_0009.jpg) 

Non-pulsar: ![cand1](https://github.com/as595/HTRU1/blob/master/media/cand_000002.jpg) ![cand2](https://github.com/as595/HTRU1/blob/master/media/cand_000003.jpg) ![cand3](https://github.com/as595/HTRU1/blob/master/media/cand_000014.jpg) ![cand4](https://github.com/as595/HTRU1/blob/master/media/cand_000015.jpg) ![cand5](https://github.com/as595/HTRU1/blob/master/media/cand_000018.jpg) ![cand6](https://github.com/as595/HTRU1/blob/master/media/cand_000019.jpg) ![cand7](https://github.com/as595/HTRU1/blob/master/media/cand_000022.jpg) ![cand8](https://github.com/as595/HTRU1/blob/master/media/cand_000023.jpg) ![cand9](https://github.com/as595/HTRU1/blob/master/media/cand_000034.jpg) ![cand10](https://github.com/as595/HTRU1/blob/master/media/cand_000035.jpg) 


## Using the Dataset in PyTorch

The [htru1.py](https://raw.githubusercontent.com/as595/HTRU1/master/htru1.py) file contains an instance of the [torchvision Dataset()](https://pytorch.org/docs/stable/torchvision/datasets.html) for the HTRU1 Batched Dataset. To use it with PyTorch in Python, first import the torchvision datasets and transforms libraries:

```python
from torchvision import datasets
import torchvision.transforms as transforms
```

Then import the HTRU1 class:

```python
from htru1 import HTRU1
```

Define the transform:

```python
# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
 ```

Read the HTRU1 dataset:

```python
# choose the training and test datasets
train_data = HTRU1('data', train=True, download=True, transform=transform)
test_data = HTRU1('data', train=False, download=True, transform=transform)
```

An example of classification using the HTRU1 class is provided in [this jupyter notebook].
