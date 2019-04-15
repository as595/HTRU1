
# HTRU1

**This site is still underconstruction - data are not yet ready for use**

The [HTRU1 Batched Dataset](https://raw.githubusercontent.com/as595/HTRU1/master/htru1-batches-py.tar.gz) is a subset of the HTRU Medlat Training Data, a collection of labeled pulsar candidates from the intermediate galactic latitude part of the HTRU survey. HTRU1 was originally assembled to train the SPINN pulsar classifier. If you use this dataset please cite:

*SPINN: a straightforward machine learning solution to the pulsar candidate selection problem*
V. Morello, E.D. Barr, M. Bailes, C.M. Flynn, E.F. Keane and W. van Straten, 2014, Monthly Notices of the Royal Astronomical Society, vol. 443, pp. 1651-1662 [arXiv:1406:3627](http://arxiv.org/abs/1406.3627)

*The High Time Resolution Universe Pulsar Survey - I. System Configuration and Initial Discoveries* 
M. J. Keith et al., 2010, Monthly Notices of the Royal Astronomical Society, vol. 409, pp. 619-627 [arXiv:1006.5744](https://arxiv.org/abs/1006.5744)

The full HTRU dataset is available [here](https://archive.ics.uci.edu/ml/datasets/HTRU2#). 

## The HTRU1 Batched Dataset

The [HTRU1 Batched Dataset](https://raw.githubusercontent.com/as595/HTRU1/master/htru1-batches-py.tar.gz) consists of 60000 32x32 images in 2 classes: pulsar & non-pulsar. Each image has 3 channels (equivalent to RGB), but the channels contain different information:

 * **Channel 0:** *Period Correction - Dispersion Measure surface*
 * **Channel 1:** *Phase - Sub-band surface*
 * **Channel 2:** *Phase - Sub-integration surface*

There are 50000 training images and 10000 test images. The [HTRU1 Batched Dataset](https://raw.githubusercontent.com/as595/HTRU1/master/htru1-batches-py.tar.gz) is inspired by the [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html).

The dataset is divided into five training batches and one test batch. Each batch contains 10000 images. These are in random order, but each batch contains the same balance of pulsar and non-pulsar images. Between them, the six batches contain 1194 true pulsars and 58806 non-pulsars. 

This is an *imbalanced dataset*.

Pulsar: ![pulsar1](/media/pulsar_0000.jpg) ![pulsar2](/media/pulsar_0001.jpg) ![pulsar3](/media/pulsar_0002.jpg) ![pulsar4](/media/pulsar_0003.jpg) ![pulsar5](/media/pulsar_0004.jpg) ![pulsar6](/media/pulsar_0005.jpg) ![pulsar7](/media/pulsar_0006.jpg) ![pulsar8](/media/pulsar_0007.jpg) ![pulsar9](/media/pulsar_0008.jpg) ![pulsar10](/media/pulsar_0009.jpg) 

Non-pulsar: ![cand1](/media/cand_000002.jpg) ![cand2](/media/cand_000003.jpg) ![cand3](/media/cand_000014.jpg) ![cand4](/media/cand_000015.jpg) ![cand5](/media/cand_000018.jpg) ![cand6](/media/cand_000019.jpg) ![cand7](/media/cand_000022.jpg) ![cand8](/media/cand_000023.jpg) ![cand9](/media/cand_000034.jpg) ![cand10](/media/cand_000035.jpg) 


## Using the Dataset in PyTorch

The [htru1.py](https://raw.githubusercontent.com/as595/HTRU1/master/htru1.py) file contains an instance of the [torchvision Dataset()](https://pytorch.org/docs/stable/torchvision/datasets.html) for the HTRU1 Batched Dataset. 

To use it with PyTorch in Python, first import the torchvision datasets and transforms libraries:

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

### Using Individual Channels in PyTorch

If you want to use only one of the "channels" in the HTRU1 Batched Dataset, you can extract it using the torchvision generic transform [transforms.Lambda](https://pytorch.org/docs/stable/torchvision/transforms.html#generic-transforms). 

This function extracts a specific channel ("c") and writes the image of that channel out as a greyscale PIL Image:

```python
def select_channel(x,c):
    
    from PIL import Image
    
    np_img = np.array(x, dtype=np.uint8)
    ch_img = np_img[:,:,c]
    img = Image.fromarray(ch_img, 'L')
    
    return img
 ```
 
 You can add it to your pytorch transforms like this:
 
 ```python
 transform = transforms.Compose(
    [transforms.Lambda(lambda x: select_channel(x,0)),
     transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])
 ```
 
### Jupyter Notebooks

An example of classification using the HTRU1 class in PyTorch is provided as a Jupyter notebook [treating the dataset as an RGB image](https://github.com/as595/HTRU1/blob/master/htru1_tutorial.ipynb) and also [extracting an individual channel as a greyscale image](https://github.com/as595/HTRU1/blob/master/htru1_tutorial_channel.ipynb).

[![HitCount](http://hits.dwyl.io/as595/HTRU1.svg)](http://hits.dwyl.io/as595/HTRU1)

