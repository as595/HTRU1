# HTRU1

The HRTU1 Batched Dataset is a subset of the HTRU Medlat Training Data, a collection of labeled pulsar candidates from the intermediate galactic latitude part of the HTRU survey. It was assembled to train the SPINN pulsar classifier described in:

*SPINN: a straightforward machine learning solution to the pulsar candidate selection problem*
V. Morello, E.D. Barr, M. Bailes, C.M. Flynn, E.F. Keane and W. van Straten [arXiv:1406:3627](http://arxiv.org/abs/1406.3627)


The full HTRU dataset is available [here](https://archive.ics.uci.edu/ml/datasets/HTRU2#). If you use these data please cite:

*The High Time Resolution Universe Pulsar Survey - I. System Configuration and Initial Discoveries* 
M. J. Keith et al., 2010, Monthly Notices of the Royal Astronomical Society, vol. 409, pp. 619-627. DOI: 10.1111/j.1365-2966.2010.17325.x 

## The HTRU1 Batched Dataset

The HTRU1-BD consists of 60000 32x32 colour images in 2 classes: pulsar & non-pulsar. There are 50000 training images and 10000 test images. The HTRU1-BD is inspired by the [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html).

The dataset is divided into five training batches and one test batch. Each batch contains 10000 images. These are in random order, but each batch contains the same balance of pulsar and non-pulsar images. Between them, the six batches contain 1196 true pulsars and 58804 non-pulsars. 

This is an *imbalanced dataset*.

### Get the data:

<p align="center">[Download the Dataset](https://raw.githubusercontent.com/as595/HTRU1/master/htru1-batches-py.tar.gz)</p>

| Pulsar: | <p align="left"><img width=10% src="https://github.com/as595/HTRU1/blob/master/media/pulsar_0000.jpg"></p> | <p align="left"><img width=10% src="https://github.com/as595/HTRU1/blob/master/media/pulsar_0001.jpg"></p> |

