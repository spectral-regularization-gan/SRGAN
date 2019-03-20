# SRGAN

## Prerequisites

--chainer==3.3.0

--tensorflow-gpu==1.2.0

--numpy==1.11.1

--cupy==2.0.0

--scipy==0.19.0

--pillow==4.3.0

--pyyaml==3.12

--h5py==2.7.1

## Usage

edit ./configs/sn_cifar10_unconditional.yml to set your optimization settings, including batchsize, channel size in the discriminator.


To train a model:

    $ python train.py
    
    
 ## Pretrained models
 
pretrained models can be found in ./results.
 
 
    
