# Homework 2

The project was made by Massimiliano Roccamena, Abednego Wamuhindo, Muhammad Irfan Mas'udi.

# Overview

The overall project is splitted in multiple stages, performed in parallel by each member by maximimizing the overall performances of previous stages.

Early stages focus on implementing architectures for segmentation

- Basic VGG
- U-NET

Latest stages are focus on optimizing data preprocessing

- Data Augmentation
- Tiling

# Analyzing Data

We started by analyzing strategies to tackle the problem: first of all we observed high resolution images, than we noticed that data was more structured than usual, in the sense that the collection of images was performed grouping by 2 different crops and 4 different teams, and so we could then be able to observe that teams data have unbalanced sizes and that one of the team obtained images from different perspective.

So we discussed on how to exploit also this knowledge of the problem: at first we found that we could actually split one model in one combination of multiple models capturing different point of view of problem's semantic, for example by optimizing one model for crop or reserve one model for the team with different capturing perspective.

Then we thought that models (we planned to use) themselves should be able to distinguish between above semantics, that this startegy could be much more time consuming and also that the models can exploit some data preprocessing which should (with other things) greatly help the network on these concerns.

# Summary

We first processed train set shuffling data by team and crop, then we identified 20% of it as a validation set.

We started training some models to test different architectures on our current configurations, by early stopping on validation loss and (manually) optimizing on IoU metric; we always used a learning rate of 1e-4.

We implemented a first solution using a basic segmentation deep learning architecture made by a VGG encoder and a decoder realized by the usual convolution + upsampling architecture.

Next we decided to implement one very good segmentation architecture, so we selected U-NET architecture to handle the job from now on.

After already getting very good results, we decided to switch our attention to optimizing the data, as well as to adapt our model overall behavior.

First of all we introduced some augmentation to original data and measured differences for both models, then we exploited the fact that images were high resolutives to tile (augmented as well) this big images in order to boost data size and provide brand new prediction semantics to the model, which now use network predictive power on multiple tiles of the input for combining them in the overall big segmentation.