# Homework 1
The project was made by Massimiliano Roccamena, Abednego Wamuhindo, Muhammad Irfan Mas'udi.

## Overview

The overall project is splitted in multiple stages, performed in parallel by each member, grouped in a list ordered by overall network complexity and expected performaces

Each stage is driven by some instance of deep learning architecture, focusing on searching for a good deep learning encoding of labeled images

1. Basic CNN
2. VGG
3. Inception & ResNet
4. InceptionResNet & NASNetLarge

In each stage we also performed some hyperoptimization and comparison with previous architectures, as well as exploring different classifiers on top of it

In particular, we started by building our small CNN architectures from scratch focusing on the repetition of convolution and pooling layers, then exploring different patterns

Then we started using transfer learning from stage 2 with VGG architecture

In the last stage, we've taken the best deep encoder found and tried more general type of classifiers on top of it, and we also visualized best deep encoder's transformation on the training set

5. Hybrid system

## Preprocess and training
During the whole process learning rate was almost always equal to 1e-4, in general it was the right amount needed

Also fully connected classifier's weights were generally initialized with Xavier initialization; since tranfer learning is used, transferred weights were almost always initialized with image-net competitions weights

Also leaky relu activation is tested in some configurations (expecially deep ones) and also l2 regularization and dropout on classifier are tested from the start to counter some overfitting

This is a small history of our learning approach of the network, in temporal order

| Stage | Description |
| :----: | :----: |
| Stage 1 | We started focusing on early stopped learning, but also tried some longer convergence train or cross validation; images size were fixed at 255x255 |
| Stage 2 | As soon as we used transfer learning we've tried also to hyperoptimize parameters such as fine tuning and encoder parameters; with more experience we find early stopping pretty much very effective; also we started exploring each specific encoder architecture preprocessing |
| Stage 3 | We noticed using avg pooling on deep learned features in combination with high dropout was very effective; also augmentation was found very effective almost always |
| Stage 4 | We increased images size to 299x299, but we also tried higher resolution images up to 331x331; we noticed fine tuned models were in general more performing than the others |
| Stage 5 | We selected various heterogeneous models on best deep encoded data using crossvalidation and small hyperoptimizations; in the meanwhile we were hyperoptimizing previous architectures best models

Here is reported a learning comparison visualization between some Inception (M1) and InceptionResNet (M2) based models

| ![](./imgs/cmp_train.PNG) | 
|:--:| 
| *training loss between M1 (orange) and M2 (red), with smoothing 0.2* |

| ![](./imgs/cmp_val.PNG) | 
|:--:| 
| *validation accuracy between M1 (blue) and M2 (azure), with smoothing 0.5* |

## Testing

Here is listed a summary of categorical accuracy values measured on test set

| Architecture | Range |
| :----: | :----: |
| Basic CNN | 60%, 70% |
| VGG | 75%, 85% |
| Inception & ResNet | 85%, 90% |
| InceptionResNet & NASNet | 90%, 93% |
| Hybrid model | 91%, 94% |

## Optimal solution

Our optimal neural network was an InceptionResNet encoder (fine tuned from 15th layer) with avg pooling at the output, then 0.8 dropout and 3 softmax; it was early stopped on validation loss and trained on augmented data with validation split 17.5%

We focused on making the fully connected classifier a simple transformation, forcing it to co-adapt on encoder output stacking high dropout on top of it, in order to make the encoder learn high quality semantics computations

Our optimal model was obtained by replacing this classifier on top of InceptionResNet encoder of previous network with an optimized SVM (also Random Forest), gaining some accuracy on test set

| ![](./imgs/PCA.PNG) | 
|:--:| 
| *2D PCA visualization of optimal deep encoder transformation* |

| ![](./imgs/t-SNE.PNG) | 
|:--:| 
| *2D t-SNE visualization of optimal deep encoder transformation* |

| ![](./imgs/missclassified.PNG) | 
|:--:| 
| *2D informative projection of missclassified instances of optimal SVM classifier* |

### Important

During final stages of the whole process we acknowledged a bug which make the model not reproducible: even if we seeded tensorflow and numpy from the start, as well as images generators, we didn't observed same learning functions on repetead instances of same model (but same images dataset), even with all (maybe needed) seeded version of nondeterministic functionalities invoked (for example seeded droput layer)

We invite you to check best_solution.ipynb for original best model training notebook used (may be different dirs naming ex data1,log1), and also to use deep encoder checkpoint file with optimal hybrid classifier pickle file available at following link:

(WIP)

## Final considerations
We have understood how important is the underlying deep learning architecture for the overall quality of the problem solution
- We have found Inception architectures having high performance gap with previous architectures, mainly due to the Inception module giving better scaling semantics to images.

Also we can highlight that data augmentation and fine tuning can greatly impact the overall comprehension of the network about the problem

We also experienced how making an ensemble of deep learned features using high dropout leads to good learning of the problem by the network