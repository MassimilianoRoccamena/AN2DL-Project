# Homework 2

The project was made by Massimiliano Roccamena, Abednego Wamuhindo, Muhammad Irfan Mas'udi.

# Overview

The overall project is splitted in multiple stages, performed in parallel by each member by maximimizing the overall performances of previous stages.

Early stages focus on implementing basic architectures for the problem

- VGG + LSTM
- ResNet + Bidirectional LSTM

Middle stages focus on a (stacked) visual attention mechanism (inspired by https://arxiv.org/pdf/1511.02274.pdf, revisited)

- VGG + LSTM + Visual Attention
- ResNet + Bidirectional LSTM + SAN

Latest stages focus on better language modelling using Transformer

- ResNet + Transformer
- ResNet + Transformer + SAN

# Summary

The basic difference between previous project was that VQA has 2 different type of data (text + image), so we exploited 2 mechanism to merge language and vision network: concatenation and vision-by-language attention.

We first processed train set by shuffling data and preprocessing images for transfered deep encoding (no data augmentation, we have lot of images and don't want to loose some spatial info maybe asked by the questions).

We started creating some models to test different architectures, by early stopping on validation loss and (manually) optimizing on validation loss metric; we first kept a learning rate of 1e-4, and then we also implemented an adaptive learning rate to fully exploit final epoches (by exponential decay from 1/2^epoch to 4/5^epoch).

We implemented a first solution by encoding images using pretrained fine tuned VGG/ResNet (with GAP) and encoding questions using Embedding + LSTM (1 LSTM layer, then tried to stack them up to 3, then Bidirectional + forward LSTM); to classify the model we merged the 2 encoding by Concatenation. In general Bidirectional + forward LSTM was the best.

Next we noticed that maybe some spatial information of the question can be lost by GAP, so we tried a merging strategy using an attention mechanism: we removed GAP (so we have high level spatial regions of the image) and we query visual values of the regions by encoded question key; we also tried stacking some of this attentions (as in the original paper) but we found less performances of just concatenating GAP; in general this visual attention was found less performant, we concluded because questions were not very difficult to be understood in the image space, no reasoning like "what is sitting in the basket in the bycicle?" but just something like "who looks happier?" or "what is the man doing?".

Finally we trained our final architectures using ResNet + Concatenation (using previous conclusions) for image encoding and by using TokeAndPositionEmbedding + Transformer (first 1 Transformer, then stacking up to 6, then by stacking 3 + forward LSTM) for question encoding.

# Optimal Model

Our optimal model was based on:

- image (size 299x299) encoding using fine tuned (from layer 130) ResNet with Global Averaging Pooling and 0.1 dropout.
- question encoding using 3 stacked Transformer encoder with 40 dimensional embeddings and 8 attention heads each.
- softmax classifier on 800 dimensional linear transformation of concatenated encodings.

# Notebooks

- homework.ipynb: main training, testing and exporting notebook

# Links

Optimal model's checkpoints

[LINK]
