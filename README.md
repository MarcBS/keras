# MarcBS/keras Multimodal Learning fork

[![Build Status](https://travis-ci.org/MarcBS/keras.svg?branch=master)](https://travis-ci.org/MarcBS/keras)

This fork of Keras offers the following contributions:

- Caffe to Keras conversion module
- Layer-specific learning rates
- New layers for multimodal data


Contact email: marc.bolanos@ub.edu

GitHub page: https://github.com/MarcBS


MarcBS/keras has been tested with: __Python 2.7__ and __Python 3.6__  and with the __Theano__ and __Tensorflow__ backends.

## Caffe to Keras conversion module

This module allows to convert Caffe models to Keras for their later training or test use.
See [this README](keras/caffe/README.md) for further information.

**Please, be aware that this feature is not regularly maintained**. Thus, some layers or parameter definitions introduced in newer versions of either Keras or Caffe might not be compatible with the converter.

**For this reason, any pull requests with updated versions of the caffe2keras converter are highly welcome!**

## Layer-specific learning rates

This functionality allows to add learning rates multipliers to each of the learnable layers in the networks. During training they will
be multiplied by the global learning rate for modifying the weight of the error on each layer independently. Here is a simple example of usage:

```
x = Dense(100, W_learning_rate_multiplier=10.0, b_learning_rate_multiplier=10.0)  (x)
```

## New layers for sequence-to-sequence learning and multimodal data

#### [Recurrent layers](https://github.com/MarcBS/keras/blob/master/keras/layers/recurrent_advanced.py)
LSTM layers:
- [LSTMCond](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L2421): LSTM conditioned to the previously generated word (additional input with previous word).
- [AttLSTM](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L3367): LSTM with Attention mechanism.
- [AttLSTMCond](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L3783): LSTM with Attention mechanism and conditioned to previously generated word.
- [AttConditionalLSTMCond](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L4370): ConditionalLSTM [similar to Nematus](https://arxiv.org/abs/1703.04357) with Attention mechanism and conditioned to previously generated word.
- [AttLSTMCond2Inputs](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L5540): LSTM with double Attention mechanism (one for each input) and conditioned to previously generated word.
- [AttLSTMCond3Inputs](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L6192): LSTM with triple Attention mechanism (one for each input) and conditioned to previously generated word.
- others

And their corresponding GRU version:

- [GRUCond](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L628): GRU conditioned to the previously generated word (additional input with previous word).
- [AttGRUCond](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L1027): GRU with Attention mechanism and conditioned to previously generated word.
- [AttConditionalGRUCond](https://github.com/MarcBS/keras/blob/75ab7cc25db43b1f6c44496a77414a4c9030c58a/keras/layers/recurrent.py#L1564): ConditionalGRU [as in Nematus](https://arxiv.org/abs/1703.04357) with Attention 

#### Convolutional layers
- [ClassActivationMapping](https://github.com/MarcBS/keras/blob/4e6a8ec8a55bd0d5d091a44b058a797d3d934ce0/keras/layers/convolutional.py#L23): Class Activation Mapping computation used in [GAP networks](http://arxiv.org/pdf/1512.04150.pdf).
- [CompactBilinearPooling](https://github.com/MarcBS/keras/blob/4e6a8ec8a55bd0d5d091a44b058a797d3d934ce0/keras/layers/convolutional.py#L1395): compact version of bilinear pooling for [merging multimodal data](http://arxiv.org/pdf/1606.01847v2.pdf).


#### Attentional layers
- [MultiHeadAttention](https://github.com/MarcBS/keras/blob/f7caf432dc51d90ec3bbd8b141b789bc90179292/keras/layers/attention.py#L14): Multi-head attention layer. Multi-Head Attention consists of h attention layers running in parallel. Base of the [Transformer model](https://arxiv.org/abs/1706.03762).

## Projects

You can see more practical examples in projects which use this library:

[ABiViRNet for Video Description](https://github.com/lvapeab/ABiViRNet)

[Egocentric Video Description based on Temporally-Linked Sequences](https://github.com/MarcBS/TMA)

[NMT-Keras: Neural Machine Translation](https://github.com/lvapeab/nmt-keras).


## Installation

In order to install the library you just have to follow these steps:

1) Clone this repository:
```
git clone https://github.com/MarcBS/keras.git
```
2) Include the repository path into your PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/path/to/keras
```

## Keras

For additional information on the Deep Learning library, visit the official web page www.keras.io or the GitHub repository https://github.com/keras-team/keras.
