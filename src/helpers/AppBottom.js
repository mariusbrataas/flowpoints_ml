import React, { Component } from 'react';

import Fab from '@material-ui/core/Fab';
import IconButton from '@material-ui/core/IconButton';
import AddIcon from '@material-ui/icons/Add';
import CodeIcon from '@material-ui/icons/Code';
import SaveIcon from '@material-ui/icons/Save';
import ShareIcon from '@material-ui/icons/Share';
import ZoomInIcon from '@material-ui/icons/ZoomIn';
import ZoomOutIcon from '@material-ui/icons/ZoomOut';
import Tooltip from '@material-ui/core/Tooltip';

import copy from 'copy-to-clipboard';

import { parseFlowPoints } from '../flowparser/Flowparser'

function LayerTypes() {
  return {
    linear: {
      title: 'Linear',
      ref: 'nn.Linear',
      id: 0,
      params: {
        in_features: { type: 'int', current: 1},
        out_features: { type: 'int', current: 1},
        bias: { type: 'bool', current: 'True'}
      },
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.out_features.current
        return out_shape
      },
      autoparams: (inp, p) => {
        p.in_features.current = inp[inp.length-1]
        return p
      },
    },
    flatten: {
      title: 'Flatten',
      ref: 'Flatten',
      id: 1,
      params: {},
      outshape: (inp, p) => {
        var features = 1/inp[0]
        inp.map(val => {features *= val})
        return [inp[0], Math.round(features)]
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    conv1d: {
      title: 'Conv1d',
      ref: 'nn.Conv1d',
      id: 2,
      params: {
        in_channels: { type: 'int', current: 1 },
        out_channels: { type: 'int', current: 1 },
        kernel_size: { type: 'tuple', current: [3] },
        stride: { type: 'tuple', current: [1] },
        padding: { type: 'tuple', current: [0] },
        dilation: { type: 'tuple', current: [1] },
        groups: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' }
      },
      outshape: (inp, p) => {
        const L_in = inp[2]
        const L_out = Math.floor(1 + (L_in + 2 * p.padding.current[0] - p.dilation.current[0] * (p.kernel_size.current[0] - 1) - 1) / p.stride.current[0])
        return [inp[0], p.out_channels.current.current, L_out]
      },
      autoparams: (inp, p) => {
        p.in_channels.current = inp[1]
        return p
      }
    },
    conv2d: {
      title: 'Conv2d',
      ref: 'nn.Conv2d',
      id: 3,
      params: {
        in_channels: { type: 'int', current: 1 },
        out_channels: { type: 'int', current: 1 },
        kernel_size: { type: 'tuple', current: [3, 3] },
        stride: { type: 'tuple', current: [1, 1] },
        padding: { type: 'tuple', current: [0, 0] },
        dilation: { type: 'tuple', current: [1, 1] },
        groups: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' }
      },
      outshape: (inp, p) => {
        const H_in = inp[2]
        const W_in = inp[3]
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.current[0] - p.dilation.current[0] * (p.kernel_size.current[0] - 1) - 1) / p.stride.current[0])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.current[1] - p.dilation.current[1] * (p.kernel_size.current[1] - 1) - 1) / p.stride.current[1])
        return [inp[0], p.out_channels.current, H_out, W_out]
      },
      autoparams: (inp, p) => {
        p.in_channels.current = inp[1]
        return p
      }
    },
    conv3d: {
      title: 'Conv3d',
      ref: 'nn.Conv3d',
      id: 4,
      params: {
        in_channels: { type: 'int', current: 1 },
        out_channels: { type: 'int', current: 1 },
        kernel_size: { type: 'tuple', current: [3, 3, 3] },
        stride: { type: 'tuple', current: [1, 1, 1] },
        padding: { type: 'tuple', current: [0, 0, 0] },
        dilation: { type: 'tuple', current: [1, 1, 1] },
        groups: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' }
      },
      outshape: (inp, p) => {
        const D_in = inp[2]
        const H_in = inp[3]
        const W_in = inp[4]
        const D_out = Math.floor(1 + (D_in + 2 * p.padding.current[0] - p.dilation.current[0] * (p.kernel_size.current[0] - 1) - 1) / p.stride.current[0])
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.current[1] - p.dilation.current[1] * (p.kernel_size.current[1] - 1) - 1) / p.stride.current[1])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.current[2] - p.dilation.current[2] * (p.kernel_size.current[2] - 1) - 1) / p.stride.current[2])
        return [inp[0], p.out_channels.current, D_out, H_out, W_out]
      },
      autoparams: (inp, p) => {
        p.in_channels.current = inp[1]
        return p
      }
    },
    maxpool1d: {
      title: 'MaxPool1d',
      ref: 'nn.MaxPool1d',
      id: 5,
      params: {
        kernel_size: { type: 'int', current: 3 },
        stride: { type: 'int', current: 1 },
        padding: { type: 'int', current: 0 },
        dilation: { type: 'int', current: 1 },
        return_indices: { type: 'bool', current: 'False' },
        ceil_mode: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        const L_in = inp[2]
        const L_out = Math.floor(1 + (L_in + 2 * p.padding.current[0] - p.dilation.current[0] * (p.kernel_size.current[0] - 1) - 1) / p.stride.current[0])
        return [inp[0], inp[0], L_out]
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    maxpool2d: {
      title: 'MaxPool2d',
      ref: 'nn.MaxPool2d',
      id: 6,
      params: {
        kernel_size: { type: 'tuple', current: [3, 3] },
        stride: { type: 'tuple', current: [1, 1] },
        padding: { type: 'tuple', current: [0, 0] },
        dilation: { type: 'tuple', current: [1, 1] },
        return_indices: { type: 'bool', current: 'False' },
        ceil_mode: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        const H_in = inp[2]
        const W_in = inp[3]
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.current[0] - p.dilation.current[0] * (p.kernel_size.current[0] - 1) - 1) / p.stride.current[0])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.current[1] - p.dilation.current[1] * (p.kernel_size.current[1] - 1) - 1) / p.stride.current[1])
        return [inp[0], inp[1], H_out, W_out]
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    maxpool3d: {
      title: 'MaxPool3d',
      ref: 'nn.MaxPool3d',
      id: 7,
      params: {
        kernel_size: { type: 'tuple', current: [3, 3, 3] },
        stride: { type: 'tuple', current: [1, 1, 1] },
        padding: { type: 'tuple', current: [0, 0, 0] },
        dilation: { type: 'tuple', current: [1, 1, 1] },
        return_indices: { type: 'bool', current: 'False' },
        ceil_mode: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        const D_in = inp[2]
        const H_in = inp[3]
        const W_in = inp[4]
        const D_out = Math.floor(1 + (D_in + 2 * p.padding.current[0] - p.dilation.current[0] * (p.kernel_size.current[0] - 1) - 1) / p.stride.current[0])
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.current[1] - p.dilation.current[1] * (p.kernel_size.current[1] - 1) - 1) / p.stride.current[1])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.current[2] - p.dilation.current[2] * (p.kernel_size.current[2] - 1) - 1) / p.stride.current[2])
        return [inp[0], inp[1], D_out, H_out, W_out]
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    batchnorm1d: {
      title: 'BatchNorm1d',
      ref: 'nn.BatchNorm1d',
      id: 8,
      params: {
        num_features: { type: 'int', current: 1 },
        eps: { type: 'double', current: 1e-5 },
        momentum: { type: 'double', current: 0.1 },
        affine: { type: 'bool', current: 'True' },
        track_running_stats: { type: 'bool', current: 'True' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    batchnorm2d: {
      title: 'BatchNorm2d',
      ref: 'nn.BatchNorm2d',
      id: 9,
      params: {
        num_features: { type: 'int', current: 1 },
        eps: { type: 'double', current: 1e-5 },
        momentum: { type: 'double', current: 0.1 },
        affine: { type: 'bool', current: 'True' },
        track_running_stats: { type: 'bool', current: 'True' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    batchnorm3d: {
      title: 'BatchNorm3d',
      ref: 'nn.BatchNorm3d',
      id: 10,
      params: {
        num_features: { type: 'int', current: 1 },
        eps: { type: 'double', current: 1e-5 },
        momentum: { type: 'double', current: 0.1 },
        affine: { type: 'bool', current: 'True' },
        track_running_stats: { type: 'bool', current: 'True' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    groupnorm: {
      title: 'GroupNorm',
      ref: 'nn.GroupNorm',
      id: 11,
      params: {
        num_groups: { type: 'int', current: 1 },
        num_channels: { type: 'int', current: 1 },
        eps: { type: 'double', current: 1e-5 },
        affine: { type: 'bool', current: 'True' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_channels.current = inp[1]
        return p
      }
    },
    instancenorm1d: {
      title: 'InstanceNorm1d',
      ref: 'nn.InstanceNorm1d',
      id: 12,
      params: {
        num_features: { type: 'int', current: 1 },
        eps: { type: 'double', current: 1e-5 },
        momentum: { type: 'double', current: 0.1 },
        affine: { type: 'bool', current: 'False' },
        track_running_stats: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    instancenorm2d: {
      title: 'InstanceNorm2d',
      ref: 'nn.InstanceNorm2d',
      id: 13,
      params: {
        num_features: { type: 'int', current: 1 },
        eps: { type: 'double', current: 1e-5 },
        momentum: { type: 'double', current: 0.1 },
        affine: { type: 'bool', current: 'False' },
        track_running_stats: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    instancenorm3d: {
      title: 'InstanceNorm3d',
      ref: 'nn.InstanceNorm3d',
      id: 14,
      params: {
        num_features: { type: 'int', current: 1 },
        eps: { type: 'double', current: 1e-5 },
        momentum: { type: 'double', current: 0.1 },
        affine: { type: 'bool', current: 'False' },
        track_running_stats: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    rnn: {
      title: 'RNN',
      ref: 'nn.RNN',
      id: 15,
      params: {
        input_size: { type: 'int', current: 1 },
        hidden_size: { type: 'int', current: 1 },
        num_layers: { type: 'int', current: 1 },
        nonlinearity: { type: 'select', options:['relu', 'tanh'], current: 'tanh' },
        bias: { type: 'bool', current: 'True' },
        batch_first: { type: 'bool', current: 'False' },
        dropout: { type: 'double', current: 0 },
        bidirectional: { type: 'bool', current: 'False' },
      },
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.current
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.current = inp[inp.length-1]
        return p
      }
    },
    lstm: {
      title: 'LSTM',
      ref: 'nn.LSTM',
      id: 16,
      params: {
        input_size: { type: 'int', current: 1 },
        hidden_size: { type: 'int', current: 1 },
        num_layers: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' },
        batch_first: { type: 'bool', current: 'False' },
        dropout: { type: 'double', current: 0 },
        bidirectional: { type: 'bool', current: 'False' },
      },
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.current
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.current = inp[inp.length-1]
        return p
      }
    },
    gru: {
      title: 'GRU',
      ref: 'nn.GRU',
      id: 17,
      params: {
        input_size: { type: 'int', current: 1 },
        hidden_size: { type: 'int', current: 1 },
        num_layers: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' },
        batch_first: { type: 'bool', current: 'False' },
        dropout: { type: 'double', current: 0 },
        bidirectional: { type: 'bool', current: 'False' },
      },
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.current
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.current = inp[inp.length-1]
        return p
      }
    },
    rnncell: {
      title: 'RNNCell',
      ref: 'nn.RNNCell',
      id: 18,
      params: {
        input_size: { type: 'int', current: 1 },
        hidden_size: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' },
        nonlinearity: { type: 'select', options:['relu', 'tanh'], current: 'tanh' },
      },
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.current
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.current = inp[inp.length-1]
        return p
      }
    },
    lstmcell: {
      title: 'LSTMCell',
      ref: 'nn.LSTMCell',
      id: 19,
      params: {
        input_size: { type: 'int', current: 1 },
        hidden_size: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' },
      },
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.current
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.current = inp[inp.length-1]
        return p
      }
    },
    grucell: {
      title: 'GRUCell',
      ref: 'nn.GRUCell',
      id: 20,
      params: {
        input_size: { type: 'int', current: 1 },
        hidden_size: { type: 'int', current: 1 },
        bias: { type: 'bool', current: 'True' },
      },
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.current
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.current = inp[inp.length-1]
        return p
      }
    },
    dropout: {
      title: 'Dropout',
      ref: 'nn.Dropout',
      id: 21,
      params: {
        p: { type: 'double', current: 0.5 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    dropout2d: {
      title: 'Dropout2d',
      ref: 'nn.Dropout2d',
      id: 22,
      params: {
        p: { type: 'double', current: 0.5 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    dropout3d: {
      title: 'Dropout3d',
      ref: 'nn.Dropout3d',
      id: 23,
      params: {
        p: { type: 'double', current: 0.5 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    alphadropout: {
      title: 'AlphaDropout',
      ref: 'nn.AlphaDropout',
      id: 24,
      params: {
        p: { type: 'double', current: 0.5 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
  }
}

function ActivationTypes() {
  return {
    elu: {
      title: 'ELU',
      ref: 'nn.ELU',
      id: 25,
      params: {
        alpha: { type: 'double', current: 1 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    hardshrink: {
      title: 'Hardshrink',
      ref: 'nn.Hardshrink',
      id: 26,
      params: {
        lambd: { type: 'double', current: '0.5' },
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    hardtanh: {
      title: 'Hardtanh',
      ref: 'nn.Hardtanh',
      id: 27,
      params: {
        min_val: { type: 'double', current: -1 },
        max_val: { type: 'double', current: 1 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    leakyrelu: {
      title: 'LeakyReLU',
      ref: 'nn.LeakyReLU',
      id: 28,
      params: {
        negative_slope: { type: 'double', current: 0.01 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    logsigmoid: {
      title: 'LogSigmoid',
      ref: 'nn.LogSigmoid',
      id: 29,
      params: {},
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    prelu: {
      title: 'PReLU',
      ref: 'nn.PReLU',
      id: 30,
      params: {
        num_parameters: { type: 'int', current: 1 },
        init: { type: 'double', current: 0.25 }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    relu: {
      title: 'ReLU',
      ref: 'nn.ReLU',
      id: 31,
      params: {
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    relu6: {
      title: 'ReLU6',
      ref: 'nn.ReLU6',
      id: 32,
      params: {
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    rrelu: {
      title: 'RReLU',
      ref: 'nn.RReLU',
      id: 33,
      params: {
        lower: { type: 'double', current: 1/8 },
        upper: { type: 'double', current: 1/3 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    selu: {
      title: 'SELU',
      ref: 'nn.SELU',
      id: 34,
      params: {
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    celu: {
      title: 'CELU',
      ref: 'nn.CELU',
      id: 35,
      params: {
        alpha: { type: 'double', current: 1 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    sigmoid: {
      title: 'Sigmoid',
      ref: 'nn.Sigmoid',
      id: 36,
      params: {},
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    softplus: {
      title: 'Softplus',
      ref: 'nn.Softplus',
      id: 37,
      params: {
        beta: { type: 'double', current: 1 },
        threshold: { type: 'double', current: 20 }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    softshrink: {
      title: 'Softshrink',
      ref: 'nn.Softshrink',
      id: 38,
      params: {
        lambd: { type: 'double', current: 0.5 }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    softsign: {
      title: 'Softsign',
      ref: 'nn.Softsign',
      id: 39,
      params: {},
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    tanh: {
      title: 'Tanh',
      ref: 'nn.Tanh',
      id: 40,
      params: {},
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    tanhshrink: {
      title: 'Tanhshrink',
      ref: 'nn.Tanhshrink',
      id: 41,
      params: {},
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    threshold: {
      title: 'Threshold',
      ref: 'nn.Threshold',
      id: 42,
      params: {
        threshold: { type: 'double', current: 0.1 },
        value: { type: 'double', current: 0.1 },
        inplace: { type: 'bool', current: 'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    softmin: {
      title: 'Softmin',
      ref: 'nn.Softmin',
      id: 43,
      params: {
        dim: { type: 'int', current: -1 }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    softmax: {
      title: 'Softmax',
      ref: 'nn.Softmax',
      id: 44,
      params: {
        dim: { type: 'int', current: -1 }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    softmax2d: {
      title: 'Softmax2d',
      ref: 'nn.Softmax2d',
      id: 45,
      params: {},
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    logsoftmax: {
      title: 'LogSoftmax',
      ref: 'nn.LogSoftmax',
      id: 46,
      params: {
        dim: { type: 'int', current: -1 }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
  }
}

function DefaultPointState(key) {
  return {
    key: parseInt(key),
    name: null,
    openMenu: 'nothing',
    anchorEl: null,
    x: 0,
    y: 0,
    width: 375,
    height: 200,
    inputs: [],
    outputs: [],
    gotInput: true,
    gotOutput: true,
    isOpen: false,
    layertypes: LayerTypes(),
    activationtypes: ActivationTypes(),
    flowtype:'linear',
    output_shape: [1,1],
    input_point_dims: 2
  }
}

/*
PARSING:
 _  =  New flowpoint
 &  =  New flowpoint main param
 ,  =  New param within main
*/

function inputToString(point) {
  // Default msg contents: [key, name, flowtype_id, outputs, position]
  var msg = '' + point.key + '&' + (point.name ? point.name : '') + '&i&'
  var gotKey = false
  // Adding outputs
  point.outputs.map(key => {
    gotKey = true
    msg += key + ','
  })
  if (gotKey) { msg = msg.substring(0, msg.length - 1) }
  msg += '&'
  // Adding position
  msg += Math.round(point.x/10) + ',' + Math.round(point.y/10) + '&'
  // Adding parameters: output_shape
  gotKey = false
  point.output_shape.map(val => {
    gotKey = true
    msg += val + ','
  })
  if (gotKey) { msg = msg.substring(0, msg.length - 1) }
  return msg
}

// key, name, type_id, outputs, params

function pointToString(point) {
  // Getting params
  var params = null
  if (point.flowtype in point.layertypes) {
    params = point.layertypes[point.flowtype]
  } else {
    params = point.activationtypes[point.flowtype]
  }
  // Default msg contents: [key, name, flowtype_id, outputs, position]
  var msg = '' + point.key + '&' + (point.name ? point.name : '') + '&' + params.id + '&'
  var gotKey = false
  // Adding outputs
  point.outputs.map(key => {
    gotKey = true
    msg += key + ','
  })
  if (gotKey) { msg = msg.substring(0, msg.length - 1) }
  msg += '&'
  // Adding position
  msg += Math.round(point.x/10) + ',' + Math.round(point.y/10) + '&'
  // Adding parameters
  gotKey = false
  Object.keys(params.params).sort().map((paramkey, idx) => {
    gotKey = true
    switch (params.params[paramkey].type) {
      case 'bool':
        msg += params.params[paramkey].current.includes('True') ? '1,' : '0,'
        break;
      case 'tuple':
        params.params[paramkey].current.map(val => {
          msg += val + ';'
        })
        msg = msg.substring(0, msg.length - 1) + ','
        break;
      default:
        msg += params.params[paramkey].current + ','
    }
  })
  if (gotKey) { msg = msg.substring(0, msg.length - 1) }
  return msg
}


// Default msg contents: [key, name, flowtype_id, outputs, position]
function parsePoint(msg) {
  var newpoint = DefaultPointState(0)
  const mainparams = msg.split('&')
  newpoint.key = parseInt(mainparams[0])
  newpoint.name = mainparams[1]
  // Setting flow type
  if (mainparams[2].includes('i')) {
    newpoint.flowtype = 'input'
  } else {
    const raw_type = parseInt(mainparams[2])
    var done = false
    var params = null
    Object.keys(newpoint.layertypes).map(t_key => {
      if (!done && (raw_type === newpoint.layertypes[t_key].id)) {
        done = true
        newpoint.flowtype = t_key.toString()
        params = newpoint.layertypes[newpoint.flowtype]
      }
    })
    Object.keys(newpoint.activationtypes).map(t_key => {
      if (!done && (raw_type === newpoint.activationtypes[t_key].id)) {
        done = true
        newpoint.flowtype = t_key.toString()
        params = newpoint.activationtypes[newpoint.flowtype]
      }
    })
  }
  // Adding outputs
  mainparams[3].split(',').map(val => {
    if (!('').includes(val)) {
      newpoint.outputs.push(parseInt(val))
    }
  })
  // Adding position
  newpoint.x = parseInt(mainparams[4].split(',')[0]) * 10
  newpoint.y = parseInt(mainparams[4].split(',')[1]) * 10
  // Adding parameters
  if (newpoint.flowtype.includes('input')) {
    newpoint.output_shape = []
    mainparams[5].split(',').map(val => {
      newpoint.output_shape.push(parseInt(val))
    })
    newpoint.input_point_dims = newpoint.output_shape.length
  } else {
    var raw_params = mainparams[5].split(',')
    Object.keys(params.params).sort().map((paramkey, idx) => {
      switch (params.params[paramkey].type) {
        case 'bool':
          params.params[paramkey].current = (parseInt(raw_params[idx]) === 1) ? 'True' : 'False'
          break;
        case 'double':
          params.params[paramkey].current = parseFloat(raw_params[idx])
          break;
        case 'tuple':
          params.params[paramkey].current = []
          raw_params[idx].split(';').map(val => {
            params.params[paramkey].current.push(parseInt(val))
          })
          break;
        default: params.params[paramkey].current = parseInt(raw_params[idx])
      }
    })
  }
  return newpoint
}

export function parse_url(msg) {
  var flowpoints = {}
  msg.split('_').map(pointmsg => {
    var newpoint = parsePoint(pointmsg)
    flowpoints[newpoint.key] = newpoint
  })
  Object.keys(flowpoints).map(key => {
    key = parseInt(key)
    flowpoints[key].outputs.map(outkey => {
      outkey = parseInt(outkey)
      if (!flowpoints[outkey].inputs.includes(key)) {
        flowpoints[outkey].inputs.push(key)
      }
    })
  })
  return flowpoints
}

export function gen_url(flowpoints, order) {
  var msg = ''
  order.map(key => {
    if (flowpoints[key].flowtype.includes('input')) {
      msg += inputToString(flowpoints[key]) + '_'
    } else {
      msg += pointToString(flowpoints[key]) + '_'
    }
  })
  return msg.substring(0, msg.length - 1)
}

export const AppBottom = (props) => {
  const snapX = props.snapX || 25
  const snapY = props.snapY || 25
  const settings = props.refresh().settings
  return (
    <div>
      <div style={{position:'fixed', bottom:0, left:0, padding:'5px', paddingRight:'30px'}}>
        <Tooltip title='Zoom in'>
          <IconButton
            onClick={() => {
              var settings = props.refresh().settings
              settings.zoom = Math.min(1.5, Math.max(0.5, settings.zoom + 0.1))
              props.updateSettings(settings)
            }}>
            <ZoomInIcon/>
          </IconButton>
        </Tooltip>
        <Tooltip title='Zoom out'>
          <IconButton
            onClick={() => {
              var settings = props.refresh().settings
              settings.zoom = Math.min(1.5, Math.max(0.5, settings.zoom - 0.1))
              props.updateSettings(settings)
            }}>
            <ZoomOutIcon/>
          </IconButton>
        </Tooltip>
      </div>
        <div style={{position:'fixed', bottom:0, right:0, padding:'10px', paddingLeft:'30px'}}>
        <Tooltip title="Add">
          <Fab
            style={{background:'#00a8ff', color:'#ffffff'}}
            aria-label="Add"
            onClick={() => {
              var state = props.refresh()
              var flowpoints = state.flowpoints
              var settings = state.settings
              var newpoint = DefaultPointState(settings.count)
              const snapX = settings.snapX
              const snapY = settings.snapY
              newpoint.x = props.lastPosX + (Object.keys(flowpoints).length > 0 ? snapX : 0)
              newpoint.y = props.lastPosY
              if (Object.keys(flowpoints).length == 0) {
                newpoint.gotInput = false
                newpoint.flowtype = 'input'
              }
              props.updateLastPos(newpoint.x, newpoint.y)
              flowpoints[settings.count] = newpoint
              settings.count += 1
              props.updateView(flowpoints, settings)
            }}>
            <AddIcon />
          </Fab>
        </Tooltip>{' '}
        <Tooltip title="Show/hide code">
          <Fab
            style={{background:'#00ed3f', color:'white'}}
            aria-label="Add"
            onClick={() => {
              var settings = props.refresh().settings
              settings.showPaper ^= true
              props.updateSettings(settings)
            }}>
            <CodeIcon />
          </Fab>
        </Tooltip>{' '}
        <Tooltip title="Copy code">
          <Fab
            style={{background:'#bfbfbf', color:'black'}}
            aria-label="Add"
            onClick={() => {
              copy(parseFlowPoints(props.refresh().flowpoints))
              var settings = props.refresh().settings
              settings.snackbarMsg = 'Copied code to clipboard'
              settings.showSnackbar = true
              props.updateSettings(settings)
            }}>
            <SaveIcon />
          </Fab>
        </Tooltip>{'  '}
      </div>
    </div>
  )
}

/*
<Tooltip title="Share link">
  <Fab
    style={{background:'#7700ff', color:'white'}}
    aria-label="Share"
    onClick={() => {
      // 'https://mariusbrataas.github.io/torchflow/?p='
      copy('localhost:3000/' + 'load?' + gen_url(props.refresh().flowpoints, props.order))
      var settings = props.refresh().settings
      settings.snackbarMsg = 'Copied link to clipboard'
      settings.showSnackbar = true
      props.updateSettings(settings)
    }}>
    <ShareIcon />
  </Fab>
</Tooltip>

*/
