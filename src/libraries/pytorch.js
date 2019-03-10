import React, { Component } from 'react';

export function PyTorchModules() {
  return {
    input: {
      title: 'Input',
      params: {
        n_dims: { min:1, max:Infinity, current:2 },
        dimensions: { type:'tuple', min:1, max:Infinity, current:[1,1] }
      },
      autoparams: (inp, p) => {
        return p
      },
      outshape: (inp, p) => {
        return p.dimensions.current
      }
    },
    linear: {
      title: 'Linear',
      ref: 'nn.Linear',
      id: 0,
      params: {
        in_features: {type:'int', min:1, max:Infinity, current:1 },
        out_features: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type:'bool', current:'True' }
      },
      outshape: (inp, p) => {
        var idx = 0;
        var out_shape = inp.map((val, index) => {
          idx = index;
          return val
        });
        out_shape[idx] = p.out_features.current;
        return out_shape
      },
      autoparams: (inp, p) => {
        p.in_features.current = inp[inp.length - 1]
        return p
      }
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
        in_channels: { type:'int', min:1, max:Infinity, current:1 },
        out_channels: { type:'int', min:1, max:Infinity, current:1 },
        kernel_size: { type:'tuple', min:1, max:Infinity, current:[3] },
        stride: { type:'tuple', min:1, max:Infinity, current:[1] },
        padding: { type:'tuple', min:0, max:Infinity, current:[0] },
        dilation: { type:'tuple', min:1, max:Infinity, current:[1] },
        groups: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' }
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
        in_channels: { type:'int', min:1, max:Infinity, current:1 },
        out_channels: { type:'int', min:1, max:Infinity, current:1 },
        kernel_size: { type:'tuple', min:1, max:Infinity, current:[3, 3] },
        stride: { type:'tuple', min:1, max:Infinity, current:[1, 1] },
        padding: { type:'tuple', min:0, max:Infinity, current:[0, 0] },
        dilation: { type:'tuple', min:1, max:Infinity, current:[1, 1] },
        groups: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' }
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
        in_channels: { type:'int', min:1, max:Infinity, current:1 },
        out_channels: { type:'int', min:1, max:Infinity, current:1 },
        kernel_size: { type:'tuple', min:1, max:Infinity, current:[3, 3, 3] },
        stride: { type:'tuple', min:1, max:Infinity, current:[1, 1, 1] },
        padding: { type:'tuple', min:0, max:Infinity, current:[0, 0, 0] },
        dilation: { type:'tuple', min:1, max:Infinity, current:[1, 1, 1] },
        groups: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' }
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
        kernel_size: { type:'int', min:1, max:Infinity, current:3 },
        stride: { type:'int', min:1, max:Infinity, current:1 },
        padding: { type:'int', min:0, max:Infinity, current:0 },
        dilation: { type:'int', min:1, max:Infinity, current:1 },
        return_indices: { type: 'bool', current:'False' },
        ceil_mode: { type: 'bool', current:'False' }
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
        kernel_size: { type:'tuple', min:1, max:Infinity, current:[3, 3] },
        stride: { type:'tuple', min:1, max:Infinity, current:[1, 1] },
        padding: { type:'tuple', min:0, max:Infinity, current:[0, 0] },
        dilation: { type:'tuple', min:1, max:Infinity, current:[1, 1] },
        return_indices: { type: 'bool', current:'False' },
        ceil_mode: { type: 'bool', current:'False' }
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
        kernel_size: { type:'tuple', min:1, max:Infinity, current:[3, 3, 3] },
        stride: { type:'tuple', min:1, max:Infinity, current:[1, 1, 1] },
        padding: { type:'tuple', min:0, max:Infinity, current:[0, 0, 0] },
        dilation: { type:'tuple', min:1, max:Infinity, current:[1, 1, 1] },
        return_indices: { type: 'bool', current:'False' },
        ceil_mode: { type: 'bool', current:'False' }
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
        num_features: { type:'int', min:1, max:Infinity, current:1 },
        eps: { type:'double', min:1e-5, max:1, current:1e-5 },
        momentum: { type:'double', min:0, max:Infinity, current:0.1 },
        affine: { type: 'bool', current:'True' },
        track_running_stats: { type: 'bool', current:'True' }
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
        num_features: { type:'int', min:1, max:Infinity, current:1 },
        eps: { type:'double', min:1e-5, max:1, current:1e-5 },
        momentum: { type:'double', min:0, max:Infinity, current:0.1 },
        affine: { type: 'bool', current:'True' },
        track_running_stats: { type: 'bool', current:'True' }
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
        num_features: { type:'int', min:1, max:Infinity, current:1 },
        eps: { type:'double', min:1e-5, max:1, current:1e-5 },
        momentum: { type:'double', min:0, max:Infinity, current:0.1 },
        affine: { type: 'bool', current:'True' },
        track_running_stats: { type: 'bool', current:'True' }
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
        num_groups: { type:'int', min:1, max:Infinity, current:1 },
        num_channels: { type:'int', min:1, max:Infinity, current:1 },
        eps: { type:'double', min:1e-5, max:1, current:1e-5 },
        affine: { type: 'bool', current:'True' }
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
        num_features: { type:'int', min:1, max:Infinity, current:1 },
        eps: { type:'double', min:1e-5, max:1, current:1e-5 },
        momentum: { type:'double', min:0, max:Infinity, current:0.1 },
        affine: { type: 'bool', current:'False' },
        track_running_stats: { type: 'bool', current:'False' }
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
        num_features: { type:'int', min:1, max:Infinity, current:1 },
        eps: { type:'double', min:1e-5, max:1, current:1e-5 },
        momentum: { type:'double', min:0, max:Infinity, current:0.1 },
        affine: { type: 'bool', current:'False' },
        track_running_stats: { type: 'bool', current:'False' }
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
        num_features: { type:'int', min:1, max:Infinity, current:1 },
        eps: { type:'double', min:1e-5, max:1, current:1e-5 },
        momentum: { type:'double', min:0, max:Infinity, current:0.1 },
        affine: { type: 'bool', current:'False' },
        track_running_stats: { type: 'bool', current:'False' }
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
        input_size: { type:'int', min:1, max:Infinity, current:1 },
        hidden_size: { type:'int', min:1, max:Infinity, current:1 },
        num_layers: { type:'int', min:1, max:Infinity, current:1 },
        nonlinearity: { type: 'select', options:['relu', 'tanh'], current:'tanh' },
        bias: { type: 'bool', current:'True' },
        batch_first: { type: 'bool', current:'False' },
        dropout: { type:'double', min:0, max:Infinity, current:0 },
        bidirectional: { type: 'bool', current:'False' },
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
        input_size: { type:'int', min:1, max:Infinity, current:1 },
        hidden_size: { type:'int', min:1, max:Infinity, current:1 },
        num_layers: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' },
        batch_first: { type: 'bool', current:'False' },
        dropout: { type:'double', min:0, max:Infinity, current:0 },
        bidirectional: { type: 'bool', current:'False' },
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
        input_size: { type:'int', min:1, max:Infinity, current:1 },
        hidden_size: { type:'int', min:1, max:Infinity, current:1 },
        num_layers: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' },
        batch_first: { type: 'bool', current:'False' },
        dropout: { type:'double', min:0, max:Infinity, current:0 },
        bidirectional: { type: 'bool', current:'False' },
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
        input_size: { type:'int', min:1, max:Infinity, current:1 },
        hidden_size: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' },
        nonlinearity: { type: 'select', options:['relu', 'tanh'], current:'tanh' },
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
        input_size: { type:'int', min:1, max:Infinity, current:1 },
        hidden_size: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' },
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
        input_size: { type:'int', min:1, max:Infinity, current:1 },
        hidden_size: { type:'int', min:1, max:Infinity, current:1 },
        bias: { type: 'bool', current:'True' },
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
        p: { type:'double', min:0, max:1, current:0.5 },
        inplace: { type: 'bool', current:'False' }
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
        p: { type:'double', min:0, max:1, current:0.5 },
        inplace: { type: 'bool', current:'False' }
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
        p: { type:'double', min:0, max:1, current:0.5 },
        inplace: { type: 'bool', current:'False' }
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
        p: { type:'double', min:0, max:1, current:0.5 },
        inplace: { type: 'bool', current:'False' }
      },
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    elu: {
      title: 'ELU',
      ref: 'nn.ELU',
      id: 25,
      params: {
        alpha: { type:'double', min:0, max:Infinity, current:1 },
        inplace: { type:'bool', current:'False' }
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
        lambd: { type:'double', min:-Infinity, max:Infinity, current:'0.5' },
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
        min_val: { type:'double', min:-Infinity, max:Infinity, current:-1 },
        max_val: { type:'double', min:-Infinity, max:Infinity, current:1 },
        inplace: { type:'bool', current:'False' }
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
        negative_slope: { type:'double', min:0, max:Infinity, current:1e-2 },
        inplace: { type:'bool', current:'False' }
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
        num_parameters: { type:'int', min:0, max:Infinity, current:1 },
        init: { type:'double', min:0, max:Infinity, current:0.25 }
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
        inplace: { type:'bool', current:'False' }
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
        inplace: { type:'bool', current:'False' }
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
        lower: { type:'double', min:0, max:Infinity, current:1/8 },
        upper: { type:'double', min:0, max:Infinity, current:1/3 },
        inplace: { type:'bool', current:'False' }
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
        inplace: { type:'bool', current:'False' }
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
        alpha: { type:'double', min:0, max:Infinity, current:1 },
        inplace: { type:'bool', current:'False' }
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
        beta: { type:'double', min:0, max:Infinity, current:1 },
        threshold: { type:'double', min:0, max:Infinity, current:20 }
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
        lambd: { type:'double', min:0, max:Infinity, current:0.5 }
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
        threshold: { type:'double', min:0, max:Infinity, current:0.1 },
        value: { type:'double', min:0, max:Infinity, current:0.1 },
        inplace: { type:'bool', current:'False' }
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
        dim: { type:'int', min:0, max:Infinity, current:-1 }
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
        dim: { type:'int', min:0, max:Infinity, current:-1 }
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
        dim: { type:'int', min:0, max:Infinity, current:-1 }
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
