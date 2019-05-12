export function getPyTorchAutoparams() {
  return {
    "AdaptiveAvgPool1d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 1] = p.output_size.value;
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AdaptiveAvgPool2d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 2] = p.output_size.value[0];
        tmp[tmp.length - 1] = p.output_size.value[1];
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AdaptiveAvgPool3d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 3] = p.output_size.value[0];
        tmp[tmp.length - 2] = p.output_size.value[1];
        tmp[tmp.length - 1] = p.output_size.value[2];
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AdaptiveLogSoftmaxWithLoss": {
      outshape: (inp, p) => {
        return inp.slice(0, inp.length - 1)
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AdaptiveMaxPool1d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 1] = p.output_size.value;
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AdaptiveMaxPool2d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 2] = p.output_size.value[0];
        tmp[tmp.length - 1] = p.output_size.value[1];
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AdaptiveMaxPool3d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 3] = p.output_size.value[0];
        tmp[tmp.length - 2] = p.output_size.value[1];
        tmp[tmp.length - 1] = p.output_size.value[2];
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AlphaDropout": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AvgPool1d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 1] = 1 + ((inp[inp.length - 1] + 2 * p.padding.value - p.kernel_size.value) / p.stride.value)
        if (p.ceil_mode.value) {
          tmp[tmp.length - 1] = Math.ceil(tmp[tmp.length - 1])
        } else {
          tmp[tmp.length - 1] = Math.floor(tmp[tmp.length - 1])
        }
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AvgPool2d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 2] = 1 + ((inp[inp.length - 2] + 2 * p.padding.value[0] - p.kernel_size.value[0]) / p.stride.value[0])
        tmp[tmp.length - 1] = 1 + ((inp[inp.length - 1] + 2 * p.padding.value[1] - p.kernel_size.value[1]) / p.stride.value[1])
        if (p.ceil_mode.value) {
          tmp[tmp.length - 2] = Math.ceil(tmp[tmp.length - 2])
          tmp[tmp.length - 1] = Math.ceil(tmp[tmp.length - 1])
        } else {
          tmp[tmp.length - 2] = Math.floor(tmp[tmp.length - 2])
          tmp[tmp.length - 1] = Math.floor(tmp[tmp.length - 1])
        }
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "AvgPool3d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 3] = 1 + ((inp[inp.length - 3] + 2 * p.padding.value[0] - p.kernel_size.value[0]) / p.stride.value[0])
        tmp[tmp.length - 2] = 1 + ((inp[inp.length - 2] + 2 * p.padding.value[1] - p.kernel_size.value[1]) / p.stride.value[1])
        tmp[tmp.length - 1] = 1 + ((inp[inp.length - 1] + 2 * p.padding.value[2] - p.kernel_size.value[2]) / p.stride.value[2])
        if (p.ceil_mode.value) {
          tmp[tmp.length - 3] = Math.ceil(tmp[tmp.length - 3])
          tmp[tmp.length - 2] = Math.ceil(tmp[tmp.length - 2])
          tmp[tmp.length - 1] = Math.ceil(tmp[tmp.length - 1])
        } else {
          tmp[tmp.length - 3] = Math.floor(tmp[tmp.length - 3])
          tmp[tmp.length - 2] = Math.floor(tmp[tmp.length - 2])
          tmp[tmp.length - 1] = Math.floor(tmp[tmp.length - 1])
        }
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "BatchNorm1d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    "BatchNorm2d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    "BatchNorm3d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    "Bilinear": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 1] = p.out_features.value;
        return tmp
      },
      autoparams: (inp, p) => {
        p.num_features.current = inp[1]
        return p
      }
    },
    "CELU": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ConstantPad1d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 1] = inp[inp.length - 1] + 2 * p.padding.value;
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ConstantPad2d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 2] = inp[inp.length - 2] + 2 * p.padding.value[0]
        tmp[tmp.length - 1] = inp[inp.length - 1] + 2 * p.padding.value[1]
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ConstantPad3d": {
      outshape: (inp, p) => {
        var tmp = inp;
        tmp[tmp.length - 3] = inp[inp.length - 3] + 2 * p.padding.value[0]
        tmp[tmp.length - 2] = inp[inp.length - 2] + 2 * p.padding.value[1]
        tmp[tmp.length - 1] = inp[inp.length - 1] + 2 * p.padding.value[2]
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Conv1d": {
      outshape: (inp, p) => {
        var tmp = inp;
        const L_in = inp[inp.length - 1]
        const L_out = Math.floor(1 + (L_in + 2 * p.padding.value - p.dilation.value * (p.kernel_size.value - 1) - 1) / p.stride.value)
        tmp[tmp.length - 2] = p.out_channels.value
        tmp[tmp.length - 1] = L_out
        return tmp
      },
      autoparams: (inp, p) => {
        p.in_channels.value = inp[inp.length - 2]
        return p
      }
    },
    "Conv2d": {
      outshape: (inp, p) => {
        var tmp = inp;
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.value[0] - p.dilation.value[0] * (p.kernel_size.value[0] - 1) - 1) / p.stride.value[0])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.value[1] - p.dilation.value[1] * (p.kernel_size.value[1] - 1) - 1) / p.stride.value[1])
        tmp[tmp.length - 3] = p.out_channels.value
        tmp[tmp.length - 2] = H_out
        tmp[tmp.length - 1] = W_out
        return tmp
      },
      autoparams: (inp, p) => {
        p.in_channels.value = inp[inp.length - 3]
        return p
      }
    },
    "Conv3d": {
      outshape: (inp, p) => {
        var tmp = inp;
        const D_in = inp[inp.length - 3]
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        const D_out = Math.floor(1 + (D_in + 2 * p.padding.value[0] - p.dilation.value[0] * (p.kernel_size.value[0] - 1) - 1) / p.stride.value[0])
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.value[1] - p.dilation.value[1] * (p.kernel_size.value[1] - 1) - 1) / p.stride.value[1])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.value[2] - p.dilation.value[2] * (p.kernel_size.value[2] - 1) - 1) / p.stride.value[2])
        tmp[tmp.length - 4] = p.out_channels.value
        tmp[tmp.length - 3] = D_out
        tmp[tmp.length - 2] = H_out
        tmp[tmp.length - 1] = W_out
        return tmp
      },
      autoparams: (inp, p) => {
        p.in_channels.value = inp[inp.length - 4]
        return p
      }
    },
    "ConvTranspose1d": {
      outshape: (inp, p) => {
        var tmp = inp;
        const L_in = inp[inp.length - 1]
        const L_out = (L_in - 1) * p.stride.value + p.dilation.value * (p.kernel_size.value - 1) + p.output_padding.value + 1
        tmp[tmp.length - 2] = p.out_channels.value
        tmp[tmp.length - 1] = L_out
        return tmp
      },
      autoparams: (inp, p) => {
        p.in_channels.value = inp[inp.length - 2]
        return p
      }
    },
    "ConvTranspose2d": {
      outshape: (inp, p) => {
        var tmp = inp;
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        const H_out = (H_in - 1) * p.stride.value[0] + p.dilation.value[0] * (p.kernel_size.value[0] - 1) + p.output_padding.value[0] + 1
        const W_out = (W_in - 1) * p.stride.value[1] + p.dilation.value[1] * (p.kernel_size.value[1] - 1) + p.output_padding.value[1] + 1
        tmp[tmp.length - 3] = p.out_channels.value
        tmp[tmp.length - 2] = H_out
        tmp[tmp.length - 1] = W_out
        return tmp
      },
      autoparams: (inp, p) => {
        p.in_channels.value = inp[inp.length - 3]
        return p
      }
    },
    "ConvTranspose3d": {
      outshape: (inp, p) => {
        var tmp = inp;
        const D_in = inp[inp.length - 3]
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        const D_out = (D_in - 1) * p.stride.value[0] + p.dilation.value[0] * (p.kernel_size.value[0] - 1) + p.output_padding.value[0] + 1
        const H_out = (H_in - 1) * p.stride.value[1] + p.dilation.value[1] * (p.kernel_size.value[1] - 1) + p.output_padding.value[1] + 1
        const W_out = (W_in - 1) * p.stride.value[2] + p.dilation.value[2] * (p.kernel_size.value[2] - 1) + p.output_padding.value[2] + 1
        tmp[tmp.length - 4] = p.out_channels.value
        tmp[tmp.length - 3] = D_out
        tmp[tmp.length - 2] = H_out
        tmp[tmp.length - 1] = W_out
        return tmp
      },
      autoparams: (inp, p) => {
        p.in_channels.value = inp[inp.length - 4]
        return p
      }
    },
    "Dropout": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Dropout2d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Dropout3d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ELU": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Embedding": {
      outshape: (inp, p) => {
        var tmp = inp
        tmp.push(p.embedding_dim.value)
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "EmbeddingBag": {
      outshape: (inp, p) => {
        var tmp = inp
        tmp.push(p.embedding_dim.value)
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Flatten": {
      outshape: (inp, p) => {
        var features = 1/inp[0]
        inp.map(val => {features *= val})
        return [inp[0], Math.round(features)]
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "FractionalMaxPool2d": {
      outshape: (inp, p) => {
        var tmp = inp;
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.value[0] - p.dilation.value[0] * (p.kernel_size.value[0] - 1) - 1) / p.stride.value[0])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.value[1] - p.dilation.value[1] * (p.kernel_size.value[1] - 1) - 1) / p.stride.value[1])
        tmp[tmp.length - 2] = H_out
        tmp[tmp.length - 1] = W_in
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "GRU": {
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.value
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.value = inp[inp.length-1]
        return p
      }
    },
    "GRUCell": {
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.value
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.value = inp[inp.length-1]
        return p
      }
    },
    "GroupNorm": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_channels.value = inp[1]
        return p
      }
    },
    "Hardshrink": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Identity": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "InstanceNorm1d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.value = inp[1]
        return p
      }
    },
    "InstanceNorm2d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.value = inp[1]
        return p
      }
    },
    "InstanceNorm3d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.value = inp[1]
        return p
      }
    },
    "LPPool1d": {
      outshape: (inp, p) => {
        var tmp = inp
        var L_out = 1 + (inp[inp.length - 1] + 2 * p.padding.value - p.kernel_size.value) / p.stride.value
        if (p.ceil_mode.value) {
          L_out = Math.ceil(L_out)
        } else {
          L_out = Math.floor(L_out)
        }
        tmp[tmp.length - 1] = L_out
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "LPPool2d": {
      outshape: (inp, p) => {
        var tmp = inp
        var H_out = 1 + (inp[inp.length - 2] + 2 * p.padding.value[0] - p.dilation[0] * (p.kernel_size[0] - 1) - 1) / p.stride.value[0]
        var W_out = 1 + (inp[inp.length - 1] + 2 * p.padding.value[1] - p.dilation[1] * (p.kernel_size[1] - 1) - 1) / p.stride.value[1]
        if (p.ceil_mode.value) {
          H_out = Math.ceil(H_out)
          W_out = Math.ceil(W_out)
        } else {
          H_out = Math.floor(H_out)
          W_out = Math.floor(W_out)
        }
        tmp[tmp.length - 2] = H_out
        tmp[tmp.length - 1] = W_out
        return tmp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "LSTM": {
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.value
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.value = inp[inp.length-1]
        return p
      }
    },
    "LSTMCell": {
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.value
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.value = inp[inp.length-1]
        return p
      }
    },
    "LayerNorm": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "LeakyReLU": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Linear": {
      outshape: (inp, p) => {
        var idx = 0;
        var out_shape = inp.map((val, index) => {
          idx = index;
          return val
        });
        out_shape[idx] = p.out_features.value;
        return out_shape
      },
      autoparams: (inp, p) => {
        p.in_features.value = inp[inp.length - 1]
        return p
      }
    },
    "LocalResponseNorm": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "LogSigmoid": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "LogSoftmax": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "MaxPool1d": {
      outshape: (inp, p) => {
        const L_in = inp[inp.length - 1]
        const L_out = Math.floor(1 + (L_in + 2 * p.padding.value - p.dilation.value * (p.kernel_size.value - 1) - 1) / p.stride.value)
        inp[inp.length - 1] = L_out;
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "MaxPool2d": {
      outshape: (inp, p) => {
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.value[0] - p.dilation.value[0] * (p.kernel_size.value[0] - 1) - 1) / p.stride.value[0])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.value[1] - p.dilation.value[1] * (p.kernel_size.value[1] - 1) - 1) / p.stride.value[1])
        inp[inp.length - 2] = H_out
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "MaxPool3d": {
      outshape: (inp, p) => {
        const D_in = inp[inp.length - 3]
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        const D_out = Math.floor(1 + (D_in + 2 * p.padding.value[0] - p.dilation.value[0] * (p.kernel_size.value[0] - 1) - 1) / p.stride.value[0])
        const H_out = Math.floor(1 + (H_in + 2 * p.padding.value[1] - p.dilation.value[1] * (p.kernel_size.value[1] - 1) - 1) / p.stride.value[1])
        const W_out = Math.floor(1 + (W_in + 2 * p.padding.value[2] - p.dilation.value[2] * (p.kernel_size.value[2] - 1) - 1) / p.stride.value[2])
        inp[inp.length - 3] = D_out
        inp[inp.length - 2] = H_out
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "MaxUnpool1d": {
      outshape: (inp, p) => {
        const H_in = inp[inp.length - 1]
        inp[inp.length - 1] = (H_in - 1) * p.stride.value - 2 * p.padding.value + p.kernel_size.value
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "MaxUnpool2d": {
      outshape: (inp, p) => {
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        inp[inp.length - 2] = (H_in - 1) * p.stride.value[0] - 2 * p.padding.value[0] + p.kernel_size.value[0]
        inp[inp.length - 1] = (W_in - 1) * p.stride.value[1] - 2 * p.padding.value[1] + p.kernel_size.value[1]
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "MaxUnpool3d": {
      outshape: (inp, p) => {
        const D_in = inp[inp.length - 3]
        const H_in = inp[inp.length - 2]
        const W_in = inp[inp.length - 1]
        inp[inp.length - 3] = (D_in - 1) * p.stride.value[0] - 2 * p.padding.value[0] + p.kernel_size.value[0]
        inp[inp.length - 2] = (H_in - 1) * p.stride.value[1] - 2 * p.padding.value[1] + p.kernel_size.value[1]
        inp[inp.length - 1] = (W_in - 1) * p.stride.value[2] - 2 * p.padding.value[2] + p.kernel_size.value[2]
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "PReLU": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "RNN": {
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.value
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.value = inp[inp.length-1]
        return p
      }
    },
    "RNNCell": {
      outshape: (inp, p) => {
        var idx = 0
        var out_shape = inp.map((val, index) => {
          idx = index
          return val
        })
        out_shape[idx] = p.hidden_size.value
        return out_shape
      },
      autoparams: (inp, p) => {
        p.input_size.value = inp[inp.length-1]
        return p
      }
    },
    "RReLU": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ReLU": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ReLU6": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ReflectionPad1d": {
      outshape: (inp, p) => {
        let W_out = inp[inp.length - 1] + 2 * p.padding.value
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ReflectionPad2d": {
      outshape: (inp, p) => {
        let H_out = inp[inp.length - 2] + 2 * p.padding.value[0]
        let W_out = inp[inp.length - 1] + 2 * p.padding.value[1]
        inp[inp.length - 2] = H_out
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ReplicationPad1d": {
      outshape: (inp, p) => {
        let W_out = inp[inp.length - 1] + 2 * p.padding.value
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ReplicationPad2d": {
      outshape: (inp, p) => {
        let H_out = inp[inp.length - 2] + 2 * p.padding.value[0]
        let W_out = inp[inp.length - 1] + 2 * p.padding.value[1]
        inp[inp.length - 2] = H_out
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ReplicationPad3d": {
      outshape: (inp, p) => {
        let D_out = inp[inp.length - 3] + 2 * p.padding.value[0]
        let H_out = inp[inp.length - 2] + 2 * p.padding.value[1]
        let W_out = inp[inp.length - 1] + 2 * p.padding.value[2]
        inp[inp.length - 3] = D_out
        inp[inp.length - 2] = H_out
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "SELU": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Sigmoid": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Softmax": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Softmax2d": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Softmin": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Softplus": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Softshrink": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Softsign": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "SyncBatchNorm": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        p.num_features.value = inp[1]
        return p
      }
    },
    "Tanh": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Tanhshrink": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "Threshold": {
      outshape: (inp, p) => {
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    },
    "ZeroPad2d": {
      outshape: (inp, p) => {
        const H_out = inp[inp.length - 2] + 2 * p.padding[0]
        const W_out = inp[inp.length - 1] + 2 * p.padding[1]
        inp[inp.length - 2] = H_out
        inp[inp.length - 1] = W_out
        return inp
      },
      autoparams: (inp, p) => {
        return p
      }
    }
  }
}