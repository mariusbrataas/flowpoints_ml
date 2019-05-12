export function getPyTorchLibrary() {
  return {
    "AdaptiveAvgPool1d": {
      "output_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveAvgPool2d": {
      "output_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveAvgPool3d": {
      "output_size": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveLogSoftmaxWithLoss": {
      "in_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "n_classes": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "cutoffs": {
        "type": "sequence",
        "value": null,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "div_value": {
        "type": "float",
        "value": 4.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "head_bias": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveMaxPool1d": {
      "output_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveMaxPool2d": {
      "output_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveMaxPool3d": {
      "output_size": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AlphaDropout": {
      "p": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AvgPool1d": {
      "kernel_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "count_include_pad": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AvgPool2d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "count_include_pad": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AvgPool3d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "count_include_pad": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "BatchNorm1d": {
      "num_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "momentum": {
        "type": "float",
        "value": 0.1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "track_running_stats": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "BatchNorm2d": {
      "num_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "momentum": {
        "type": "float",
        "value": 0.1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "track_running_stats": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "BatchNorm3d": {
      "num_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "momentum": {
        "type": "float",
        "value": 0.1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "track_running_stats": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Bilinear": {
      "in1_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "in2_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "CELU": {
      "alpha": {
        "type": "float",
        "value": 1.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConstantPad1d": {
      "padding": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConstantPad2d": {
      "padding": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConstantPad3d": {
      "padding": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Conv1d": {
      "in_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "kernel_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": 0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "padding_mode": {
        "type": "string",
        "value": "'zeros'",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "groups": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Conv2d": {
      "in_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding_mode": {
        "type": "string",
        "value": "'zeros'",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "groups": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Conv3d": {
      "in_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding_mode": {
        "type": "string",
        "value": "'zeros'",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "groups": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConvTranspose1d": {
      "in_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "kernel_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": 0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "output_padding": {
        "type": "int",
        "value": 0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "groups": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConvTranspose2d": {
      "in_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "output_padding": {
        "type": "int",
        "value": [
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "groups": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConvTranspose3d": {
      "in_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "output_padding": {
        "type": "int",
        "value": [
          0,
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "groups": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Dropout": {
      "p": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Dropout2d": {
      "p": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Dropout3d": {
      "p": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ELU": {
      "alpha": {
        "type": "float",
        "value": 1.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Embedding": {
      "num_embeddings": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "embedding_dim": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "padding_idx": {
        "type": "int",
        "value": "None",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "max_norm": {
        "type": "float",
        "value": "None",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "norm_type": {
        "type": "float",
        "value": 2.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "scale_grad_by_freq": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "sparse": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "EmbeddingBag": {
      "num_embeddings": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "embedding_dim": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "max_norm": {
        "type": "float",
        "value": "None",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "norm_type": {
        "type": "float",
        "value": 2.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "scale_grad_by_freq": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "mode": {
        "type": "string",
        "value": "'mean'",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "sparse": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Flatten": {},
    "Fold": {
      "output_size": {
        "type": "int",
        "value": [
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "kernel_size": {
        "type": "int",
        "value": [
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "FractionalMaxPool2d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "output_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "output_ratio": {
        "type": "float",
        "value": [
          0.5,
          0.5
        ],
        "istuple": true,
        "min": -Infinity,
        "max": 1
      },
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "GRU": {
      "input_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "hidden_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_layers": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "batch_first": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dropout": {
        "type": "float",
        "value": 0.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bidirectional": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "GRUCell": {
      "input_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "hidden_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "GroupNorm": {
      "num_groups": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_channels": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Hardshrink": {
      "lambd": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Hardtanh": {
      "min_val": {
        "type": "float",
        "value": -1.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "max_val": {
        "type": "float",
        "value": 1.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Identity": {},
    "InstanceNorm1d": {
      "num_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "momentum": {
        "type": "float",
        "value": 0.1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "track_running_stats": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "InstanceNorm2d": {
      "num_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "momentum": {
        "type": "float",
        "value": 0.1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "track_running_stats": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "InstanceNorm3d": {
      "num_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "momentum": {
        "type": "float",
        "value": 0.1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "track_running_stats": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LPPool1d": {
      "kernel_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LPPool2d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LSTM": {
      "input_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "hidden_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_layers": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "batch_first": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dropout": {
        "type": "float",
        "value": 0.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bidirectional": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LSTMCell": {
      "input_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "hidden_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LayerNorm": {
      "normalized_shape": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "elementwise_affine": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LeakyReLU": {
      "negative_slope": {
        "type": "float",
        "value": 0.01,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Linear": {
      "in_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "out_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LocalResponseNorm": {
      "size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "alpha": {
        "type": "float",
        "value": 0.0001,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "beta": {
        "type": "float",
        "value": 0.75,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "k": {
        "type": "float",
        "value": 1.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LogSigmoid": {},
    "LogSoftmax": {
      "dim": {
        "type": "int",
        "value": "None",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MaxPool1d": {
      "kernel_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": 0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MaxPool2d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MaxPool3d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          0,
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MaxUnpool1d": {
      "kernel_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": 0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MaxUnpool2d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MaxUnpool3d": {
      "kernel_size": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0,
          0,
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MultiheadAttention": {
      "embed_dim": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_heads": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "PReLU": {
      "num_parameters": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "init": {
        "type": "float",
        "value": 0.25,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "RNN": {
      "input_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "hidden_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_layers": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "nonlinearity": {
        "type": "select",
        "value": "tanh",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity,
        "options": [
          "relu",
          "tanh"
        ]
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "batch_first": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "dropout": {
        "type": "float",
        "value": 0.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bidirectional": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "RNNCell": {
      "input_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "hidden_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "bias": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "nonlinearity": {
        "type": "select",
        "value": "tanh",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity,
        "options": [
          "relu",
          "tanh"
        ]
      }
    },
    "RReLU": {
      "lower": {
        "type": "float",
        "value": 0.125,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "upper": {
        "type": "float",
        "value": 0.3333333333333333,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReLU": {
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReLU6": {
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReflectionPad1d": {
      "padding": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReflectionPad2d": {
      "padding": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReplicationPad1d": {
      "padding": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReplicationPad2d": {
      "padding": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReplicationPad3d": {
      "padding": {
        "type": "int",
        "value": [
          1,
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "SELU": {
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Sigmoid": {},
    "Softmax": {},
    "Softmax2d": {},
    "Softmin": {
      "dim": {
        "type": "int",
        "value": "None",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Softplus": {
      "beta": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "threshold": {
        "type": "int",
        "value": 20,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Softshrink": {
      "lambd": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Softsign": {},
    "SyncBatchNorm": {
      "num_features": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "eps": {
        "type": "float",
        "value": 1e-05,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "momentum": {
        "type": "float",
        "value": 0.1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "affine": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "track_running_stats": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Tanh": {},
    "Tanhshrink": {},
    "Threshold": {
      "threshold": {
        "type": "float",
        "value": 0.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "value": {
        "type": "float",
        "value": 1.0,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Unfold": {
      "kernel_size": {
        "type": "int",
        "value": [
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "stride": {
        "type": "int",
        "value": [
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "padding": {
        "type": "int",
        "value": [
          0
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      },
      "dilation": {
        "type": "int",
        "value": [
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ZeroPad2d": {
      "padding": {
        "type": "int",
        "value": [
          1,
          1
        ],
        "istuple": true,
        "min": -Infinity,
        "max": Infinity
      }
    }
  }
}