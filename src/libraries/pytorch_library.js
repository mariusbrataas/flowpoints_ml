export function getPyTorchLibrary() {
  return {
    "AdaptiveAvgPool1d": {
      "extras": {
        "gothidden": false
      },
      "output_size": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveAvgPool2d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "AdaptiveAvgPool3d": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
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
      "return_indices": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "AdaptiveMaxPool3d": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
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
      "extras": {
        "gothidden": false
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
      "extras": {
        "gothidden": false
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
    "Concatenate": {
      "extras": {
        "gothidden": false
      },
      "dim": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConstantPad1d": {
      "extras": {
        "gothidden": false
      },
      "padding": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ConstantPad2d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "ConstantPad3d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "Conv1d": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
    "Flatten": {
      "extras": {
        "gothidden": false
      },
    },
    "Fold": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
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
      "extras": {
        "gothidden": true
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
      "lambd": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Hardtanh": {
      "extras": {
        "gothidden": false
      },
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
    "Identity": {
      "extras": {
        "gothidden": false
      },
    },
    "InstanceNorm1d": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
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
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LPPool2d": {
      "extras": {
        "gothidden": false
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
      "ceil_mode": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "LSTM": {
      "extras": {
        "gothidden": true
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
    "LogSigmoid": {
      "extras": {
        "gothidden": false
      },
    },
    "LogSoftmax": {
      "extras": {
        "gothidden": false
      },
      "dim": {
        "type": "int",
        "value": "None",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Max": {
      "extras": {
        "gothidden": false
      },
      "dim": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "keepdim": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MaxPool1d": {
      "extras": {
        "gothidden": false
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
      "extras": {
        "gothidden": false
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
      "extras": {
        "gothidden": false
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
      "extras": {
        "gothidden": false
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
      }
    },
    "MaxUnpool2d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "MaxUnpool3d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "Mean": {
      "extras": {
        "gothidden": false
      },
      "dim": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "keepdim": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "MultiheadAttention": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": true
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReLU6": {
      "extras": {
        "gothidden": false
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReflectionPad1d": {
      "extras": {
        "gothidden": false
      },
      "padding": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReflectionPad2d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "ReplicationPad1d": {
      "extras": {
        "gothidden": false
      },
      "padding": {
        "type": "int",
        "value": 1,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "ReplicationPad2d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "ReplicationPad3d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "SELU": {
      "extras": {
        "gothidden": false
      },
      "inplace": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Sigmoid": {
      "extras": {
        "gothidden": false
      },
    },
    "Softmax": {
      "extras": {
        "gothidden": false
      },
      "dim": {
        "type": "int",
        "value": "1",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Softmax2d": {
      "extras": {
        "gothidden": false
      },
    },
    "Softmin": {
      "extras": {
        "gothidden": false
      },
      "dim": {
        "type": "int",
        "value": "None",
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Softplus": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
      },
      "lambd": {
        "type": "float",
        "value": 0.5,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      }
    },
    "Softsign": {
      "extras": {
        "gothidden": false
      },
    },
    "SyncBatchNorm": {
      "extras": {
        "gothidden": false
      },
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
    "Tanh": {
      "extras": {
        "gothidden": false
      },
    },
    "Tanhshrink": {
      "extras": {
        "gothidden": false
      },
    },
    "Threshold": {
      "extras": {
        "gothidden": false
      },
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
      "extras": {
        "gothidden": false
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
    "ZeroPad2d": {
      "extras": {
        "gothidden": false
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
      }
    },
    "resnet18": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "resnet34": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "resnet50": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "resnet101": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "resnet152": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "vgg11": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "vgg13": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "vgg16": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "vgg19": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "alexnet": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "squeezenet1_0": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "squeezenet1_1": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "densenet121": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "densenet169": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "densenet161": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "densenet201": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "inception_v3": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "googlenet": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "shufflenet_v2_x0_5": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "shufflenet_v2_x1_0": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "shufflenet_v2_x1_5": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "shufflenet_v2_x2_0": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "mobilenet_v2": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "resnext50_32x4d": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "resnext101_32x8d": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 1000,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "segmentation.fcn_resnet50": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 21,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "segmentation.fcn_resnet101": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 21,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "segmentation.deeplabv3_resnet50": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 21,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "segmentation.deeplabv3_resnet101": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 21,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "detection.fasterrcnn_resnet50_fpn": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "pretrained_backbone": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 91,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "detection.maskrcnn_resnet50_fpn": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "pretrained_backbone": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 91,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
    "detection.keypointrcnn_resnet50_fpn": {
      "extras": {
        "gothidden": false,
        "torchvision": true
      },
      "pretrained": {
        "type": "bool",
        "value": false,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "pretrained_backbone": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "progress": {
        "type": "bool",
        "value": true,
        "istuple": false,
        "min": -Infinity,
        "max": Infinity
      },
      "num_classes": {
        "type": "int",
        "value": 91,
        "istuple": false,
        "min": 0,
        "max": Infinity
      }
    },
  }
}