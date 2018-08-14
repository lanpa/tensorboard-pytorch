from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import caffe2.python.brew as brew
import caffe2.python.cnn as cnn
import caffe2.python.core as core
import caffe2.python.model_helper as model_helper
import unittest

from caffe2.proto import caffe2_pb2
import tensorboardX.caffe2_graph as tb


EXPECTED_CNN = """
node {
  name: "conv1/XavierFill"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 96
          }
          dim {
            size: 3
          }
          dim {
            size: 11
          }
          dim {
            size: 11
          }
        }
      }
    }
  }
}
node {
  name: "conv1/ConstantFill"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 96
          }
        }
      }
    }
  }
}
node {
  name: "classifier/XavierFill"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1000
          }
          dim {
            size: 4096
          }
        }
      }
    }
  }
}
node {
  name: "classifier/ConstantFill"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1000
          }
        }
      }
    }
  }
}
node {
  name: "ImageInput"
  op: "ImageInput"
  input: "db"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "is_test"
    value {
      i: 0
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "NHWC2NCHW"
  op: "NHWC2NCHW"
  input: "data_nhwc"
  device: "/gpu:0"
}
node {
  name: "conv1/Conv"
  op: "Conv"
  input: "data"
  input: "conv1/conv1_w"
  input: "conv1/conv1_b"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 11
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 4
    }
  }
}
node {
  name: "conv1/Relu"
  op: "Relu"
  input: "conv1/conv1"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "conv1/MaxPool"
  op: "MaxPool"
  input: "conv1/conv1_1"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "classifier/FC"
  op: "FC"
  input: "conv1/pool1"
  input: "classifier/fc_w"
  input: "classifier/fc_b"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "classifier/Softmax"
  op: "Softmax"
  input: "classifier/fc"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "classifier/LabelCrossEntropy"
  op: "LabelCrossEntropy"
  input: "classifier/pred"
  input: "label"
  device: "/gpu:0"
}
node {
  name: "classifier/AveragedLoss"
  op: "AveragedLoss"
  input: "classifier/xent"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/ConstantFill"
  op: "ConstantFill"
  input: "classifier/loss"
  device: "/gpu:0"
  attr {
    key: "value"
    value {
      f: 1.0
    }
  }
}
node {
  name: "GRADIENTS/classifier/AveragedLossGradient"
  op: "AveragedLossGradient"
  input: "classifier/xent"
  input: "GRADIENTS/classifier/loss_autogen_grad"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/LabelCrossEntropyGradient"
  op: "LabelCrossEntropyGradient"
  input: "classifier/pred"
  input: "label"
  input: "GRADIENTS/classifier/xent_grad"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/SoftmaxGradient"
  op: "SoftmaxGradient"
  input: "classifier/pred"
  input: "GRADIENTS/classifier/pred_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/c/FCGradient"
  op: "FCGradient"
  input: "conv1/pool1"
  input: "classifier/fc_w"
  input: "GRADIENTS/classifier/fc_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "GRADIENTS/conv1/MaxPoolGradient"
  op: "MaxPoolGradient"
  input: "conv1/conv1_1"
  input: "conv1/pool1"
  input: "GRADIENTS/conv1/pool1_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "GRADIENTS/conv1/ReluGradient"
  op: "ReluGradient"
  input: "conv1/conv1_1"
  input: "GRADIENTS/conv1/conv1_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/ConvGradient"
  op: "ConvGradient"
  input: "data"
  input: "conv1/conv1_w"
  input: "GRADIENTS/conv1/conv1_grad_1"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 11
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 4
    }
  }
}
node {
  name: "GRADIENTS/NCHW2NHWC"
  op: "NCHW2NHWC"
  input: "GRADIENTS/data_grad"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_w"
  op: "Blob"
  input: "conv1/XavierFill:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc"
  op: "Blob"
  input: "classifier/FC:0"
  device: "/gpu:0"
}
node {
  name: "data_nhwc"
  op: "Blob"
  input: "ImageInput:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_b_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:1"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/pred_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/LabelCrossEntropyGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/SoftmaxGradient:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_b"
  op: "Blob"
  input: "conv1/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc_b_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:1"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc_w_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:0"
  device: "/gpu:0"
}
node {
  name: "label"
  op: "Blob"
  input: "ImageInput:1"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/data_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:2"
  device: "/gpu:0"
}
node {
  name: "classifier/loss"
  op: "Blob"
  input: "classifier/AveragedLoss:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1"
  op: "Blob"
  input: "conv1/Conv:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_grad"
  op: "Blob"
  input: "GRADIENTS/conv1/MaxPoolGradient:0"
  device: "/gpu:0"
}
node {
  name: "classifier/xent"
  op: "Blob"
  input: "classifier/LabelCrossEntropy:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/loss_autogen_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc_w"
  op: "Blob"
  input: "classifier/XavierFill:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_1"
  op: "Blob"
  input: "conv1/Relu:0"
  device: "/gpu:0"
}
node {
  name: "db"
  op: "Placeholder"
}
node {
  name: "classifier/pred"
  op: "Blob"
  input: "classifier/Softmax:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc_b"
  op: "Blob"
  input: "classifier/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/xent_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/AveragedLossGradient:0"
  device: "/gpu:0"
}
node {
  name: "data"
  op: "Blob"
  input: "NHWC2NCHW:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_w_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_grad_1"
  op: "Blob"
  input: "GRADIENTS/conv1/ReluGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/data_nhwc_grad"
  op: "Blob"
  input: "GRADIENTS/NCHW2NHWC:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/pool1_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:2"
  device: "/gpu:0"
}
node {
  name: "conv1/pool1"
  op: "Blob"
  input: "conv1/MaxPool:0"
  device: "/gpu:0"
}
"""

EXPECTED_MNIST = """
node {
  name: "conv1/XavierFill"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 20
          }
          dim {
            size: 1
          }
          dim {
            size: 5
          }
          dim {
            size: 5
          }
        }
      }
    }
  }
}
node {
  name: "conv1/ConstantFill"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 20
          }
        }
      }
    }
  }
}
node {
  name: "conv1/XavierFill_1"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 100
          }
          dim {
            size: 20
          }
          dim {
            size: 5
          }
          dim {
            size: 5
          }
        }
      }
    }
  }
}
node {
  name: "conv1/ConstantFill_1"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 100
          }
        }
      }
    }
  }
}
node {
  name: "classifier/XavierFill"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 500
          }
          dim {
            size: 1600
          }
        }
      }
    }
  }
}
node {
  name: "classifier/ConstantFill"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 500
          }
        }
      }
    }
  }
}
node {
  name: "classifier/XavierFill_1"
  op: "XavierFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 10
          }
          dim {
            size: 500
          }
        }
      }
    }
  }
}
node {
  name: "classifier/ConstantFill_1"
  op: "ConstantFill"
  device: "/gpu:0"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 10
          }
        }
      }
    }
  }
}
node {
  name: "ImageInput"
  op: "ImageInput"
  input: "db"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "is_test"
    value {
      i: 0
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "NHWC2NCHW"
  op: "NHWC2NCHW"
  input: "data_nhwc"
  device: "/gpu:0"
}
node {
  name: "conv1/Conv"
  op: "Conv"
  input: "data"
  input: "conv1/conv1_w"
  input: "conv1/conv1_b"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 5
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "conv1/MaxPool"
  op: "MaxPool"
  input: "conv1/conv1"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "conv1/Conv_1"
  op: "Conv"
  input: "conv1/pool1"
  input: "conv1/conv2_w"
  input: "conv1/conv2_b"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 5
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "conv1/MaxPool_1"
  op: "MaxPool"
  input: "conv1/conv2"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "classifier/FC"
  op: "FC"
  input: "conv1/pool2"
  input: "classifier/fc3_w"
  input: "classifier/fc3_b"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "classifier/Relu"
  op: "Relu"
  input: "classifier/fc3"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "classifier/FC_1"
  op: "FC"
  input: "classifier/fc3_1"
  input: "classifier/pred_w"
  input: "classifier/pred_b"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "classifier/Softmax"
  op: "Softmax"
  input: "classifier/pred"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "classifier/LabelCrossEntropy"
  op: "LabelCrossEntropy"
  input: "classifier/softmax"
  input: "label"
  device: "/gpu:0"
}
node {
  name: "classifier/AveragedLoss"
  op: "AveragedLoss"
  input: "classifier/xent"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/ConstantFill"
  op: "ConstantFill"
  input: "classifier/loss"
  device: "/gpu:0"
  attr {
    key: "value"
    value {
      f: 1.0
    }
  }
}
node {
  name: "GRADIENTS/classifier/AveragedLossGradient"
  op: "AveragedLossGradient"
  input: "classifier/xent"
  input: "GRADIENTS/classifier/loss_autogen_grad"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/LabelCrossEntropyGradient"
  op: "LabelCrossEntropyGradient"
  input: "classifier/softmax"
  input: "label"
  input: "GRADIENTS/classifier/xent_grad"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/SoftmaxGradient"
  op: "SoftmaxGradient"
  input: "classifier/softmax"
  input: "GRADIENTS/classifier/softmax_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/classifier/FCGradient"
  op: "FCGradient"
  input: "classifier/fc3_1"
  input: "classifier/pred_w"
  input: "GRADIENTS/classifier/pred_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "GRADIENTS/classifier/ReluGradient"
  op: "ReluGradient"
  input: "classifier/fc3_1"
  input: "GRADIENTS/classifier/fc3_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/c/FCGradient"
  op: "FCGradient"
  input: "conv1/pool2"
  input: "classifier/fc3_w"
  input: "GRADIENTS/classifier/fc3_grad_1"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "use_cudnn"
    value {
      i: 1
    }
  }
}
node {
  name: "GRADIENTS/conv1/MaxPoolGradient"
  op: "MaxPoolGradient"
  input: "conv1/conv2"
  input: "conv1/pool2"
  input: "GRADIENTS/conv1/pool2_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "GRADIENTS/conv1/ConvGradient"
  op: "ConvGradient"
  input: "conv1/pool1"
  input: "conv1/conv2_w"
  input: "GRADIENTS/conv1/conv2_grad"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 5
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/conv1/MaxPoolGradient_1"
  op: "MaxPoolGradient"
  input: "conv1/conv1"
  input: "conv1/pool1"
  input: "GRADIENTS/conv1/pool1_grad"
  device: "/gpu:0"
  attr {
    key: "cudnn_exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 2
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "stride"
    value {
      i: 2
    }
  }
}
node {
  name: "GRADIENTS/ConvGradient"
  op: "ConvGradient"
  input: "data"
  input: "conv1/conv1_w"
  input: "GRADIENTS/conv1/conv1_grad"
  device: "/gpu:0"
  attr {
    key: "exhaustive_search"
    value {
      i: 0
    }
  }
  attr {
    key: "kernel"
    value {
      i: 5
    }
  }
  attr {
    key: "order"
    value {
      s: "NCHW"
    }
  }
}
node {
  name: "GRADIENTS/NCHW2NHWC"
  op: "NCHW2NHWC"
  input: "GRADIENTS/data_grad"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc3_grad_1"
  op: "Blob"
  input: "GRADIENTS/classifier/ReluGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/xent_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/AveragedLossGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/pred_w_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/FCGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/data_nhwc_grad"
  op: "Blob"
  input: "GRADIENTS/NCHW2NHWC:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc3_w_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_grad"
  op: "Blob"
  input: "GRADIENTS/conv1/MaxPoolGradient_1:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_b_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:1"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv2_w_grad"
  op: "Blob"
  input: "GRADIENTS/conv1/ConvGradient:0"
  device: "/gpu:0"
}
node {
  name: "classifier/pred"
  op: "Blob"
  input: "classifier/FC_1:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/pool2_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:2"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv1_w_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:0"
  device: "/gpu:0"
}
node {
  name: "data"
  op: "Blob"
  input: "NHWC2NCHW:0"
  device: "/gpu:0"
}
node {
  name: "classifier/xent"
  op: "Blob"
  input: "classifier/LabelCrossEntropy:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/pool1_grad"
  op: "Blob"
  input: "GRADIENTS/conv1/ConvGradient:2"
  device: "/gpu:0"
}
node {
  name: "db"
  op: "Placeholder"
}
node {
  name: "classifier/fc3_b"
  op: "Blob"
  input: "classifier/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "classifier/pred_b"
  op: "Blob"
  input: "classifier/ConstantFill_1:0"
  device: "/gpu:0"
}
node {
  name: "classifier/softmax"
  op: "Blob"
  input: "classifier/Softmax:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/data_grad"
  op: "Blob"
  input: "GRADIENTS/ConvGradient:2"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/pred_b_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/FCGradient:1"
  device: "/gpu:0"
}
node {
  name: "label"
  op: "Blob"
  input: "ImageInput:1"
  device: "/gpu:0"
}
node {
  name: "conv1/pool1"
  op: "Blob"
  input: "conv1/MaxPool:0"
  device: "/gpu:0"
}
node {
  name: "data_nhwc"
  op: "Blob"
  input: "ImageInput:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv2"
  op: "Blob"
  input: "conv1/Conv_1:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv2_grad"
  op: "Blob"
  input: "GRADIENTS/conv1/MaxPoolGradient:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv2_b"
  op: "Blob"
  input: "conv1/ConstantFill_1:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_b"
  op: "Blob"
  input: "conv1/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc3_w"
  op: "Blob"
  input: "classifier/XavierFill:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc3_b_grad"
  op: "Blob"
  input: "GRADIENTS/c/FCGradient:1"
  device: "/gpu:0"
}
node {
  name: "classifier/pred_w"
  op: "Blob"
  input: "classifier/XavierFill_1:0"
  device: "/gpu:0"
}
node {
  name: "conv1/pool2"
  op: "Blob"
  input: "conv1/MaxPool_1:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/conv1/conv2_b_grad"
  op: "Blob"
  input: "GRADIENTS/conv1/ConvGradient:1"
  device: "/gpu:0"
}
node {
  name: "classifier/fc3_1"
  op: "Blob"
  input: "classifier/Relu:0"
  device: "/gpu:0"
}
node {
  name: "classifier/loss"
  op: "Blob"
  input: "classifier/AveragedLoss:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/fc3_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/FCGradient:2"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1_w"
  op: "Blob"
  input: "conv1/XavierFill:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv1"
  op: "Blob"
  input: "conv1/Conv:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/loss_autogen_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/ConstantFill:0"
  device: "/gpu:0"
}
node {
  name: "classifier/fc3"
  op: "Blob"
  input: "classifier/FC:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/pred_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/SoftmaxGradient:0"
  device: "/gpu:0"
}
node {
  name: "GRADIENTS/classifier/softmax_grad"
  op: "Blob"
  input: "GRADIENTS/classifier/LabelCrossEntropyGradient:0"
  device: "/gpu:0"
}
node {
  name: "conv1/conv2_w"
  op: "Blob"
  input: "conv1/XavierFill_1:0"
  device: "/gpu:0"
}
"""


class TensorboardExporterTest(unittest.TestCase):
    def test_that_operators_gets_non_colliding_names(self):
        op = caffe2_pb2.OperatorDef()
        op.type = 'foo'
        op.input.extend(['foo'])
        tb._fill_missing_operator_names([op])
        self.assertEqual(op.input[0], 'foo')
        self.assertEqual(op.name, 'foo_1')

    def test_that_replacing_colons_gives_non_colliding_names(self):
        # .. and update shapes
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo:0'
        op.input.extend(['foo:0', 'foo$0'])
        shapes = {'foo:0': [1]}
        blob_name_tracker = tb._get_blob_names([op])
        tb._replace_colons(shapes, blob_name_tracker, [op], '$')
        self.assertEqual(op.input[0], 'foo$0')
        self.assertEqual(op.input[1], 'foo$0_1')
        # Collision but blobs and op names are handled later by
        # _fill_missing_operator_names.
        self.assertEqual(op.name, 'foo$0')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['foo$0'], [1])
        self.assertEqual(len(blob_name_tracker), 2)
        self.assertEqual(blob_name_tracker['foo$0'], 'foo:0')
        self.assertEqual(blob_name_tracker['foo$0_1'], 'foo$0')

    def test_that_adding_gradient_scope_does_no_fancy_renaming(self):
        # because it cannot create collisions
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo_grad'
        op.input.extend(['foo_grad', 'foo_grad_1'])
        shapes = {'foo_grad': [1]}
        blob_name_tracker = tb._get_blob_names([op])
        tb._add_gradient_scope(shapes, blob_name_tracker, [op])
        self.assertEqual(op.input[0], 'GRADIENTS/foo_grad')
        self.assertEqual(op.input[1], 'GRADIENTS/foo_grad_1')
        self.assertEqual(op.name, 'GRADIENTS/foo_grad')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['GRADIENTS/foo_grad'], [1])
        self.assertEqual(len(blob_name_tracker), 2)
        self.assertEqual(
            blob_name_tracker['GRADIENTS/foo_grad'], 'foo_grad')
        self.assertEqual(
            blob_name_tracker['GRADIENTS/foo_grad_1'], 'foo_grad_1')

    def test_that_auto_ssa_gives_non_colliding_names(self):
        op1 = caffe2_pb2.OperatorDef()
        op1.output.extend(['foo'])
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['foo'])
        op2.output.extend(['foo'])
        op2.output.extend(['foo_1'])
        shapes = {'foo': [1], 'foo_1': [2]}
        blob_name_tracker = tb._get_blob_names([op1, op2])
        tb._convert_to_ssa(shapes, blob_name_tracker, [op1, op2])
        self.assertEqual(op1.output[0], 'foo')
        self.assertEqual(op2.input[0], 'foo')
        self.assertEqual(op2.output[0], 'foo_1')
        # Unfortunate name but we do not parse original `_` for now.
        self.assertEqual(op2.output[1], 'foo_1_1')
        self.assertEqual(len(shapes), 3)
        self.assertEqual(shapes['foo'], [1])
        self.assertEqual(shapes['foo_1'], [1])
        self.assertEqual(shapes['foo_1_1'], [2])
        self.assertEqual(len(blob_name_tracker), 3)
        self.assertEqual(blob_name_tracker['foo'], 'foo')
        self.assertEqual(blob_name_tracker['foo_1'], 'foo')
        self.assertEqual(blob_name_tracker['foo_1_1'], 'foo_1')

    def test_renaming_tensorflow_style(self):
        # Construct some dummy operators here
        # NOTE: '_w', '_bn', etc without the postfix '_' are only renamed when
        # they are at the very end of the name.
        # Test that '_w', '_w_' are renamed to '/weight', '/weight_', resp.
        op1 = caffe2_pb2.OperatorDef()
        op1.input.extend(['foo_w'])
        op1.output.extend(['foo_w_2'])
        # Test that '_bn', '_bn_' are renamed to '/batchnorm', '/batchnorm_',
        # respectively.
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['foo_bn'])
        op2.output.extend(['foo_bn_2'])
        # Test that '_b', '_b_', are renamed to '/bias', '/bias_', resp.
        op3 = caffe2_pb2.OperatorDef()
        op3.input.extend(['foo_b'])
        op3.output.extend(['foo_b_2'])
        # Test that '_s', '_s_', are renamed to '/scale', '/scale_', resp.
        op4 = caffe2_pb2.OperatorDef()
        op4.input.extend(['foo_s'])
        op4.output.extend(['foo_s_2'])
        # Test that '_sum', '_sum_', are renamed to '/sum', '/sum_', resp.
        op5 = caffe2_pb2.OperatorDef()
        op5.input.extend(['foo_sum'])
        op5.output.extend(['foo_sum_2'])
        # Test that '_branch', '_branch_', are renamed to '/branch', '/branch_',
        # respectively. Multiple inputs/outputs are also tested in this case.
        op6 = caffe2_pb2.OperatorDef()
        op6.input.extend(['foo_branch'])
        op6.input.extend(['test_branch_2'])
        op6.output.extend(['foo_branch_3'])
        op6.output.extend(['test_branch4'])
        shapes = {
            'foo_w': [1], 'foo_w_2': [2], 'foo_bn': [3], 'foo_bn_2': [4],
            'foo_b': [5], 'foo_b_2': [6], 'foo_s': [7], 'foo_s_2': [8],
            'foo_sum': [9], 'foo_sum_2': [10], 'foo_branch': [11],
            'test_branch_2': [12], 'foo_branch_3': [13], 'test_branch4': [14],
        }
        ops = [op1, op2, op3, op4, op5, op6]
        blob_name_tracker = tb._get_blob_names(ops)
        tb._rename_tensorflow_style(shapes, blob_name_tracker, ops)
        # Testing that keys in blob name tracker were renamed correctly
        self.assertEqual(blob_name_tracker['foo/weight'], 'foo_w')
        self.assertEqual(blob_name_tracker['foo/weight_2'], 'foo_w_2')
        self.assertEqual(blob_name_tracker['foo/batchnorm'], 'foo_bn')
        self.assertEqual(blob_name_tracker['foo/batchnorm_2'], 'foo_bn_2')
        self.assertEqual(blob_name_tracker['foo/bias'], 'foo_b')
        self.assertEqual(blob_name_tracker['foo/bias_2'], 'foo_b_2')
        self.assertEqual(blob_name_tracker['foo/scale'], 'foo_s')
        self.assertEqual(blob_name_tracker['foo/scale_2'], 'foo_s_2')
        self.assertEqual(blob_name_tracker['foo/sum'], 'foo_sum')
        self.assertEqual(blob_name_tracker['foo/sum_2'], 'foo_sum_2')
        self.assertEqual(blob_name_tracker['foo/branch'], 'foo_branch')
        self.assertEqual(blob_name_tracker['test/branch_2'], 'test_branch_2')
        self.assertEqual(blob_name_tracker['foo/branch_3'], 'foo_branch_3')
        self.assertEqual(blob_name_tracker['test/branch4'], 'test_branch4')
        # Testing that keys in shapes were renamed correctly
        self.assertEqual(shapes['foo/weight'], [1])
        self.assertEqual(shapes['foo/batchnorm_2'], [4])
        self.assertEqual(shapes['foo/sum'], [9])
        self.assertEqual(shapes['test/branch_2'], [12])
        # Testing that the ops were renamed correctly
        self.assertEqual(op1.input[0], 'foo/weight')
        self.assertEqual(op1.output[0], 'foo/weight_2')
        self.assertEqual(op2.input[0], 'foo/batchnorm')
        self.assertEqual(op2.output[0], 'foo/batchnorm_2')
        self.assertEqual(op3.input[0], 'foo/bias')
        self.assertEqual(op3.output[0], 'foo/bias_2')
        self.assertEqual(op4.input[0], 'foo/scale')
        self.assertEqual(op4.output[0], 'foo/scale_2')
        self.assertEqual(op5.input[0], 'foo/sum')
        self.assertEqual(op5.output[0], 'foo/sum_2')
        self.assertEqual(op6.input[0], 'foo/branch')
        self.assertEqual(op6.input[1], 'test/branch_2')
        self.assertEqual(op6.output[0], 'foo/branch_3')
        self.assertEqual(op6.output[1], 'test/branch4')

    def test_filter_ops(self):
        op1 = caffe2_pb2.OperatorDef()
        op1.input.extend(['remove_this'])
        op1.output.extend(['random_output'])
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['leave_this'])
        op2.output.extend(['leave_this_also'])
        op3 = caffe2_pb2.OperatorDef()
        op3.input.extend(['random_input'])
        op3.output.extend(['remove_this_also'])

        def filter_fn(blob):
            # Filter all blobs with names containing 'remove'
            return 'remove' not in str(blob)

        op_set1 = [op1, op2, op3]
        op_set2 = [op1, op2, op3]

        # Test case for when perform_filter = True.
        result_ops1 = tb._filter_ops(op_set1, filter_fn, True)
        new_op1, new_op2 = result_ops1[0], result_ops1[1]
        # input named 'remove_this' should have been filtered
        self.assertEqual(len(new_op1.input), 0)
        self.assertEqual(new_op1.output, ['random_output'])
        self.assertEqual(new_op2.input, ['leave_this'])
        self.assertEqual(new_op2.output, ['leave_this_also'])
        # output named 'remove_this_also' should have been filtered as well.
        # This should have also removed op3 as the filter function excludes ops
        # with no outputs.
        self.assertEqual(len(result_ops1), 2)

        # Test case for when perform_filter = False. op_set2 should remain
        # unchanged.
        result_ops2 = tb._filter_ops(op_set2, filter_fn, False)
        self.assertEqual(result_ops2, op_set2)

    # Use show_simplified=False. This shows the original style of graph
    # visualization from caffe2.contrib.tensorboard.
    # TODO: Add test for show_simplified=True.
    def test_simple_cnnmodel(self):
        model = cnn.CNNModelHelper("NCHW", name="overfeat")
        data, label = model.ImageInput(["db"], ["data", "label"], is_test=0)
        with core.NameScope("conv1"):
            conv1 = model.Conv(data, "conv1", 3, 96, 11, stride=4)
            relu1 = model.Relu(conv1, conv1)
            pool1 = model.MaxPool(relu1, "pool1", kernel=2, stride=2)
        with core.NameScope("classifier"):
            fc = model.FC(pool1, "fc", 4096, 1000)
            pred = model.Softmax(fc, "pred")
            xent = model.LabelCrossEntropy([pred, label], "xent")
            loss = model.AveragedLoss(xent, "loss")
        model.net.RunAllOnGPU()
        model.param_init_net.RunAllOnGPU()
        model.AddGradientOperators([loss], skip=1)
        blob_name_tracker = {}
        graph = tb.model_to_graph_def(
            model,
            blob_name_tracker=blob_name_tracker,
            shapes={},
            show_simplified=False,
        )
        self.assertEqual(
            blob_name_tracker['GRADIENTS/conv1/conv1_b_grad'],
            'conv1/conv1_b_grad',
        )
        self.maxDiff = None
        # We can't guarantee the order in which they appear, so we sort
        # both before we compare them
        sep = "node {"
        expected = "\n".join(sorted(
            sep + "\n  " + part.strip()
            for part in EXPECTED_CNN.strip().split(sep)
            if part.strip()
        ))
        actual = "\n".join(sorted(
            sep + "\n  " + part.strip()
            for part in str(graph).strip().split(sep)
            if part.strip()
        ))
        self.assertMultiLineEqual(actual, expected)

    # cnn.CNNModelHelper is deprecated, so we also test with
    # model_helper.ModelHelper. The model used in this test is taken from the
    # Caffe2 MNIST tutorial. Also use show_simplified=False here.
    def test_simple_model(self):
        model = model_helper.ModelHelper(name="mnist")
        data, label = brew.image_input(
            model,
            ["db"],
            ["data", "label"],
            order="NCHW",
            use_gpu_transform=False,
            is_test=0
        )
        with core.NameScope("conv1"):
            conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
            # Image size: 24 x 24 -> 12 x 12
            pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
            # Image size: 12 x 12 -> 8 x 8
            conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=100, kernel=5)
            # Image size: 8 x 8 -> 4 x 4
            pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
        with core.NameScope("classifier"):
            # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
            fc3 = brew.fc(model, pool2, 'fc3', dim_in=100 * 4 * 4, dim_out=500)
            relu = brew.relu(model, fc3, fc3)
            pred = brew.fc(model, relu, 'pred', 500, 10)
            softmax = brew.softmax(model, pred, 'softmax')
            xent = model.LabelCrossEntropy([softmax, label], 'xent')
            # compute the expected loss
            loss = model.AveragedLoss(xent, "loss")
        model.net.RunAllOnGPU()
        model.param_init_net.RunAllOnGPU()
        model.AddGradientOperators([loss], skip=1)
        blob_name_tracker = {}
        graph = tb.model_to_graph_def(
            model,
            blob_name_tracker=blob_name_tracker,
            shapes={},
            show_simplified=False,
        )
        self.assertEqual(
            blob_name_tracker['GRADIENTS/conv1/conv1_b_grad'],
            'conv1/conv1_b_grad',
        )
        self.maxDiff = None
        # We can't guarantee the order in which they appear, so we sort
        # both before we compare them
        sep = "node {"
        expected = "\n".join(sorted(
            sep + "\n  " + part.strip()
            for part in EXPECTED_MNIST.strip().split(sep)
            if part.strip()
        ))
        actual = "\n".join(sorted(
            sep + "\n  " + part.strip()
            for part in str(graph).strip().split(sep)
            if part.strip()
        ))
        self.assertMultiLineEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
