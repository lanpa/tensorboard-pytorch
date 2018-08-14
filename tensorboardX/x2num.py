# DO NOT alter/distruct/free input object !
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six


def make_np(x, modality=None):
    if isinstance(x, np.ndarray):
        return prepare_numpy(x, modality)
    if isinstance(
            x, six.string_types):  # Caffe2 will pass name of blob(s) to fetch
        return prepare_caffe2(x, modality)
    if np.isscalar(x):
        return np.array([x])
    if 'torch' in str(type(x)):
        return prepare_pytorch(x, modality)
    if 'chainer' in str(type(x)):
        return prepare_chainer(x, modality)
    if 'mxnet' in str(type(x)):
        return prepare_mxnet(x, modality)


def prepare_numpy(x, modality):
    if modality == 'IMG':
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        x = _prepare_image(x)
    if modality == 'VID':
        x = _prepare_video(x)
    return x


def prepare_pytorch(x, modality):
    import torch
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    if modality == 'IMG':
        x = _prepare_image(x)
    if modality == 'VID':
        x = _prepare_video(x)
    return x


def prepare_theano(x):
    import theano
    pass


def prepare_caffe2(x, modality):
    try:
        from caffe2.python import workspace
    except ImportError:
        # TODO (ml7): Remove try-except when PyTorch 1.0 merges PyTorch and Caffe2
        # Caffe2 is not installed, disabling Caffe2 functionality
        pass
    x = workspace.FetchBlob(x)
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def prepare_mxnet(x, modality):
    x = x.asnumpy()
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def prepare_chainer(x, modality):
    import chainer
    x = chainer.cuda.to_cpu(x.data)
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def make_grid(I, ncols=8):
    assert isinstance(
        I, np.ndarray), 'plugin error, should pass numpy array here'
    assert I.ndim == 4 and I.shape[1] == 3
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((3, H * nrows, W * ncols))
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H:(y + 1) * H, x * W:(x + 1) * W] = I[i]
            i = i + 1
    return canvas


def _prepare_image(I):
    # convert [N]CHW image to HWC
    assert isinstance(
        I, np.ndarray), 'plugin error, should pass numpy array here'
    assert I.ndim == 2 or I.ndim == 3 or I.ndim == 4
    if I.ndim == 4:  # NCHW
        if I.shape[1] == 1:  # N1HW
            I = np.concatenate((I, I, I), 1)  # N3HW
        assert I.shape[1] == 3
        I = make_grid(I)  # 3xHxW
    if I.ndim == 3 and I.shape[0] == 1:  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
    if I.ndim == 2:  # HxW
        I = np.expand_dims(I, 0)  # 1xHxW
        I = np.concatenate((I, I, I), 0)  # 3xHxW
    I = I.transpose(1, 2, 0)

    return I


def _prepare_video(V):
    b, c, t, h, w = V.shape

    if V.dtype == np.uint8:
        V = np.float32(V) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    # pad to nearest power of 2, all at once
    if not is_power2(V.shape[0]):
        len_addition = int(2**V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate(
            (V,
             np.zeros(
                 shape=(
                     len_addition,
                     c,
                     t,
                     h,
                     w))),
            axis=0)

    b = V.shape[0]
    n_rows = 2**((b.bit_length() - 1) // 2)
    n_cols = b // n_rows

    V = np.reshape(V, newshape=(n_rows, n_cols, c, t, h, w))
    V = np.transpose(V, axes=(3, 0, 4, 1, 5, 2))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))

    return V
