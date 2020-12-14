from vtrtool import * 
import os
import sys
import cv2 as cv

import tempfile
import segyio
import numpy as np

import boto3
from botocore.exceptions import ClientError
import ipywidgets as widgets
from importlib import reload
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt


def ndarray_to_segyfile(file_name, m):
    if len(m.shape) == 3:
        # XWI 3D segy models have inline and crossline swapped
        m = m.swapaxes(0, 1)
        segyio.tools.from_array3D(
            file_name,
            m,
            iline=TraceField.INLINE_3D,
            xline=TraceField.CROSSLINE_3D,
            format=SegySampleFormat.IEEE_FLOAT_4_BYTE,
            dt=1000,
        )
    elif len(m.shape) == 2:
        segyio.tools.from_array2D(
            file_name,
            m,
            iline=TraceField.INLINE_3D,
            xline=TraceField.CROSSLINE_3D,
            format=SegySampleFormat.IEEE_FLOAT_4_BYTE,
            dt=1000,
        )


def check_model(bucket, dims, m, percent_ani=False):
    """
    if m is a string, get model from file
    (else, simply check model is consistent with dimensions)
    also converts model to fractions if it's in percent
    returns the model ndarray
    """
    if isinstance(m, str):
        m = open_modelfile(bucket, m, dims)
    if np.prod(dims) != m.size:
        raise Exception(
            f"number of elements in model {m.size} does not match dims {np.prod(dims)}"
        )
    if dims[0] != m.shape[0]:
        raise Exception(
            f"first dimension of model {m.shape[0]} does not match dims {dims}"
        )
    m = m.astype(np.float32, copy=False)
    if percent_ani:
        m = m / 100.0
    return m


def open_modelfile(bucket, filename, dims=None):
    (_, model_ext) = os.path.splitext(filename)
    with tempfile.NamedTemporaryFile() as tmp:
        s3 = boto3.client("s3")
        s3.download_file(bucket, filename, tmp.name)
        if model_ext in (".sgy", ".segy"):
            if dims is None:
                raise Exception("need valid dimensions when opening SEGY models")
            return segymodel_to_vtrmodel(tmp.name, dims).arrays[0]
        elif model_ext == ".vtr":
            return vtrfile_to_ndarray(tmp.name)
        else:
            raise Exception(
                f"unrecognized file extension for model file {filename}, was expecting one of .segy/.sgy/.vtr"
            )


def write_modelfile(bucket, filename, model):
    (_, model_ext) = os.path.splitext(filename)
    with tempfile.NamedTemporaryFile() as tmp:
        if model_ext in (".sgy", ".segy"):
            ndarray_to_segyfile(tmp.name, model)
        elif model_ext == ".vtr":
            ndarrays_to_vtrfile(tmp.name, model)
        else:
            raise Exception(
                f"unrecognized file extension for model file {filename}, was expecting one of .segy/.sgy/.vtr"
            )
        contents = tmp.read()
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=filename, Body=contents)


def get_dims(m):
    if m.ndim == 3:
        (nx1, nx2, nx3) = m.shape
    elif m.ndim == 2:
        nx2 = 1
        (nx1, nx3) = m.shape
    elif m.ndim == 1:
        nx1 = 1
        nx2 = 1
        (nx3,) = m.shape
    else:
        raise Exception(f"expecting at most 3 dimensions, got {m.ndim} instead")
    return (nx1, nx2, nx3)
