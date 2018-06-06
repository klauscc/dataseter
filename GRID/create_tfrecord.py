#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: create_tfrecord.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/06/06
#   description:
#
#================================================================

import numpy as np
import tensorflow as tf
from tf_record import feature


def _convert2example(video, label, align):
    """get the tf example given the video and label

    An example of label and align:
        label: 'sil bin blue at f two now sil'
        align: [0, 23, 29, 34, 35, 41, 47, 53]

    Args:
        video: Numpy array contains image sequence. Shape: (T, H, W, C).
        label: String. The corresponding sentence of the video.
        align: A list, each element is the begining of a word.
            The time align of each words.

    Returns:
        A tf example

    """
    return tf.train.Example(
        features=tf.train.Feature({
            'video':
            feature.bytes_feature(tf.compat.as_bytes(video.tostring())),
            'shape':
            feature.float_list_feature(video.shape),
            'label':
            feature.bytes_feature(tf.compat.as_bytes(label)),
            'align':
            feature.float_list_feature(align)
        }))

def read_example_from_path(video_path, align_path):
    """TODO: Docstring for read_example_from_path.

    Args:
        video_path (TODO): TODO
        align_path (TODO): TODO

    Returns: TODO

    """
    pass


def write_tfrecord(split_file,
                   tfrecord_save_path,
                   video_dir,
                   align_dir,
                   seed=10000):
    """write tf record file given split.

    Args:
        split_file: The txt file contains the videos to be written.
        tfrecord_save_path: The path (include filename) to save tfrecord.
        video_dir: The video dir contains the videos (extracted mouth sequence).
        align_dir: The align dir contains the align files.

    """
    video_list = [line.rstrip('\n') for line in open(split_file)]
    np.random.seed(seed)
    np.random.shuffle(video_list)
