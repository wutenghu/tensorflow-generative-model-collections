# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: wutenghu <wutenghu.ubc@gmail.com>
# Date:   2022-08-27

import tensorflow as tf


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
