# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: wutenghu <wutenghu.ubc@gmail.com>
# Date:   2022-08-27

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model


def linear_model(time_steps, input_dims):
    inputs = Input(shape=(time_steps, input_dims))
    x = Flatten()(inputs)
    output = Dense(1)(x)
    return Model(inputs=[inputs], outputs=output)
