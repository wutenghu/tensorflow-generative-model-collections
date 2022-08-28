# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: wutenghu <wutenghu@pin-dao.cn>
# Date:   2022-08-28

from keras.layers import Input, Dense, LSTM, merge, Conv1D, Dropout, \
    Bidirectional, Multiply

from keras.models import Model

from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import pandas as pd
import numpy as np

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    # output_attention_mul = merge([inputs, a_probs], name='attention_mul',
    #                              mode='mul')
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# 注意力机制的另一种写法 适合上述报错使用
# 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def attention_model(lstm_units, time_steps, input_dims):
    inputs = Input(shape=(time_steps, input_dims))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(
            LSTM(lstm_units, activation='relu', return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = attention_3d_block(x)
    x = Flatten()(x)

    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[inputs], outputs=output)


