# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: wutenghu <wutenghu.ubc@gmail.com>
# Date:   2022-08-28

from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Dropout, \
    Bidirectional, Multiply, Permute, Lambda, RepeatVector, Flatten
from tensorflow.keras.models import Model
from keras.layers.core import K

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


# 注意力机制的另一种写法 适合上述报错使用
# 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
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

    output = Dense(1)(x)
    return Model(inputs=[inputs], outputs=output)


