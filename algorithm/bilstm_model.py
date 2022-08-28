# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: wutenghu <wutenghu.ubc@gmail.com>
# Date:   2022-08-28

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, \
    Bidirectional, Flatten
from tensorflow.keras.models import Model


def recurrent_model(lstm_units, time_steps, input_dims):
    inputs = Input(shape=(time_steps, input_dims))

    x = Bidirectional(
            LSTM(lstm_units, activation='relu', return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Flatten()(x)

    output = Dense(1)(x)
    return Model(inputs=[inputs], outputs=output)


