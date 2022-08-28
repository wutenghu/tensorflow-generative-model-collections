# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: wutenghu <wutenghu@pin-dao.cn>
# Date:   2022-08-27


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Model, Input


def generate_examples(sequence, input_step, pred_stride):
    n_patterns = len(sequence) - input_step - pred_stride + 2
    X, y = list(), list()
    for i in range(n_patterns - 1):
        X.append(sequence[i:i + input_step])
        y.append(sequence[i + pred_stride:i + pred_stride + input_step])
    X_enc = np.array(X)
    X_dec = np.zeros(np.shape(X_enc))
    X_dec[:, 0:input_step - pred_stride, :] = X_enc[:, pred_stride:, :]
    y = np.array(y)
    y = y[:, :, [0, 1]]

    return X_enc, X_dec, y


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1,
                 **kwargs):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential(
                [
                    layers.Dense(feed_forward_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"att": self.att, "ffn": self.ffn,
                  "layernorm1": self.layernorm1,
                  "layernorm2": self.layernorm2, "dropout1": self.dropout1,
                  "dropout2": self.dropout2}
        base_config = super(TransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1,
                 **kwargs):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads,
                                                 key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(rate)
        self.ffn_dropout = layers.Dropout(rate)
        self.ffn = keras.Sequential(
                [
                    layers.Dense(feed_forward_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
        )

    def call(self, enc_out, target):
        target_att = self.self_att(target, target)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"layernorm1": self.layernorm1, "layernorm2": self.layernorm2,
                  "layernorm3": self.layernorm3,
                  "self_att": self.self_att, "enc_att": self.enc_att,
                  "self_dropout": self.self_dropout,
                  "enc_dropout": self.enc_dropout,
                  "ffn_dropout": self.ffn_dropout, "ffn": self.ffn}
        base_config = super(TransformerDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Define the way for positional encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        if self.embedding_dim == None:
            self.embedding_dim = int(inputs.shape[-1])

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in
             range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(
                position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(
                position_embedding[:, 1::2])  # dim 2i+1
        position_embedding = np.expand_dims(position_embedding, axis=0)
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        return position_embedding

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"sequence_len": self.sequence_len,
                  "embedding_dim": self.embedding_dim}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Define the Schedule for training
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            init_lr=0.00001,
            lr_after_warmup=0.001,
            final_lr=0.00001,
            warmup_epochs=15,
            decay_epochs=85,
            steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
                self.init_lr
                + ((self.lr_after_warmup - self.init_lr) / (
                self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
                self.final_lr,
                self.lr_after_warmup
                - (epoch - self.warmup_epochs)
                * (self.lr_after_warmup - self.final_lr)
                / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"init_lr": self.init_lr,
                  "lr_after_warmup": self.lr_after_warmup,
                  "final_lr": self.final_lr,
                  "warmup_epochs": self.warmup_epochs,
                  "decay_epochs": self.decay_epochs,
                  "steps_per_epoch": self.steps_per_epoch}

        return config


def main():
    # Generate the training data
    root_dir = "./data/"
    train_csv = "test5_denoise.csv"
    csv_file = os.path.join(root_dir, train_csv)
    df = pd.read_csv(csv_file, usecols=['latitude', 'longitude',
                                        'gyro_x', 'gyro_y', 'gyro_z',
                                        'acc_x', 'acc_y', 'acc_z'])
    data = np.array(df).astype(np.float)  # keep the precision
    mm = MinMaxScaler()
    train_data = mm.fit_transform(data)

    input_step = 30
    pred_stride = 3
    x_enc, x_dec, y_train = generate_examples(train_data, input_step,
                                              pred_stride)

    # build the model
    embed_dim = 20
    num_heads = 2
    num_feed_forward = 32
    input_shape = x_enc.shape[1:]
    print("input shape", input_shape)
    decoder_input_shape = x_dec.shape[1:]
    output_dim = 2

    # encoder
    enc_input = Input(shape=input_shape, name="encoder_inputs")
    input_pos_encoding = PositionalEncoding(input_step, embed_dim)(enc_input)
    enc_out = TransformerEncoder(embed_dim, num_heads, num_feed_forward)(
            input_pos_encoding)
    encoder = Model(enc_input, enc_out, name="encoder")

    # decoder
    decoder_inputs = Input(shape=decoder_input_shape, name="decoder_inputs")
    encoder_outputs = Input(shape=(30, embed_dim), name="encoder_outputs")
    decoder_inputs_pos_encoding = PositionalEncoding(input_step, embed_dim)(
            decoder_inputs)
    dec_out = TransformerDecoder(embed_dim, num_heads, num_feed_forward)(
            decoder_inputs_pos_encoding, encoder_outputs)
    dec_out = layers.Dropout(0.5)(dec_out)
    dec_out = layers.Dense(output_dim)(dec_out)
    decoder = Model([decoder_inputs, encoder_outputs], dec_out, name='decoder')

    # Transformer
    decoder_outputs = decoder([decoder_inputs, enc_out])
    transformer = Model([enc_input, decoder_inputs], decoder_outputs,
                        name="Transformer")

    # Train
    loss_fn = losses.MeanSquaredError()
    learning_rate = CustomSchedule(
            init_lr=0.001,
            lr_after_warmup=0.01,
            final_lr=0.0001,
            warmup_epochs=5,
            decay_epochs=15,
            steps_per_epoch=100,
    )
    optimizer = keras.optimizers.Adam(learning_rate)

    model = transformer
    keras.utils.plot_model(
            encoder,
            to_file='./model/Transformer_encoder.png',
            show_shapes=True)
    keras.utils.plot_model(
            decoder,
            to_file='./model/Transformer_decoder.png',
            show_shapes=True)
    keras.utils.plot_model(
            model,
            to_file='./model/Transformer_regression.png',
            show_shapes=True)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss_fn)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    history = model.fit(
            [x_enc, x_dec],
            y_train,
            validation_split=0.1,
            epochs=20,
            batch_size=100,
            callbacks=callbacks,)

    # Save the model
    model_saved_path = './model/Transformer_regression.h5'
    model.save(model_saved_path)
    model.save_weights('./model/Transformer_regression_weight.h5')

    # Save the model configs
    paras = {'input_step': input_step,
             'pred_stride': pred_stride,
             'embed_dim': embed_dim,
             'num_heads': num_heads,
             'num_feed_forward': num_feed_forward,
             'input_shape': input_shape,
             'decoder_input_shape': decoder_input_shape,
             'output_dim': output_dim}
    np.save('./model/Transformer_regression_weight_configs.npy', paras)

    # Plot the model's training and validation loss
    metric = "loss"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model MSE loss")
    plt.ylabel("MSE loss", fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig('TS_regression_Transformer_training_loss.png')
    plt.show()
    plt.close()


def predict():
    paras = np.load('./model/Transformer_regression_weight_configs.npy',
                    allow_pickle=True)

    # Generate the test data
    root_dir = "./data/"
    train_csv = "test5_denoise.csv"
    test_csv = 'test5_denoise_trail.csv'
    csv_file_train = os.path.join(root_dir, train_csv)
    csv_file_test = os.path.join(root_dir, test_csv)
    df_train = pd.read_csv(csv_file_train, usecols=['latitude', 'longitude',
                                                    'gyro_x', 'gyro_y',
                                                    'gyro_z',
                                                    'acc_x', 'acc_y', 'acc_z'])
    df_test = pd.read_csv(csv_file_test, usecols=['latitude', 'longitude',
                                                  'gyro_x', 'gyro_y', 'gyro_z',
                                                  'acc_x', 'acc_y', 'acc_z'])
    data_train = np.array(df_train).astype(np.float)  # keep the precision
    data_test = np.array(df_test).astype(np.float)

    mm = MinMaxScaler()
    scaled_training_data = mm.fit_transform(data_train)
    data_test = (data_test - mm.data_min_) / (mm.data_max_ - mm.data_min_)

    input_step = paras.item()['input_step']
    pred_stride = paras.item()['pred_stride']
    x_enc, x_dec, _ = generate_examples(data_test, input_step, pred_stride)

    embed_dim = paras.item()['embed_dim']
    num_heads = paras.item()['num_heads']
    num_feed_forward = paras.item()['num_feed_forward']
    input_shape = paras.item()['input_shape']
    decoder_input_shape = paras.item()['decoder_input_shape']
    output_dim = paras.item()['output_dim']

    # encoder
    enc_input = Input(shape=input_shape, name="encoder_inputs")
    input_pos_encoding = PositionalEncoding(input_step, embed_dim)(enc_input)
    enc_out = TransformerEncoder(embed_dim, num_heads, num_feed_forward)(
        input_pos_encoding)
    encoder = Model(enc_input, enc_out, name="encoder")

    # decoder
    decoder_inputs = Input(shape=decoder_input_shape, name="decoder_inputs")
    encoder_outputs = Input(shape=(30, embed_dim), name="encoder_outputs")
    decoder_inputs_pos_encoding = PositionalEncoding(input_step, embed_dim)(
        decoder_inputs)
    dec_out = TransformerDecoder(embed_dim, num_heads, num_feed_forward)(
        decoder_inputs_pos_encoding, encoder_outputs)
    dec_out = layers.Dropout(0.5)(dec_out)
    dec_out = layers.Dense(output_dim)(dec_out)
    decoder = Model([decoder_inputs, encoder_outputs], dec_out, name='decoder')

    # Transformer
    decoder_outputs = decoder([decoder_inputs, enc_out])
    model = Model([enc_input, decoder_inputs], decoder_outputs,
                  name="Transformer")
    model.load_weights('./model/Transformer_regression_weight.h5')

    # predict
    # 这里务必注意，由于需求是一个一个预测，需要指定batch_size=1,不然默认batch_size=32，每32个才给出一个预测值
    preds = model.predict([x_enc, x_dec], batch_size=1)

    preds = preds * [mm.data_max_[0:2] - mm.data_min_[0:2]] + mm.data_min_[0:2]
    preds_1 = []
    # preds_2=[]
    # preds_3=[]
    for i in range(preds.shape[0]):
        preds1 = preds[i, input_step - pred_stride, :]
        np.reshape(preds1, [1, output_dim])
        preds_1.append(preds1)

        # 多步预测提取数据的示例
        # preds2=preds[i,input_step-pred_stride+1,:]
        # np.reshape(preds2,[1,output_dim])
        # preds_2.append(preds2)
        # preds3=preds[i,input_step-pred_stride+2,:]
        # np.reshape(preds3,[1,output_dim])
        # preds_3.append(preds3)

    saved_data = pd.DataFrame(preds_1)
    saved_data.to_csv('TRIAL.csv', header=None, index=None)

    lat_pred = np.array(preds_1)[:, 0]
    lon_pred = np.array(preds_1)[:, 1]
    plt.plot(lat_pred)
    plt.savefig('lat_TRIAL.png')
    plt.figure()
    plt.plot(lon_pred)
    plt.savefig('lon_TRIAL.png')
    plt.show()


if __name__ == '__main__':
    main()
    # predict()
