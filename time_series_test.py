# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: wutenghu <wutenghu.ubc@gmail.com>
# Date:   2022-08-27

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from algorithm.cnn_bilstm_attention_model import attention_model

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def plot_figure(date_time, df):
    plot_cols = ['return', 'month_return']
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)
    plt.show()


def plot_distribution(df, mean, std):
    std = (df - mean) / std
    std = std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()


def data_process(df, date_time):
    """

    """

    # Time feature
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day

    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()
    plot_distribution(df=df, mean=train_mean, std=train_std)

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    return train_df, val_df, test_df


def create_dataset(dataset, look_back):
    """

    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y


def normalize_mult(data):
    data = np.array(data)
    return data


def main():
    df = pd.read_csv('stk_000012_sz.csv')
    df = df[['trade_date', 'close', 'pct_chg', 'vol', 'amount']]
    df['return'] = df.apply(
        lambda row: np.log(1.0 + row['pct_chg'] / 100.0) * 100.0, axis=1)
    df['month_return'] = df['return'].rolling(window=30).sum().shift(-30)
    df = df.dropna()
    date_time = pd.to_datetime(df.pop('trade_date'), format='%Y%m%d')

    # plot_figure(date_time=date_time, df=df)

    column_indices = {name: i for i, name in enumerate(df.columns)}
    print(column_indices)

    train_df, val_df, test_df = data_process(df, date_time)
    print(train_df.columns)
    print(train_df.shape)

    INPUT_DIMS = 8
    TIME_STEPS = 5
    lstm_units = 64

    # 归一化
    X = np.array(train_df)
    y = train_df['month_return'].values.reshape(len(X), 1)

    train_X, _ = create_dataset(X, TIME_STEPS)
    _, train_Y = create_dataset(y, TIME_STEPS)

    print(train_X.shape, train_Y.shape)
    model = attention_model(
            lstm_units=lstm_units,
            time_steps=TIME_STEPS,
            input_dims=INPUT_DIMS)
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    model.fit(
            [train_X], train_Y,
            epochs=50,
            batch_size=64,
            validation_split=0.1)

    X = np.array(test_df)
    y = test_df['month_return'].values.reshape(len(X), 1)

    test_X, _ = create_dataset(X, TIME_STEPS)
    _, test_Y = create_dataset(y, TIME_STEPS)

    pred_Y = model.predict([test_X], verbose=1)
    print(pred_Y.shape)
    print(test_Y.shape)

    plt.figure(figsize=[12, 6])
    X_idx = range(len(test_Y))
    plt.plot(X_idx, test_Y, label='true value')
    plt.plot(X_idx, pred_Y, label='predict value')
    plt.legend(['true value', 'predict value'])
    plt.show()

    # m.save("./model.h5")
    # np.save("normalize.npy",normalize)


if __name__ == '__main__':
    main()
