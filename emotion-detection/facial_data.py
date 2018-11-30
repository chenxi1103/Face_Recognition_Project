import pandas as pd
import numpy as np

TRAIN_DATA_PATH = "all/fer2013/fer2013.csv"


def read():
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    emotion_col = np.array(raw_df['emotion'])
    pixels_col = np.array(raw_df['pixels'])
    samples = len(emotion_col)

    labels = np.zeros((samples, 7), dtype=int)
    data = np.zeros((samples, 48*48))

    for i in range(samples):
        array = np.fromstring(pixels_col[i], sep=' ')
        array = array/(array.max() + 0.001)  # normalize pixels
        data[i] = array
        labels[i, emotion_col[i]] = 1  # hot spot

    data_test = data[30000:35000]
    labels_test = labels[30000:35000]

    data_train = data[0:30000]
    labels_train = labels[0:30000]

    print("Data is ready.")

    return data_test, labels_test, data_train, labels_train













