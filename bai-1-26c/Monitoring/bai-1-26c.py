import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.layers import Input, Dense
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Modeli yükle / Load model
model = load_model('model/path')

# Model ayarları / Model settings
new_input = Input(shape=(26,))
new_output = model.layers[1](new_input)

new_output = Dense(128)(new_output)
new_output = Dense(3, activation='softmax')(new_output)

new_model = tf.keras.Model(new_input, new_output)

# Veri setini yükle / Load dataset
df = pd.read_csv('dataset/path')

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

columns_drop = ['patient','time','epoch']
df.drop(columns=columns_drop, inplace=True)

eeg_data = df.iloc[:, 4:].values
eeg_data = np.reshape(eeg_data, (eeg_data.shape[0], -1))  # (None, 19)

eeg_data = np.hstack((eeg_data, np.zeros((eeg_data.shape[0], 26 - eeg_data.shape[1]))))  # (None, 26)
eeg_data = np.reshape(eeg_data, (eeg_data.shape[0], -1, 1))  # (None, 26, 1)

predictions = tf.convert_to_tensor(new_model.predict(eeg_data))
class_labels = ['BETA', 'ALFA', 'DELTA']

num_rows, num_cols = predictions.shape

colors = plt.cm.viridis(np.linspace(0, 1, num_cols))

plt.figure(figsize=(12, 6))
for col_index in range(num_cols):
    plt.plot(predictions[:, col_index], color=colors[col_index], label=class_labels[col_index])

plt.ylim(-100, 100)
plt.xlabel('Zaman')
plt.ylabel('Tahmin')
plt.title('Düşünce Tahmini')
plt.legend(title='Lejant', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
