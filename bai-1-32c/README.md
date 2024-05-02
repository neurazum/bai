# bai-1-32c

## "bai-1-32c" modeli; düşüncelerin duygusal veya davranışsal olup olmadığının ayrımını yapar.

# ------------------------------------------------------------------------------

## The "bai-1-32c" model distinguishes whether thoughts are emotional or behavioural.

[![image](https://r.resimlink.com/0VwialgtOd.png)](https://resimlink.com/0VwialgtOd)

# Kullanım / Usage

```python
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import pandas as pd

# Modeli yükle/ Load model
model = load_model('../Model/bai-1-32c.h5')


# EEG verilerini okuma fonksiyonu / EEG data reading function
def get_real_time_eeg_data():
    eeg_data = pd.read_csv("../Dataset/DEAP/featurew_raw.csv")
    eeg_data_np = eeg_data.to_numpy()
    return eeg_data_np

# Grafik oluşturma / Create graph
while True:
    eeg_data = get_real_time_eeg_data()
    if eeg_data is not None:
        prediction = model.predict(eeg_data)
        plt.figure()
        plt.plot(np.arange(len(prediction)), prediction)
        plt.xlabel('Zaman')
        plt.ylabel('Tahmin')
        plt.title('Düşünce Tahmini')
        plt.show()
```
