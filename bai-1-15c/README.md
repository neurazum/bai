# bai-1-15c

## "bai-1-15c" modeli; göz hareketlerini ve açık olma durumunun beyinde oluşturduğu sinyalleri ayıran modeldir.

### https://www.kaggle.com/datasets/shrutimechlearn/eye-movement-data-eeg-1 | https://www.kaggle.com/datasets/gauravduttakiit/neuroheadstate-eyestate-classification?select=train.csv veri setleriyle çalışır.

# ------------------------------------------------------------------------------

## The "bai-1-15c" model; is the model that separates eye movements and the signals generated in the brain by the state of being open.

### https://www.kaggle.com/datasets/shrutimechlearn/eye-movement-data-eeg-1 | https://www.kaggle.com/datasets/gauravduttakiit/neuroheadstate-eyestate-classification?select=train.csv works with datasets.

[![image](https://r.resimlink.com/cmYsjuz.png)](https://resimlink.com/cmYsjuz)

# Kullanım / Usage

```python
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import pandas as pd

# Modeli yükle / Load model
model = load_model('model/path')

def get_real_time_eeg_data():
    eeg_data1 = pd.read_csv("dataset/path")
    eeg_data2 = pd.read_csv("dataset/path")
    return eeg_data1.to_numpy(), eeg_data2.to_numpy()

while True:
    eeg_data1, eeg_data2 = get_real_time_eeg_data()
    if eeg_data1 is not None and eeg_data2 is not None:
        prediction1 = model.predict(eeg_data1)
        prediction2 = model.predict(eeg_data2)
        plt.figure()
        plt.plot(np.arange(len(prediction1)), prediction1, label='Göz Hareketleri', color='blue')
        plt.plot(np.arange(len(prediction2)), prediction2, label='Göz Durumu', color='red')
        plt.xlabel('Zaman')
        plt.ylabel('Tahmin')
        plt.title('Göz Tahmini')
        plt.legend()
        plt.show()
```