# bai-2.1 (338787 parametre)

## EEG üzerinden duygu sınıflandırması yapan "bai-2.1" modeli, bir önceki model olan "bai-2.0" modeline göre overfitting ihtimali azaltılmış ve optimize edilmiş versiyonudur. Tüm işlevleri aynıdır.

#### NOT: Gerçek zamanlı EEG veri takibi uygulamasına modeli entegre ederseniz, gerçek zamanlı olarak duygu tahmini yapabilmektedir. Uygulamaya erişebilmek için: https://github.com/neurazum/Realtime-EEG-Monitoring

## -----------------------------------------------------------------------------------

# bai-2.1 (338787 parameters)

## The "bai-2.1" model, which performs emotion classification over EEG, is an optimised version of the previous model "bai-2.0" with reduced overfitting probability. All functions are the same.

#### NOTE: If you integrate the model into a real-time EEG data tracking application, it can predict emotions in real time. To access the application: https://github.com/neurazum/Realtime-EEG-Monitoring

**Doğruluk/Accuracy: %97.93621013133207**

## -----------------------------------------------------------------------------------

# Kullanım / Usage:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model_path = 'model-path'

model = load_model(model_path)

model_name = model_path.split('/')[-1].split('.')[0]

plt.figure(figsize=(10, 6))
plt.title(f'Duygu Tahmini ({model_name}.1)')
plt.xlabel('Zaman')
plt.ylabel('Sınıf')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
model.summary()

```