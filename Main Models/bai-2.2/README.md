# bai-2.2 (164790 parametre)

## bai-2.0 ve 2.1 sürümlerinin daha hızlı ve optimize edilmiş versiyonudur. Tüm işlevleri aynıdır.

#### NOT: Gerçek zamanlı EEG veri takibi uygulamasına modeli entegre ederseniz, gerçek zamanlı olarak duygu tahmini yapabilmektedir. Uygulamaya erişebilmek için: https://github.com/neurazum/Realtime-EEG-Monitoring

## -----------------------------------------------------------------------------------

# bai-2.2 (164790 parameters)

## It is a faster and optimized version of bai-2.0 and 2.1. All functions are the same.

#### NOTE: If you integrate the model into a real-time EEG data tracking application, it can predict emotions in real time. To access the application: https://github.com/neurazum/Realtime-EEG-Monitoring


**Doğruluk/Accuracy: %94,8874296435272**

## -----------------------------------------------------------------------------------

# Kullanım / Usage

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
plt.title(f'Duygu Tahmini ({model_name}.2)')
plt.xlabel('Zaman')
plt.ylabel('Sınıf')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
model.summary()
```

# Tahmin / Prediction

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

model_path = 'model-path'

model = load_model(model_path)

scaler = StandardScaler()

predictions = model.predict(X_new_reshaped)
predicted_labels = np.argmax(predictions, axis=1)

label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
label_mapping_reverse = {v: k for k, v in label_mapping.items()}

#new_input = np.array([[23, 465, 12, 9653] * 637])
new_input = np.random.rand(1, 2548)  # 1 örnek ve 2548 özellik
new_input_scaled = scaler.fit_transform(new_input)
new_input_reshaped = new_input_scaled.reshape((new_input_scaled.shape[0], 1, new_input_scaled.shape[1]))

new_prediction = model.predict(new_input_reshaped)
predicted_label = np.argmax(new_prediction, axis=1)[0]
predicted_emotion = label_mapping_reverse[predicted_label]

if predicted_emotion == 'NEGATIVE':
    predicted_emotion = 'Negatif'
elif predicted_emotion == 'NEUTRAL':
    predicted_emotion = 'Nötr'
elif predicted_emotion == 'POSITIVE':
    predicted_emotion = 'Pozitif'

print(f'Giriş Verileri: {new_input}')
print(f'Tahmin Edilen Duygu: {predicted_emotion}')
```
