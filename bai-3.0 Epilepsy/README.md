# bai-3.0 Epilepsy (45851parametre)

## "bai-3.0 Epilepsy" modeli, hastanın epilepsi nöbeti durumunu tespit eder.

#### NOT: Gerçek zamanlı EEG veri takibi uygulamasına modeli entegre ederseniz, gerçek zamanlı olarak duygu tahmini yapabilmektedir. Uygulamaya erişebilmek için: https://github.com/neurazum/Realtime-EEG-Monitoring

## -----------------------------------------------------------------------------------

# bai-3.0 Epilepsy (45851 parameters)

## The "bai-3.0 Epilepsy" model detects the patient's epileptic seizure status.

#### NOTE: If you integrate the model into a real-time EEG data tracking application, it can predict emotions in real time. To access the application: https://github.com/neurazum/Realtime-EEG-Monitoring
**Doğruluk/Accuracy: %68,90829694323143**

# Kullanım / Usage

```python
import pandas as pd
import numpy as np
import ast
from tensorflow.keras.models import load_model, Sequential
from sklearn.metrics import accuracy_score

model_path = 'model/path'

model = load_model(model_path)

test_data_path = 'epilepsy/dataset'
test_data = pd.read_csv(test_data_path)

test_data['sample'] = test_data['sample'].apply(ast.literal_eval)

X_test = np.array(test_data['sample'].tolist())
y_test = test_data['label'].values.astype(int)

timesteps = 10

X_test_reshaped = []

for i in range(len(X_test) - timesteps):
    X_test_reshaped.append(X_test[i:i + timesteps])

X_test_reshaped = np.array(X_test_reshaped)

y_pred = model.predict(X_test_reshaped)
y_pred_classes = (y_pred &gt; 0.77).astype(int) # En kararlı sonuçlar -&gt; 0.78 ve 0.77. Eşik değeri: çıkan sonucun yuvarlama değerini artırıp azaltma.
# Örn. Olasılık &lt; 0.77 ise "0", olasılık &gt;= 0.77 ise "1" tahminini yap.

accuracy = accuracy_score(y_test[timesteps:], y_pred_classes)

print("Gerçek Değerler (1: Nöbet, 0: Nöbet Değil) ve Tahminler:")
for i in range(len(y_pred_classes)):
    print(f"Gerçek: {y_test[i + timesteps]}, Tahmin: {y_pred_classes[i][0]}")
print(f"Modelin doğruluk oranı: %{accuracy * 100}")
model.summary()
```

# Python Sürümü / Python Version

### 3.9 &lt;=&gt; 3.13

# Modüller / Modules

```bash
matplotlib==3.8.0
matplotlib-inline==0.1.6
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.3.1
tensorflow==2.15.0
```