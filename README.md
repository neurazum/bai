# **bai** Modelleri
[![image](https://r.resimlink.com/h8f9_zPND.png)](https://resimlink.com/h8f9_zPND)

bai modelleri Neurazum projesine ait bir EEG verisi okuması üzerine eğitilen yapay zekaya aittir. Bu modeller şimdilik hazır veri setlerini okumak üzere eğitilmiştir. Her modelin kendine ait farklı özellikleri vardır. Bu modelimiz sadece demodur. Projemiz hakkında bilgi edinmek isterseniz LinkedIn profilimize göz atabilirsiniz: https://www.linkedin.com/company/neurazum
(TÜRKÇESİ YAKINDA GELECEKTİR)
----------------------------------------------------------------------------------------

# bai Models

## Model Details

#### bai Models are trained to read EEG data. The data sets on which these models are trained are kept confidential by Neurazum. It is trained with deep learning methods and can work precisely on EEG with very high accuracy rates. It can work on all kinds of EEG devices regardless of the number of electrodes (Optimisation and improvements are ongoing). It aims to end the backwardness, primitiveness and error margins in the field of neuroscience.

### Model Description

- **Developed by:** _Neurazum_
- **Shared by:** _Eyüp İpler_
- **Model type:** _EEG_
- **License:** CC-BY-NC-SA-4.0

## Uses

**Our aim in these models;**

- _To analyse the person's emotion instantly,_
- _To warn dangerous patients such as epilepsy and MS early before the seizure and to take the necessary precautions,_
- _Early diagnosis for Alzheimer's patients and the bai model helps the person by memorising forgotten words,_
- _Development of mind-controlled games for players,_
- _Development of a voice assistant that can be used in everyday life._

## Direct Uses

**Classical Use:**

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
plt.title(f'Duygu Tahmini ({model_name})')
plt.xlabel('Zaman')
plt.ylabel('Sınıf')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
model.summary()
```

**Prediction Test:**

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
new_input = np.random.rand(1, 2548)  # 1 sample and 2548 features
new_input_scaled = scaler.fit_transform(new_input)
new_input_reshaped = new_input_scaled.reshape((new_input_scaled.shape[0], 1, new_input_scaled.shape[1]))

new_prediction = model.predict(new_input_reshaped)
predicted_label = np.argmax(new_prediction, axis=1)[0]
predicted_emotion = label_mapping_reverse[predicted_label]

# TR Lang
if predicted_emotion == 'NEGATIVE':
    predicted_emotion = 'Negatif'
elif predicted_emotion == 'NEUTRAL':
    predicted_emotion = 'Nötr'
elif predicted_emotion == 'POSITIVE':
    predicted_emotion = 'Pozitif'

print(f'Input Data: {new_input}')
print(f'Predicted Emotion: {predicted_emotion}')
```

**Realtime Use (EEG Monitoring without AI Model):**

```python
import sys
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


CHUNK = 1000  # Chunk size
FORMAT = pyaudio.paInt16  # Data type (16-bit PCM)
CHANNELS = 1  # (Mono)
RATE = 2000  # Sample rate (Hz)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)

    def initUI(self):
        self.setWindowTitle('EEG Monitoring by Neurazum')
        self.setWindowIcon(QIcon('/neurazumicon.ico'))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [9, 1]})
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)

        self.layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        self.x = np.arange(0, 2 * CHUNK, 2)
        self.line1, = self.ax1.plot(self.x, np.random.rand(CHUNK))
        self.line2, = self.ax2.plot(self.x, np.random.rand(CHUNK))

        self.legend_elements = [
            Line2D([0, 4], [0], color='yellow', lw=4, label='DELTA (0hz-4hz)'),
            Line2D([4, 7], [0], color='blue', lw=4, label='THETA (4hz-7hz)'),
            Line2D([8, 12], [0], color='green', lw=4, label='ALPHA (8hz-12hz)'),
            Line2D([12, 30], [0], color='red', lw=4, label='BETA (12hz-30hz)'),
            Line2D([30, 100], [0], color='purple', lw=4, label='GAMMA (30hz-100hz)')
        ]

    def update_plot(self):
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        data = np.abs(data)
        voltage_data = data * (3.3 / 1024)  # Voltage to "mV"

        self.line1.set_ydata(data)
        self.line2.set_ydata(voltage_data)

        for coll in self.ax1.collections:
            coll.remove()

        self.ax1.fill_between(self.x, data, where=((self.x >= 0) & (self.x <= 4)), color='yellow', alpha=1)
        self.ax1.fill_between(self.x, data, where=((self.x >= 4) & (self.x <= 7)), color='blue', alpha=1)
        self.ax1.fill_between(self.x, data, where=((self.x >= 8) & (self.x <= 12)), color='green', alpha=1)
        self.ax1.fill_between(self.x, data, where=((self.x >= 12) & (self.x <= 30)), color='red', alpha=1)
        self.ax1.fill_between(self.x, data, where=((self.x >= 30) & (self.x <= 100)), color='purple', alpha=1)

        self.ax1.legend(handles=self.legend_elements, loc='upper right')
        self.ax1.set_ylabel('Value (dB)')
        self.ax1.set_xlabel('Frequency (Hz)')
        self.ax1.set_title('Frequency and mV')

        self.ax2.set_ylabel('Voltage (mV)')
        self.ax2.set_xlabel('Time')

        self.canvas.draw()

    def close_application(self):
        self.timer.stop()
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
```

**Emotion Dataset Prediction Use:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

model_path = 'model-path'
new_data_path = 'dataset-path'

model = load_model(model_path)

new_data = pd.read_csv(new_data_path)

X_new = new_data.drop('label', axis=1)
y_new = new_data['label']

scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)
X_new_reshaped = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

predictions = model.predict(X_new_reshaped)
predicted_labels = np.argmax(predictions, axis=1)

label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
label_mapping_reverse = {v: k for k, v in label_mapping.items()}
actual_labels = y_new.replace(label_mapping).values

accuracy = np.mean(predicted_labels == actual_labels)

new_input = np.random.rand(2548, 2548)  # 1 sample and 2548 features
new_input_scaled = scaler.transform(new_input)
new_input_reshaped = new_input_scaled.reshape((new_input_scaled.shape[0], 1, new_input_scaled.shape[1]))

new_prediction = model.predict(new_input_reshaped)
predicted_label = np.argmax(new_prediction, axis=1)[0]
predicted_emotion = label_mapping_reverse[predicted_label]


# TR Lang
if predicted_emotion == 'NEGATIVE':
    predicted_emotion = 'Negatif'
elif predicted_emotion == 'NEUTRAL':
    predicted_emotion = 'Nötr'
elif predicted_emotion == 'POSITIVE':
    predicted_emotion = 'Pozitif'

print(f'Inputs: {new_input}')
print(f'Predicted Emotion: {predicted_emotion}')
print(f'Accuracy: %{accuracy * 100:.5f}')
```

## Bias, Risks, and Limitations

**bai Models;**

- _The biggest risk is wrong prediction :),_
- _It does not contain any restrictions in any area (for now),_
- _Data from brain signals do not contain personal information (because they are only mV values). Therefore, every guess made by bai is only a "GUESS"._

### Recommendations

- _Do not experience too many mood changes,_
- _Do not take thoughts/decisions with too many different qualities,_
- _When he/she makes a lot of mistakes, do not think that he/she gave the wrong answer (think of it as giving the correct answer),_

**Note: These items are only recommendations for better operation of the model. They do not carry any risk.**

## How to Get Started with the Model

- ```bash
  pip install -r requirements.txt
  ```
- Place the path of the model in the example uses.
- And run the file.

## Evaluation

- bai-2.0 (Accuracy very high = 97%, 93621013133208)(EMOTIONAL CLASSIFICATION) (AUTONOMOUS MODEL) (High probability of OVERFITTING)
- bai-2.1 (Accuracy very high = 97%, 93621013133208)(EMOTIONAL CLASSIFICATION) (AUTONOMOUS MODEL) (Low probability of OVERFITTING)
- bai-2.2 (Accuracy very high = 94%, 8874296435272)(EMOTIONAL CLASSIFICATION) (AUTONOMOUS MODEL) (Low probability of OVERFITTING)

### Results

[![image](https://r.resimlink.com/O7GyMoQL.png)](https://resimlink.com/O7GyMoQL)

[![image](https://r.resimlink.com/gdyCW3RP.png)](https://resimlink.com/gdyCW3RP)

[![image](https://r.resimlink.com/MpH9XS_0E.png)](https://resimlink.com/MpH9XS_0E)

[![image](https://r.resimlink.com/vsyYqJnQ4k.png)](https://resimlink.com/vsyYqJnQ4k)

#### Summary

In summary, bai models continue to be developed to learn about and predict a person's thoughts and emotions.

#### Hardware

The EEG is the only hardware!

#### Software

You can then operate this EEG device (for the time being only with audio input) with the real-time data monitoring application we have published.

GitHub: https://github.com/neurazum/Realtime-EEG-Monitoring

## More Information

LinkedIn: https://www.linkedin.com/company/neurazum

## Model Card Authors

Eyüp İpler - https://www.linkedin.com/in/eyupipler/

## Model Card Contact

neurazum@gmail.com
