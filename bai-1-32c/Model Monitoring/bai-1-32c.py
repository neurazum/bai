import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import pandas as pd

# Eğitilen model
model = load_model('../../../Models/bai-1-32c.h5')


# Gerçek zamanlı olarak EEG verilerini almak için fonksiyon
def get_real_time_eeg_data():
    # EEG verilerini CSV dosyasından okuma
    eeg_data = pd.read_csv("../../Pre-Training/Datasets/DEAP/features_raw.csv")
    # Veriyi uygun formata dönüştürme, gerekirse işleme
    # Örneğin, veriyi Numpy dizisine dönüştürebilirsiniz
    eeg_data_np = eeg_data.to_numpy()
    return eeg_data_np

print()

# EEG verilerini alarak tahmin yapın ve grafik oluşturun
while True:
    eeg_data = get_real_time_eeg_data()
    if eeg_data is not None:
        # EEG verilerini modelinizle tahmin edin
        prediction = model.predict(eeg_data)

        # Tahmin sonuçlarını bir grafik üzerinde gösterin
        plt.figure()
        plt.plot(np.arange(len(prediction)), prediction)
        plt.xlabel('Zaman')
        plt.ylabel('Tahmin')
        plt.title('Düşünce Tahmini')
        plt.show()
