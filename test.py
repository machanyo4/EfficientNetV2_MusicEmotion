import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
import os

# Small Value
smal = 0.0001

# 可視化ファイルの作成
os.makedirs('waveimg', exist_ok=True)
os.makedirs('spectrogram', exist_ok=True)

# WAVファイルの読み込み
file_path = '/chess/project/project1/music/MER_audio_taffc_dataset_wav/Q1/MT0000040632.wav'
sample_rate, audio_data = wavfile.read(file_path)

print('sample_rate', sample_rate)
print('len_data', len(audio_data))

plt.figure()
plt.plot(audio_data)
plt.ylabel('Signal')
plt.xlabel('Times')
plt.title('Audio Signal')
plt.show()
plt.savefig('./waveimg/audio_signal.png')

# STFTパラメータの設定
nperseg = 1024  # 各セグメントのサンプル数
noverlap = 256  # セグメント間のオーバーラップ数

# STFTの計算
frequencies, times, Zxx = stft(audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
dB = 10 * np.log10(np.abs(Zxx + smal))

print('time_size : ', len(times))
print('frequencies_size : ', len(frequencies))
print('dB_size : ', len(dB))

# スペクトログラムの表示
plt.figure()
plt.pcolormesh(times, frequencies, dB, shading='auto')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.show()
plt.savefig('./test_spec_img.png')