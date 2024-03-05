import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

audio_path = '/chess/project/project1/music/MER_audio_taffc_dataset_wav/Q4/MT0000054705.wav'

y, sr = librosa.load(audio_path)  # サンプリング周波数　22.05kHzで読み込み

print([type(y), y.shape], [type(sr), sr])

plt.figure()
librosa.display.waveshow(y, sr=sr, color="blue")
plt.savefig("./sound_libfig.png")
plt.close()

D = librosa.stft(y)  # STFT
S, phase = librosa.magphase(D)  # 複素数を強度と位相へ変換
Sdb = librosa.amplitude_to_db(S)  # 強度をdb単位へ変換

plt.figure()
librosa.display.specshow(Sdb, sr=sr)  # スペクトログラムを表示
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1) #余白を調整
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.savefig("./spec_libfig.png")
plt.close()