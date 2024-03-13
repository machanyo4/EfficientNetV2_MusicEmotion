import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import re
from torchvision import transforms
from torchaudio.transforms import TimeMasking, FrequencyMasking

dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/"
renamed_dataset_path =  dataset_path + "/renamed/"

# ディレクトリ作成
if not (os.path.exists(dataset_path + "/spec")):
    os.mkdir(dataset_path + "/spec")
if not (os.path.exists(dataset_path + "/spec/Q1")):
    os.mkdir(dataset_path + "/spec/Q1")
if not (os.path.exists(dataset_path + "/spec/Q2")):
    os.mkdir(dataset_path + "/spec/Q2")
if not (os.path.exists(dataset_path + "/spec/Q3")):
    os.mkdir(dataset_path + "/spec/Q3")
if not (os.path.exists(dataset_path + "/spec/Q4")):
    os.mkdir(dataset_path + "/spec/Q4")
if not (os.path.exists(dataset_path + "/spec/Q1/noisy")):
    os.mkdir(dataset_path + "/spec/Q1/noisy")
if not (os.path.exists(dataset_path + "/spec/Q2/noisy")):
    os.mkdir(dataset_path + "/spec/Q2/noisy")
if not (os.path.exists(dataset_path + "/spec/Q3/noisy")):
    os.mkdir(dataset_path + "/spec/Q3/noisy")
if not (os.path.exists(dataset_path + "/spec/Q4/noisy")):
    os.mkdir(dataset_path + "/spec/Q4/noisy")

spec_dataset_path = dataset_path + "/spec"

# pyplot の詳細設定
plt.rcParams["figure.figsize"] = [3.84,3.84]    # 図の縦横のサイズ([横(inch),縦(inch)])
plt.rcParams["figure.dpi"] = 100                # dpi (dpts per inch)
plt.rcParams["figure.subplot.left"] = 0      # 余白なし
plt.rcParams["figure.subplot.bottom"] = 0
plt.rcParams["figure.subplot.right"] = 1
plt.rcParams["figure.subplot.top"] = 1

# audio_list の取得

def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio

# スペクトラム画像生成

# Add Gausian Noise to db
def add_noise_db(db, noise_level=0.005):
    noise = np.random.normal(0, noise_level, db.shape)
    noisy_db = db + noise
    return noisy_db

def make_noisy_spectrogram(audio_path, sampling, window, overlap):
    sample_freq, signal = librosa.load(audio_path, sr=sampling)  # サンプリング周波数　sampling kHzで読み込み
    sftp = librosa.stft(sample_freq, n_fft=window, hop_length=overlap)  # STFT
    strength, phase = librosa.magphase(sftp)  # 複素数を強度と位相へ変換
    db = librosa.amplitude_to_db(strength)  # 強度をdb単位へ変換
    noisy_db = add_noise_db(db)
    return noisy_db, signal

def save_spectrogram(save_path, db, signal):
        plt.figure()
        librosa.display.specshow(db, sr=signal)  # スペクトログラムを表示
        plt.savefig(save_path)
        plt.close()

# サンプリング周波数・窓サイズ・オーバラップ率の設定
sampling = 22050
window_values = [512, 1024, 2048]
overlap_rates = [rate * 0.1 for rate in range(1, 6)]

# Q1データ
Q1_filelist = path_to_audiofiles(renamed_dataset_path + "/Q1")

for file in Q1_filelist:
    filename = os.path.basename(file)                       # filename を格納
    filename_no_extension = os.path.splitext(filename)[0]   # extention を排除して格納
    noisy_path = spec_dataset_path + '/Q1/noisy/' + filename_no_extension + 'noisy'
    for window_value in window_values:
        for overlap_rate in overlap_rates:
            # パラメータ設定
            window = window_value
            overlap = int(window*overlap_rate)
            db, signal = make_noisy_spectrogram(file, sampling, window, overlap)
            # 保存するファイル名設定
            param_data = '_' + str(sampling) + '_' + str(window) + '_' + str(overlap)
            save_name = noisy_path + param_data + '.png'
            # ファイルが存在しない場合実行
            if not os.path.exists(save_name):
                save_spectrogram(save_name, db, signal)
                print("[LOG] " + filename + " -> /spec/Q1/noisy/" +  filename_no_extension + param_data + ".png")
    print("--------------------")



# Q2データ
Q2_filelist = path_to_audiofiles(renamed_dataset_path + "/Q2")

for file in Q2_filelist:
    filename = os.path.basename(file)                       # filename を格納
    filename_no_extension = os.path.splitext(filename)[0]   # extention を排除して格納
    noisy_path = spec_dataset_path + '/Q2/noisy/' + filename_no_extension + 'noisy'
    for window_value in window_values:
        for overlap_rate in overlap_rates:
            # パラメータ設定
            window = window_value
            overlap = int(window*overlap_rate)
            db, signal = make_noisy_spectrogram(file, sampling, window, overlap)
            # 保存するファイル名設定
            param_data = '_' + str(sampling) + '_' + str(window) + '_' + str(overlap)
            save_name = noisy_path + param_data + '.png'
            # ファイルが存在しない場合実行
            if not os.path.exists(save_name):
                save_spectrogram(save_name, db, signal)
                print("[LOG] " + filename + " -> /spec/Q2/noisy/" +  filename_no_extension + param_data + ".png")
    print("--------------------")

# Q3データ
Q3_filelist = path_to_audiofiles(renamed_dataset_path + "/Q3")

for file in Q3_filelist:
    filename = os.path.basename(file)                       # filename を格納
    filename_no_extension = os.path.splitext(filename)[0]   # extention を排除して格納
    noisy_path = spec_dataset_path + '/Q3/noisy/' + filename_no_extension + 'noisy'
    for window_value in window_values:
        for overlap_rate in overlap_rates:
            # パラメータ設定
            window = window_value
            overlap = int(window*overlap_rate)
            db, signal = make_noisy_spectrogram(file, sampling, window, overlap)
            # 保存するファイル名設定
            param_data = '_' + str(sampling) + '_' + str(window) + '_' + str(overlap)
            save_name = noisy_path + param_data + '.png'
            # ファイルが存在しない場合実行
            if not os.path.exists(save_name):
                save_spectrogram(save_name, db, signal)
                print("[LOG] " + filename + " -> /spec/Q3/noisy/" +  filename_no_extension + param_data + ".png")
    print("--------------------")

# Q4データ
Q4_filelist = path_to_audiofiles(renamed_dataset_path + "/Q4")

for file in Q4_filelist:
    filename = os.path.basename(file)                       # filename を格納
    filename_no_extension = os.path.splitext(filename)[0]   # extention を排除して格納
    noisy_path = spec_dataset_path + '/Q4/noisy/' + filename_no_extension + 'noisy'
    for window_value in window_values:
        for overlap_rate in overlap_rates:
            # パラメータ設定
            window = window_value
            overlap = int(window*overlap_rate)
            db, signal = make_noisy_spectrogram(file, sampling, window, overlap)
            # 保存するファイル名設定
            param_data = '_' + str(sampling) + '_' + str(window) + '_' + str(overlap)
            save_name = noisy_path + param_data + '.png'
            # ファイルが存在しない場合実行
            if not os.path.exists(save_name):
                save_spectrogram(save_name, db, signal)
                print("[LOG] " + filename + " -> /spec/Q4/noisy/" +  filename_no_extension + param_data + ".png")
    print("--------------------")



# ファイル数カウント関数
def count_file(folder_path):

  import pathlib
  initial_count = 0
  for path in pathlib.Path(folder_path).iterdir():
    if path.is_file():
      initial_count += 1

  return(initial_count)

# How many datas?    
print("[INFO] Datas in spec/Q1/noisy: ", end='')
print(count_file(spec_dataset_path + "/Q1/noisy"))
print("[INFO] Datas in spec/Q2/noisy: ", end='')
print(count_file(spec_dataset_path + "/Q2/noisy"))
print("[INFO] Datas in spec/Q3/noisy: ", end='')
print(count_file(spec_dataset_path + "/Q3/noisy"))
print("[INFO] Datas in spec/Q4/noisy: ", end='')
print(count_file(spec_dataset_path + "/Q4/noisy"))