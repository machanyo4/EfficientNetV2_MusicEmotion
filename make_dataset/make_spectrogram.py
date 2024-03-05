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
if not (os.path.exists(dataset_path + "/spec/unknown")):
    os.mkdir(dataset_path + "/spec/unknown")

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

def make_spectrogram(audio_path, sr=44100):
    sample_freq, signal = librosa.load(audio_path, sr=sr)  # サンプリング周波数　44.1kHzで読み込み
    sftp = librosa.stft(sample_freq)  # STFT
    strength, phase = librosa.magphase(sftp)  # 複素数を強度と位相へ変換
    db = librosa.amplitude_to_db(strength)  # 強度をdb単位へ変換
    return db, signal

# Add Gausian Noise to db
def add_noise_db(db, noise_level=0.005):
    noise = np.random.normal(0, noise_level, db.shape)
    noisy_db = db + noise
    return noisy_db

# DataAugmentation
data_transform = transforms.Compose([
    transforms.ToTensor(),  # PyTorch Tensorに変換
    TimeMasking(time_mask_param=70),  # Time Masking
    FrequencyMasking(freq_mask_param=15)  # Frequency Masking
])

# Q1データ
Q1_filelist = path_to_audiofiles(renamed_dataset_path + "/Q1")

for file in Q1_filelist:
    filename = os.path.basename(file)                       # filename を格納
    filename_no_extension = os.path.splitext(filename)[0]   # extention を排除して格納
    print("[LOG] " + filename + " -> spec/" +  filename_no_extension + "*.png")
    db, signal = make_spectrogram(file)
    # noisy_db = add_noise_db(db)
    augmented_spec = data_transform(db)
    augmented_spec = augmented_spec[0].numpy()
    # print("Original Spec Shape:", db.shape)
    # print("Noisy Spec Shape:", noisy_db.shape)
    # print("Augmented Spec Shape:", augmented_spec.shape)
    org_path = spec_dataset_path + '/Q1/' + filename_no_extension + '_org.png'
    if not os.path.exists(org_path):
        plt.figure()
        librosa.display.specshow(db, sr=signal)  # スペクトログラムを表示
        plt.savefig(org_path)
        plt.close()
    noise_path = spec_dataset_path + '/Q1/' + filename_no_extension + '_noise.png'
    if not os.path.exists(noise_path):
        plt.figure()
        librosa.display.specshow(noisy_db, sr=signal)  # スペクトログラムを表示
        plt.savefig(noise_path)
        plt.close()
    aug_path = spec_dataset_path + '/Q1/' + filename_no_extension + '_augment.png'
    if not os.path.exists(aug_path):
        plt.figure()
        librosa.display.specshow(augmented_spec, sr=signal)  # データ拡張後のスペクトログラムを表示
        plt.savefig(aug_path)
        plt.close()


# Q2データ
Q2_filelist = path_to_audiofiles(renamed_dataset_path + "/Q2")

for file in Q2_filelist:
    filename = os.path.basename(file)
    filename_no_extension = os.path.splitext(filename)[0]
    print("[LOG] " + filename + " -> spec/" +  filename_no_extension + "*.png")
    db, signal = make_spectrogram(file)
    # noisy_db = add_noise_db(db)
    augmented_spec = data_transform(db)
    augmented_spec = augmented_spec[0].numpy()
    org_path = spec_dataset_path + '/Q2/' + filename_no_extension + '_org.png'
    if not os.path.exists(org_path):
        plt.figure()
        librosa.display.specshow(db, sr=signal)  # スペクトログラムを表示
        plt.savefig(org_path)
        plt.close()
    # noise_path = spec_dataset_path + '/Q2/' + filename_no_extension + '_noise.png'
    # if not os.path.exists(noise_path):
    #     plt.figure()
    #     librosa.display.specshow(noisy_db, sr=signal)  # スペクトログラムを表示
    #     plt.savefig(noise_path)
    #     plt.close()
    aug_path = spec_dataset_path + '/Q2/' + filename_no_extension + '_augment.png'
    if not os.path.exists(aug_path):
        plt.figure()
        librosa.display.specshow(augmented_spec, sr=signal)  # データ拡張後のスペクトログラムを表示
        plt.savefig(aug_path)
        plt.close()

# Q3データ
Q3_filelist = path_to_audiofiles(renamed_dataset_path + "/Q3")

for file in Q3_filelist:
    filename = os.path.basename(file)
    filename_no_extension = os.path.splitext(filename)[0]
    print("[LOG] " + filename + " -> spec/" +  filename_no_extension + "*.png")
    db, signal = make_spectrogram(file)
    # noisy_db = add_noise_db(db)
    augmented_spec = data_transform(db)
    augmented_spec = augmented_spec[0].numpy()
    org_path = spec_dataset_path + '/Q3/' + filename_no_extension + '_org.png'
    if not os.path.exists(org_path):
        plt.figure()
        librosa.display.specshow(db, sr=signal)  # スペクトログラムを表示
        plt.savefig(org_path)
        plt.close()
    # noise_path = spec_dataset_path + '/Q3/' + filename_no_extension + '_noise.png'
    # if not os.path.exists(noise_path):
    #     plt.figure()
    #     librosa.display.specshow(noisy_db, sr=signal)  # スペクトログラムを表示
    #     plt.savefig(noise_path)
    #     plt.close()
    aug_path = spec_dataset_path + '/Q3/' + filename_no_extension + '_augment.png'
    if not os.path.exists(aug_path):
        plt.figure()
        librosa.display.specshow(augmented_spec, sr=signal)  # データ拡張後のスペクトログラムを表示
        plt.savefig(aug_path)
        plt.close()

# Q4データ
Q4_filelist = path_to_audiofiles(renamed_dataset_path + "/Q4")

for file in Q4_filelist:
    filename = os.path.basename(file)
    filename_no_extension = os.path.splitext(filename)[0]
    print("[LOG] " + filename + " -> spec/" +  filename_no_extension + "*.png")
    db, signal = make_spectrogram(file)
    # noisy_db = add_noise_db(db)
    augmented_spec = data_transform(db)
    augmented_spec = augmented_spec[0].numpy()
    org_path = spec_dataset_path + '/Q4/' + filename_no_extension + '_org.png'
    if not os.path.exists(org_path):
        plt.figure()
        librosa.display.specshow(db, sr=signal)  # スペクトログラムを表示
        plt.savefig(org_path)
        plt.close()
    # noise_path = spec_dataset_path + '/Q4/' + filename_no_extension + '_noise.png'
    # if not os.path.exists(noise_path):
    #     plt.figure()
    #     librosa.display.specshow(noisy_db, sr=signal)  # スペクトログラムを表示
    #     plt.savefig(noise_path)
    #     plt.close()
    aug_path = spec_dataset_path + '/Q4/' + filename_no_extension + '_augment.png'
    if not os.path.exists(aug_path):
        plt.figure()
        librosa.display.specshow(augmented_spec, sr=signal)  # データ拡張後のスペクトログラムを表示
        plt.savefig(aug_path)
        plt.close()

# unknown データ
unknown_filelist = path_to_audiofiles(renamed_dataset_path + "/unknown")

for file in unknown_filelist:
    filename = os.path.basename(file)
    filename_no_extension = os.path.splitext(filename)[0]
    print("[LOG] " + filename + " -> spec/unknown/" +  filename_no_extension + ".png")
    db, signal = make_spectrogram(file)
    org_path = spec_dataset_path + '/unknown/' + filename_no_extension + '.png'
    if not os.path.exists(org_path):
        plt.figure()
        librosa.display.specshow(db, sr=signal)  # スペクトログラムを表示
        plt.savefig(spec_dataset_path + '/unknown/' + filename_no_extension + '.png')
        plt.close()


# ファイル数カウント関数
def count_file(folder_path):

  import pathlib
  initial_count = 0
  for path in pathlib.Path(folder_path).iterdir():
    if path.is_file():
      initial_count += 1

  return(initial_count)

# How many datas?    
print("[INFO] Datas in spec/Q1: ", end='')
print(count_file(spec_dataset_path + "/Q1"))
print("[INFO] Datas in spec/Q2: ", end='')
print(count_file(spec_dataset_path + "/Q2"))
print("[INFO] Datas in spec/Q3: ", end='')
print(count_file(spec_dataset_path + "/Q3"))
print("[INFO] Datas in spec/Q4: ", end='')
print(count_file(spec_dataset_path + "/Q4"))

print("[INFO] Datas in spec/unknown: ", end='')
print(count_file(spec_dataset_path + "/unknown"))