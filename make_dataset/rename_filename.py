import os
import shutil
import re

# データセットの場所

dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/"

# ファイル数カウント関数
def count_file(folder_path):

  import pathlib
  initial_count = 0
  for path in pathlib.Path(folder_path).iterdir():
    if path.is_file():
      initial_count += 1

  return(initial_count)

# audio_list の取得

def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio

# ディレクトリ作成
if not (os.path.exists(dataset_path + "/renamed")):
    os.mkdir(dataset_path + "/renamed")
if not (os.path.exists(dataset_path + "/renamed/Q1")):
    os.mkdir(dataset_path + "/renamed/Q1")
if not (os.path.exists(dataset_path + "/renamed/Q2")):
    os.mkdir(dataset_path + "/renamed/Q2")
if not (os.path.exists(dataset_path + "/renamed/Q3")):
    os.mkdir(dataset_path + "/renamed/Q3")
if not (os.path.exists(dataset_path + "/renamed/Q4")):
    os.mkdir(dataset_path + "/renamed/Q4")

# Q1データ
Q1_filelist = path_to_audiofiles(dataset_path + "/Q1")

for file in Q1_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q1/"+genre +"." + songname)


# shutil.copyfile("file", "./test2.txt")
# Q2データ
Q2_filelist = path_to_audiofiles(dataset_path + "/Q2")

for file in Q2_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q2/"+genre +"." + songname)

# Q3データ
Q3_filelist = path_to_audiofiles(dataset_path + "/Q3")

for file in Q3_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q3/"+genre +"." + songname)

# Q4データ
Q4_filelist = path_to_audiofiles(dataset_path + "/Q4")

for file in Q4_filelist:
  songname = re.split("[/]", file) # ["/"] でスライス
  genre = songname[len(songname)-2] #　ジャンルを格納
  songname = songname[len(songname)-1] # 曲名部分を格納
  print("[LOG] " + genre + "/" + songname + " -> renamed/" +  genre +"." + songname)
  shutil.copyfile(file, dataset_path +"/renamed/Q4/"+genre +"." + songname)


# unkown データの作成

unknown_100_dir = dataset_path + "/renamed/unknown"
if not os.path.exists(unknown_100_dir):
  os.mkdir(unknown_100_dir)

Q1_tmp = dataset_path + "/renamed/Q1/"
Q2_tmp = dataset_path + "/renamed/Q2/"
Q3_tmp = dataset_path + "/renamed/Q3/"
Q4_tmp = dataset_path + "/renamed/Q4/"

Q1_unknown = [
"Q1.MT0011869625.wav","Q1.MT0012041920.wav","Q1.MT0013621344.wav",
"Q1.MT0013914319.wav","Q1.MT0014576739.wav","Q1.MT0015005100.wav","Q1.MT0006096934.wav",
"Q1.MT0004428604.wav","Q1.MT0003570082.wav","Q1.MT0008575372.wav","Q1.MT0032892262.wav",
"Q1.MT0001898830.wav","Q1.MT0033415296.wav","Q1.MT0005674518.wav","Q1.MT0011957429.wav",
"Q1.MT0003949060.wav","Q1.MT0033397838.wav","Q1.MT0002532237.wav","Q1.MT0015934550.wav",
"Q1.MT0001053268.wav","Q1.MT0005713768.wav","Q1.MT0017797643.wav","Q1.MT0028220133.wav",
"Q1.MT0009897495.wav","Q1.MT0007067293.wav"]

Q2_unknown = ["Q2.MT0002385077.wav",
"Q2.MT0012168286.wav","Q2.MT0014703649.wav","Q2.MT0027256574.wav","Q2.MT0010736208.wav",
"Q2.MT0030160582.wav","Q2.MT0010624346.wav","Q2.MT0004028719.wav","Q2.MT0006520567.wav",
"Q2.MT0006810990.wav","Q2.MT0012865192.wav","Q2.MT0004645468.wav","Q2.MT0026963572.wav",
"Q2.MT0004068560.wav","Q2.MT0010415730.wav","Q2.MT0011348776.wav","Q2.MT0011697297.wav",
"Q2.MT0009348908.wav","Q2.MT0033084992.wav","Q2.MT0014838459.wav","Q2.MT0026520343.wav",
"Q2.MT0030271679.wav","Q2.MT0004316859.wav","Q2.MT0015962332.wav","Q2.MT0026898936.wav"]

Q3_unknown = ["Q3.MT0005171805.wav",
"Q3.MT0014050974.wav","Q3.MT0015665796.wav","Q3.MT0031693432.wav","Q3.MT0033317646.wav",
"Q3.MT0001871732.wav","Q3.MT0035316286.wav","Q3.MT0003129858.wav","Q3.MT0004082588.wav",
"Q3.MT0006397809.wav","Q3.MT0007652281.wav","Q3.MT0010375510.wav","Q3.MT0007438571.wav",
"Q3.MT0033177286.wav","Q3.MT0011066228.wav","Q3.MT0007583962.wav","Q3.MT0012317309.wav",
"Q3.MT0001844620.wav","Q3.MT0000299291.wav","Q3.MT0000742898.wav","Q3.MT0000661250.wav",
"Q3.MT0026690204.wav","Q3.MT0011667212.wav","Q3.MT0011230792.wav","Q3.MT0002126934.wav"]
              
Q4_unknown = ["Q4.MT0000980148.wav",
"Q4.MT0008167476.wav","Q4.MT0010979481.wav","Q4.MT0011051663.wav","Q4.MT0012893353.wav",
"Q4.MT0013280170.wav","Q4.MT0013955066.wav","Q4.MT0014134790.wav","Q4.MT0028813196.wav",
"Q4.MT0003280103.wav","Q4.MT0013486354.wav","Q4.MT0003710207.wav","Q4.MT0009769814.wav",
"Q4.MT0009075362.wav","Q4.MT0000636335.wav","Q4.MT0013176970.wav","Q4.MT0026776967.wav",
"Q4.MT0002834532.wav","Q4.MT0004093767.wav","Q4.MT0012222183.wav","Q4.MT0009741521.wav",
"Q4.MT0028374210.wav","Q4.MT0004036096.wav","Q4.MT0002674708.wav","Q4.MT0005550441.wav"]


# データの移動

for song in Q1_unknown:
  file = Q1_tmp + song
  try:
    shutil.move(file, unknown_100_dir)
  except Exception:
    print("[ERROR] File: " + song + " seems like already exists in unknown_dir.")
    os.remove(file)
for song in Q2_unknown:
  file = Q2_tmp + song
  try:
    shutil.move(file, unknown_100_dir)
  except Exception:
    print("[ERROR] File: " + song + " seems like already exists in unknown_dir.")
    os.remove(file)
for song in Q3_unknown:
  file = Q3_tmp + song
  try:
    shutil.move(file, unknown_100_dir)
  except Exception:
    print("[ERROR] File: " + song + " seems like already exists in unknown_dir.")
    os.remove(file)
for song in Q4_unknown:
  file = Q4_tmp + song
  try:
    shutil.move(file, unknown_100_dir)
  except Exception:
    print("[ERROR] File: " + song + " seems like already exists in unknown_dir.")
    os.remove(file)


# How many datas?    
print("[INFO] Datas in renamed/Q1: ", end='')
print(count_file(dataset_path + "/renamed/Q1"))
print("[INFO] Datas in renamed/Q2: ", end='')
print(count_file(dataset_path + "/renamed/Q2"))
print("[INFO] Datas in renamed/Q3: ", end='')
print(count_file(dataset_path + "/renamed/Q3"))
print("[INFO] Datas in renamed/Q4: ", end='')
print(count_file(dataset_path + "/renamed/Q4"))

print("[INFO] Datas in renamed/unknown: ", end='')
print(count_file(dataset_path + "/renamed/unknown"))