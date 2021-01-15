"""
Converts .mp3 files into .wav files

Arguments: 
-i, --input: Specify the folder where the .mp3 files are stored in
-o, --output: Specify the folder where the .wav files are to be stored in
"""
import os
import re
import pydub

from pydub import AudioSegment
from tqdm import tqdm

IN_PATH = "E:/datasets/youtube/audiofiles"
OUT_PATH = "E:/datasets/youtube/wavfiles"
 
for song in tqdm(os.listdir(IN_PATH)):
    path_to_song = os.path.join(IN_PATH, song)
    sound = AudioSegment.from_mp3(path_to_song)
    filename = os.path.basename(path_to_song)
    filename = re.sub('.mp3', '.wav', filename)
    sound.export(os.path.join(OUT_PATH, filename), format="wav")