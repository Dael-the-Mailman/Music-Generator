import os
import re
import pydub
from pydub import AudioSegment

IN_PATH = "E:/datasets/youtube/audiofiles"
OUT_PATH = "E:/datasets/youtube/wavfiles"
 
for song in os.listdir(IN_PATH):
    path_to_song = os.path.join(IN_PATH, song)
    sound = AudioSegment.from_mp3(path_to_song)
    filename = os.path.basename(path_to_song)
    filename = re.sub('.mp3', '.wav', filename)
    sound.export(os.path.join(OUT_PATH, filename), format="wav")