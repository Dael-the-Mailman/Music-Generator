"""
Converts .mp3 files into .wav files

I could've made it multiprocessed but I didn't 😋
(Mainly because my computer crashes whenever I do)

Arguments:
-h, --help: Prints out help message 
-i, --input: Specify the folder where the .mp3 files are stored in
-o, --output: Specify the folder where the .wav files are to be stored in
"""
import os
import re
import sys
import getopt
import pydub

from pydub import AudioSegment
from tqdm import tqdm

if __name__ == '__main__':
    argv = sys.argv[1:]
    IN_PATH = None
    OUT_PATH = None

    # Check to see if there are any invalid arguments
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["help","input=","output="])
    except getopt.GetoptError:
        print('mp3towav.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)

    # Check to see if there are no arguments
    if not opts:
        print('mp3towav.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)
    
    # Code for options
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('mp3towav.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ('-i', '--input'):
            IN_PATH = arg
        elif opt in ('-o', '--output'):
            OUT_PATH = arg
    
    # If input and output paths are both present then run the code
    if IN_PATH and OUT_PATH:
        for song in tqdm(os.listdir(IN_PATH)):
            path_to_song = os.path.join(IN_PATH, song)
            sound = AudioSegment.from_mp3(path_to_song)
            filename = os.path.basename(path_to_song)
            filename = re.sub('.mp3', '.wav', filename)
            sound.export(os.path.join(OUT_PATH, filename), format="wav")
    else:
        print('mp3towav.py -i <inputfolder> -o <outputfolder>') 
