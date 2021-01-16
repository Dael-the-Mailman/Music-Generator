"""
Splits the .wav files into training, validation, and test folders

Default values: 
- Train: 75%
- Validation: 15%
- Test: 10%

Arguments:
-h, --help:
-i, --input:
"""
import os
import sys
import shutil
import getopt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    argv = sys.argv[1:]
    PATH = None

    # Check to see if there are any invalid arguments
    try:
        opts, args = getopt.getopt(argv,"hi:",["help","input="])
    except getopt.GetoptError:
        print('train_val_test_split.py -i <inputfolder>')
        sys.exit(2)

    # Check to see if there are no arguments
    if not opts:
        print('train_val_test_split.py -i <inputfolder>')
        sys.exit(2)

    # Code for options
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('mp3towav.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ('-i', '--input'):
            PATH = arg

    # Split songs
    songs = os.listdir(PATH)
    train, split = train_test_split(songs, test_size=0.25, shuffle=True)
    valid, test = train_test_split(split, test_size=0.4, shuffle=True)
    
    # Folders
    folders = ['train', 'valid', 'test']
    for folder in folders:
        folder_path = os.path.join(PATH, folder)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        if folder == 'train':
            print("Moving training data")
            for song in tqdm(train):
                song_path = os.path.join(PATH, song)
                if not os.path.isdir(song_path):
                    shutil.move(song_path, os.path.join(folder_path, song))
        elif folder == 'valid':
            print("Moving validation data")
            for song in tqdm(valid):
                song_path = os.path.join(PATH, song)
                if not os.path.isdir(song_path):
                    shutil.move(song_path, os.path.join(folder_path, song))
        elif folder == 'test': # Redundancy
            print("Moving test data")
            for song in tqdm(test):
                song_path = os.path.join(PATH, song)
                if not os.path.isdir(song_path):
                    shutil.move(song_path, os.path.join(folder_path, song))
    

