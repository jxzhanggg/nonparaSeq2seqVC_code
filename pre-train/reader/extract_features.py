import librosa
import numpy as np
import glob
import os
from multiprocessing import Pool, cpu_count
import sys
    
def extract_mel_spec(filename):
    y, sample_rate = librosa.load(filename)
    mel_spec = librosa.feature.melspectrogram(y=y,
                                              sr=sample_rate,
                                              n_fft=2048,
                                              hop_length=512,
                                              win_length=None,
                                              window='hann',
                                              center=True,
                                              pad_mode='reflect',
                                              power=2.0,
                                              n_mels=128,
                                              fmin=0.0,
                                              fmax=None,
                                              htk=False,
                                              norm=1)

    np.save(file=filename.replace(".wav", ".spec"), arr=mel_spec)

def extract_phonemes(filename):
    from phonemizer.phonemize import phonemize
    from phonemizer.backend import FestivalBackend
    from phonemizer.separator import Separator
    
    with open(filename) as f:
        text=f.read()
        phones = phonemize(text,
                           language='en-us',
                           backend='festival',
                           separator=Separator(phone=' ',
                                               syllable='',
                                               word='')
        )

    with open(filename.replace(".txt", ".phones"), "w") as outfile:
        print(phones, file=outfile)

    
def extract_dir(root, kind):
    if kind =="audio":
        extraction_function=extract_mel_spec
        ext=".wav"
    elif kind =="text":
        extraction_function=extract_phonemes
        ext=".txt"
    else:
        print("ERROR")
        sys.exit(1)

    # traverse over all subdirs of the provided dir, and find
    # only files with the proper extension
    abs_paths=[]
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            abs_path = os.path.abspath(os.path.join(dirpath, f))
            if abs_path.endswith(ext):
                 abs_paths.append(abs_path)
            
    pool = Pool(cpu_count())
    pool.map(extraction_function,abs_paths)
        
if __name__ == "__main__":
    try:
        path = sys.argv[1]
        kind = sys.argv[2]
    except:
        print(
        '''
        Usage:
        
        $ extract_features.py "path" "kind"
        
        path: (str) Root path to data directory, this dir will be traversed
                    and all files matching the appropriate file extension
                    (i.e. ".txt" or ".wav") will undergo feature extraction.

        kind: (kind) Either "audio" or "text". "audio" will trigger feature
                     extraction of Mel-spectrograms, and "text" will trigger
                     phoneme extraction with a Festival backend.
        '''
        )
        sys.exit(1)

    extract_dir(path,kind)
    
