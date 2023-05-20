# import dependencies
import argparse
from pathlib import Path
import math
import numpy as np, matplotlib.pyplot as plt, librosa
import librosa.display
# import IPython.display
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import GRU
import json
from numpyencoder import NumpyEncoder
from sklearn import preprocessing, cluster
from streamer import Streamer

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--audio_path", type=str)
parser.add_argument("-bl", "--block_length", type=int, default=1)
parser.add_argument("-fl", "--frame_length", type=int, default=48000)
parser.add_argument("-hl", "--hop_length", type=int, default=12000)
parser.add_argument("-fls", "--frame_length_s", type=float, default=None)
parser.add_argument("-hls", "--hop_length_s", type=float, default=None)
parser.add_argument("-bpm", "--bpm", type=int, default=None)
parser.add_argument("-bt", "--beat", type=float, default=None)
parser.add_argument("-hb", "--hop_beats", type=float, default=None)
parser.add_argument("-nc", "--n_classes", type=int, default=256)
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-bs", "--batch_size", type=int, default=64)
# parser.add_argument("-ml", "--maxlen", type=int, default=256)
parser.add_argument("-s", "--step", type=int, default=2)
parser.add_argument("-hu", "--hidden_units", type=int, default=24)
parser.add_argument("-w", "--workers", type=int, default=8)
parser.add_argument("-k", "--k_fold", type=bool, default=False)
parser.add_argument("-k_val", "--k_val", type=int, default=2)
parser.add_argument("-n", "--name", type=str, default="newmodel") # model name
parser.add_argument("-d", "--directory", type=str, default=".") # directory to save model
parser.add_argument("-v", "--verbose", type=bool, default=False)

args = parser.parse_args()
if args.audio_path is None:
    print("Please specify path to audio.")
    exit()
elif not Path(args.audio_path).exists():
    print("Cannot find file: {args.audio_path}")
    exit()

audio_path = args.audio_path
block_length = args.block_length
n_classes = args.n_classes
epochs = args.epochs
batch_size = args.batch_size
# maxlen = args.maxlen
maxlen = n_classes
step = args.step
hidden_units = args.hidden_units
workers = args.workers
k_fold_on = args.k_fold
k = args.k_val
name = args.name
directory = args.directory
verbose = args.verbose

BPM = args.bpm
beat = args.beat
hop_beats = args.hop_beats
if BPM is None and beat is not None or BPM is not None and beat is None:
    print("Please specify both BPM and beat if you are using metered divisions.")
    exit()
frame_length_s = (4 * beat * 60 / BPM) if BPM is not None else args.frame_length_s
hop_length_s = (4 * hop_beats * 60 / BPM) if BPM is not None else args.hop_length_s
sr = librosa.get_samplerate(audio_path)
frame_length = math.ceil(frame_length_s * sr) if frame_length_s is not None else args.frame_length
hop_length = math.ceil(hop_length_s * sr) if hop_length_s is not None else args.hop_length
stream = Streamer(audio_path, block_length, frame_length, hop_length)

# helper function to extract features from audio block
def extract_features(y, sr):
    if y.size >= 2048:  
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, center=False) # mfccs
    else:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=len(y), hop_length=len(y), center=False)
    m_mfccs = np.median(mfccs, axis=1)
    if m_mfccs.size == 0 and verbose:
        print("Empty frame!")
    return m_mfccs
    

print(f"Audio length: {stream.length}s, {stream.n_samples} samples")
print(f"Sample rate: {sr} Hz")
print(f"Frame length: {frame_length_s}s,  {frame_length} samples")
print(f"Hop length: {hop_length_s}s, {hop_length} samples")
print(f"Block length: {block_length} frame(s)")
print(f"Number of blocks: {len(stream)}")

# extract features from each block in audio stream
features = np.array([extract_features(block, sr) for block in stream.new()])
features_scaled = preprocessing.scale(features, axis=1)
if verbose:
    # print(features[0])
    print(features.shape)
    print(features_scaled.shape)
    print(features_scaled.min(axis=0))
    print(features_scaled.max(axis=0))
    print(features_scaled[0]) # type: ignore

# cluster features
# frames = [frame for frame in frames if frame.size != 0] # remove empty
c_model = cluster.KMeans(n_clusters=n_classes, n_init='auto')
labels = c_model.fit_predict(features_scaled)

n_labels = len(np.unique(labels))
labels = labels.tolist()
if verbose:
    print(f"Total labelled frames: {len(labels)}")
    print(f"n_labels: {n_labels} (sanity check)")

frames = []
for i, block in enumerate(stream.new()):
    frames.append([i * hop_length, i * hop_length + frame_length])
# frames # sanity check

# each dict value has to be a 1D interlaved array, as Max dict object has trouble reading 2D arrays
# start sample is always stored at an even index and is followed by end sample
labelled_frames = dict()
for label, frame in zip(labels, frames):
    if label not in labelled_frames:
        labelled_frames[label] = []
    labelled_frames[label].extend(frame) # use extend instead of append
if verbose:    
    print(len(labelled_frames[0])) # sanity check
    print(labelled_frames[0][0], labelled_frames[0][1]) # sanity check

# One-hot encode
# build a subsequence for every <step> frames
# and a corresponding label that follows it
features = [] # these will be features
targets = [] # these will be targets
for i in range(0, len(labels) - maxlen, step):
    features.append(labels[i: i + maxlen])
    targets.append(labels[i + maxlen])
# one-hot encode features and targets
# adapted from wandb character generation code referenced at the beginning of this notebook
encoded_features = np.zeros((len(features), maxlen, n_labels), dtype=bool)
encoded_targets = np.zeros((len(targets), n_labels), dtype=bool)
for i, sequence in enumerate(features):
    # print(i, sequence)
    for t, label in enumerate(sequence):
        encoded_features[i, t, label] = 1
        # print(encoded_features[i, t])
    encoded_targets[i, targets[i]] = 1
# sanity check
if verbose:
    print(encoded_features.shape)
    print(encoded_targets.shape)

# adapted from code by Lukas Biewald
# https://github.com/lukas/ml-class/blob/master/projects/7-text-generation/char-gen.py
inputs = Input(shape=(maxlen, n_labels))
x = GRU(hidden_units, return_sequences=True)(inputs)
# according to DLWP [13.2.1] softmax tends to be unstable in float16
outputs = Dense(n_labels, activation='softmax', dtype="float32")(x)
model = Model(inputs=inputs, outputs=outputs)
model._name = name
model.compile(
    loss='sparse_categorical_crossentropy', # since we are using integer labels
    optimizer="adam",
    metrics=['accuracy']
    )
model.summary()

model.fit(
    encoded_features,
    encoded_targets,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    validation_split=0.2
)

path = Path(directory + "/models/" + name)
print(f"Saving model to: {path}")
path.mkdir(exist_ok=True, parents=True)
model_path = path / (name + ".keras")
model.save(model_path)
d_path = path / (name + "_frames.json")
d_path.write_text(json.dumps(labelled_frames, cls=NumpyEncoder))
config = dict()
config["filename"] = audio_path.split('/')[-1]
config["sr"] = sr
config["BPM"] = BPM
config["beat"] = beat
config["n_classes"] = n_classes
config["onset_detection"] = False
config["hop_length"] = hop_length
config["frame_length"] = frame_length
config["block_length"] = block_length
c_path = path / (name + "_config.json")    
c_path.write_text(json.dumps(config))
if verbose:
    print(f"Saved model to: {model_path}")
    print(f"Saved frames to: {d_path}")
    print(f"Saved config to: {c_path}")