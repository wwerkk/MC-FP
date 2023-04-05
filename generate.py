#import dependencies
import math
import random, numpy as np, scipy, matplotlib.pyplot as plt, sklearn, librosa
import librosa.display
import IPython.display
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import GRU
import soundfile as sf
from pathlib import Path
import json
from numpyencoder import NumpyEncoder
from sklearn import preprocessing, cluster
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str)
parser.add_argument("-v", "--verbose", type=bool, default=False)
parser.add_argument("-sl", "--sequence_length", type=int, default=32)
parser.add_argument("-t", "--temperature", type=float, default=1.0)
parser.add_argument("-s", "--seed", type=int, default=-1) # -1 is random

args = parser.parse_args()
if args.path is None:
    print("Please specify path to model.")
    exit()
path = args.path
verbose = args.verbose
sequence_length = args.sequence_length
temperature = args.temperature
seed = args.seed

print(f"Path: {path}")

# Load model and necessary dictionaries
print(f"Loading model...")
if path[-1] == "/":
    path = path[:-1]
name = path.split("/")[-1]
path = path + "/"

config_path = path + f"{name}_config.json"
dict_path = path + f"{name}_frames.json"
features_path = path + f"{name}_features.json"
model_path = path + f"{name}.keras"

config = json.load(open(config_path))
audio_path = config['audio_path']
sr = config['sr']
n_classes = config['n_classes']
onset_detection = config['onset_detection']
frame_length = config['frame_length']
hop_length = config['hop_length']
block_length = config['block_length']
if verbose:
    print(f"Config loaded.")
    for key, value in config.items():
        print(f"{key}: {value}")
labelled_frames = json.load(open(dict_path))
if verbose:
    print(f"Labelled frames loaded.")
encoded_features = json.load(open(features_path))
if verbose:  
    print(f"Encoded features loaded.")
model = keras.models.load_model(model_path)
if verbose:  
    print(f"Model loaded.")
    model.summary()
print(f"All loaded successfully.")

# Function to sample from generated predictions
# adapted from wandb character generation code
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')[0]
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # print(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate output
def generate(sequence_length, temperature=1.0, seed=-1):
    if seed == -1:
        seq = np.array([random.choice(list(encoded_features))])
    else:
        seq = np.array([(encoded_features[seed])])
    print("\nGenerating with seed:", "random" if (seed == -1) else seed)
    if verbose:
        print(np.argmax(seq, axis=1))
    print("Temperature:\n", temperature)
    for i in range(sequence_length):
        preds = model.predict(seq, verbose=0)
        # print(preds)
        p_label = sample(preds[0], temperature)
        encoded_pred = np.eye(n_classes)[p_label].astype(bool)
        encoded_pred = np.array([[encoded_pred]])
        # print("\nPrediction: ", encoded_pred.astype(int))
        # print("\nPrediction: ", np.argmax(encoded_pred))
        # print(seed.shape, encoded_pred.shape)
        seq = np.concatenate((seq, encoded_pred), axis=1) # add the prediction to the sequence
        # print(seq.astype(int))
        # print("Predicted sequence:\n", seq.astype(int))
        # print("\n", seq.shape)
        # print("Predicted sequence:\n", seq.astype(int))
    return seq

if __name__ == "__main__":


    # Generate token sequence
    if seed == -1:
        seq = generate(sequence_length=sequence_length, temperature=temperature)
    else: 
        seq = generate(sequence_length=sequence_length, seed=seed, temperature=temperature)
    g_ints = np.argmax(seq[0], axis=1)
    print("Generated sequence:")
    print(g_ints)
    g_ints = [i for i in g_ints]
    print("Seems like it worked.")