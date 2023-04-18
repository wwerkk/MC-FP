# %% [markdown]
# ## 1. Setup

# %% [markdown]
# ### 1.1 Install and import dependencies

# %%
#@title Install and import dependencies
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
from streamer import Streamer

# %% [markdown]
# ### 1.2 Load existing model (optional)

# %% [markdown]
# **Warning: if loading an existing model, skip to Generation after you do so** 

# %%
load_existing = False
 
if load_existing:
  name = "model-name"
  path = "/content/drive/MyDrive/GrainModelling/models/"

  config_path = path + f"/{name}_config.json"
  dict_path = path + f"/{name}_frames.json"
  features_path = path + f"/{name}_features.json"
  model_path = path + f"/{name}.keras"

  config = json.load(open(config_path))
  sr = config['sr']
  n_classes = config['n_classes']
  onset_detection = config['onset_detection']
  labelled_frames = json.load(open(dict_path))
  encoded_features = json.load(open(features_path))
  model = keras.models.load_model(model_path)

# %% [markdown]
# ## 3. Generation

# %% [markdown]
# ### 3.1 Generate frame sequence

# %%
sequence_length = 256 #@param {type: "integer"} no. tokens in generated sequence
temperature = 1.73 #@param {type:"slider", min:0, max:20, step:0.01}
#@markdown set to -1 for random:
seed = -1 #@param {type: "raw"}

# Sampling function
# adapted from wandb character generation code referenced at the beginning of this notebook
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
def generate(model, sequence_length, temperature=1.0, seed=-1):
    if seed == -1:
      seq = np.array([random.choice(list(encoded_features))])
    else:
      seq = np.array([(encoded_features[seed])])
    print("\nGenerating with seed:", "random" if (seed == -1) else seed, "\n", np.argmax(seq, axis=1))
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


if seed == -1:
  seq = generate(model, sequence_length=sequence_length, temperature=temperature)
else: 
  seq = generate(model, sequence_length=sequence_length, seed=seed, temperature=temperature)
g_ints = np.argmax(seq[0], axis=1)
print("Generated sequence:")
print(g_ints)
g_ints = [i for i in g_ints]
g_frames = [stream.get_frame(labelled_frames, label) for label in g_ints]
# g_frames = [frame.astype(float) for frame in g_frames]

# %% [markdown]
# ### 3.2 Resynthesise

# %%
def apply_fade(frame):
    """Apply a fade-in and fade-out to the given frame.
    frame: input frame
    length: length of fade in seconds
    """
    length = math.ceil(len(frame) * sr)
    fadein = np.arange(0, 1, 1/length)
    fadeout = 1 - fadein
    frame[:length] *= fadein
    frame[-length:] *= fadeout
    return frame

# %%
def dense_concatenate(frames, sr, density):
    """ Concatenate frames with given density.
    frames: list of frames to resynthesise from
    density: density of grain emission in seconds
    """
    # assuming each frame has the same length
    frames = frames.astype(float)
    density = math.ceil(density * sr) # convert to samples
    out = np.zeros(len(frames) * density + len(frames[-1])) # initialize output signal
    start = 0

    for frame in frames[1:]:
        start += density
        end = start + len(frame)
        if end > len(out):
            # print(start, end, len(out))
            out = np.pad(out, (0, end - len(out)), mode='constant')
            print(f"Strange... Frame is too long to be added to output. Padding output array to length {len(out)}")
        out[start:end] += np.array(frame) # add grain to output signal

    out = out / np.max(out) # normalize
    return out # return the output signal

# %%
#@title Resynthesise
fade_length = 0.01 #@param {type: "number"}
#@markdown for emitter mode:
BPM = 140 #@param {type: "number"} BPM
beat_density = 1/64 #@param {type: "number"} grain emission rate in fractions of a beat
#@markdown for overlap mode:

density = (4 * beat_density * 60 / BPM)
print(f"Grain density: {density} seconds")
# Apply a Hann window to each frame
g_frames = np.array([frame * scipy.signal.hann(frame_length) for frame in g_frames], dtype=object)

out = dense_concatenate(g_frames, sr, density) # concatenate the grains
librosa.display.waveshow(out, sr=sr) # display the waveform
play = IPython.display.Audio(out, rate=sr) # play audio file using ipython display
IPython.display.display(play)

# %% [markdown]
# ### 3.3 Save audio

# %%
#@title Save audio
if SAVE:
    filename = name #@param {type: "string"}
    filename += f"-generation-{sequence_length}-tokens-{beat_density}n-{BPM}-BPM.wav"
    output_path = "/Users/wwerkowicz/GS/MC/MC-FP/MC-FP-master/output" #@param {type:"string"}
    output_path += ("/" + filename)
    print("Saving", output_path)
    sf.write(output_path, out, sr) # write audio file to disk
