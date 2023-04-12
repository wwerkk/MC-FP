#import dependencies
import random, numpy as np
from keras.models import load_model
import json
import argparse

import argparse
import math

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str)
parser.add_argument("-v", "--verbose", type=bool, default=False)

args = parser.parse_args()
if args.path is None:
    print("Please specify path to model.")
    exit()
path = args.path
verbose = args.verbose

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
# encoded_features = json.load(open(features_path))
# if verbose:  
    # print(f"Encoded features loaded.")
model = load_model(model_path)
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
def generate(sequence_length=4, temperature=1.0, prompt=[]):
    if temperature <= 0:
        temperature = 0.01
    if len(prompt) > 0:
        encoded_prompt = []
        for label in prompt:
            encoded_label = np.eye(n_classes)[label].astype(bool).tolist() # this doesn't actually encode the prompt
            encoded_prompt.extend([encoded_label])
            # print("Encoded prompt shape: ", np.array(encoded_prompt).shape)
            # print("Encoded prompt: ", encoded_prompt)
        seq = np.array(encoded_prompt)
        if verbose:
            print("Encoded prompt shape: ", seq.shape)
        if seq.shape[0] < n_classes:
            seq = np.pad(seq, ((0, n_classes - seq.shape[0]), (0, 0)), 'constant')
            if verbose:
                print("Padded to shape: ", seq.shape)
        # print("SEQ: ", seq.shape)
    # seq = np.array([random.choice(list(encoded_features))]) # prompt with random sequence of frames from original data
    init_seq_len = seq.shape[1]
    if verbose:
        print(f"Generating sequence of length: {sequence_length}")
        print("Prompt:", prompt if (len(prompt) > 0) else "random")
        print(f"Prompt sequence shape: {seq.shape}")
        print(np.argmax(seq, axis=1))
    print("Temperature:\n", temperature)
    seq = np.array([seq])
    for i in range(sequence_length):
        preds = model.predict(seq, verbose=0)
        # print(preds)
        p_label = sample(preds[0], temperature)
        encoded_pred = np.eye(n_classes)[p_label].astype(bool) # encode the predicted label
        encoded_pred = np.array([[encoded_pred]])
        # print("\nPrediction: ", encoded_pred.astype(int))
        # print("\nPrediction: ", np.argmax(encoded_pred))
        # print(seed.shape, encoded_pred.shape)
        seq = np.concatenate((seq, encoded_pred), axis=1) # add the prediction to the sequence
        # print(seq.astype(int))
        # print("Predicted sequence:\n", seq.astype(int))
        # print("\n", seq.shape)
        # print("Predicted sequence:\n", seq.astype(int))
    g_ints = np.argmax(seq[0], axis=1)
    return g_ints[init_seq_len:]

def handle_i(unused_addr, args, msg):
  print(f"[{args[0]}] ~ {msg}")
  if msg == "ping":
    client.send_message("/i/", "pong")

def handle_g(unused_addr, args, msg):
    msg = msg.split(" ")
    # print(msg)
    sequence_length = int(msg[0])
    temperature = float(msg[1])
    prompt = msg[2:]
    prompt = list(map(int, prompt))
    print("Prompt: ", prompt)
    print("Generating sequence...")
    # Generate token sequence
    seq = generate(sequence_length=sequence_length, temperature=temperature, prompt=prompt)
    seq = seq.tolist()
    print("Generated sequence:")
    print(seq)
    client.send_message("/generate/", seq)
    print("Predictions sent.")

if __name__ == "__main__":
    # Set up OSC server and client
    client = udp_client.SimpleUDPClient("127.0.0.1", 5555)
    dispatcher = Dispatcher()
    dispatcher.map("/i/", handle_i, "i")
    dispatcher.map("/g/", handle_g, "g")
    server = osc_server.ThreadingOSCUDPServer(
        ("127.0.0.1", 4444), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
