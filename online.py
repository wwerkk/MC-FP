## TODO: cleanup


#import dependencies
import numpy as np
from keras.models import load_model
import json
import argparse
import time
import threading
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client
from keras.utils import to_categorical

lock = threading.Lock()
# Parse rguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str)
parser.add_argument("-v", "--verbose", type=bool, default=False)
parser.add_argument("-sl", "--sequence_length", type=int, default=16)
parser.add_argument("-t", "--temperature", type=float, default=0.0)
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("-w", "--workers", type=int, default=2)

# from character generation code tutorial by Lukas Biewald
# https://github.com/lukas/ml-class/blob/master/projects/7-text-generation/char-gen.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = preds[0]
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # print(preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate output
def generate(prompt=[], length=4, temperature=1.0, include_prompt=False, verbose=0):
    print("Read prompt: ", prompt)
    # generate a sequence of given length       
    if len(prompt) == 0:
        # if it's there's no prompt, use an array of zeroes
        prompt = np.zeros((maxlen), dtype="uint8")
        print(f"No prompt provided, using {maxlen} zeroes instead.")
    elif len(prompt) < maxlen:
        # if prompt is too short, pad it with zeroes from the left to match correct input shape
        prompt = np.pad(prompt, (maxlen - len(prompt), 0), 'constant', constant_values=(0, 0))
        print("Prompt too short, padded to length: ", maxlen)
    elif len(prompt) > maxlen:
        # if it's too long, then trim it
        prompt = prompt[-maxlen:]
        print(f"Prompt too long, using {maxlen} last elements: ")
    prompt_ = to_categorical(prompt, num_classes=n_classes)
    prompt_ = np.array([prompt_])
    
    seq = []
    for i in range(length):
        # make prediction based on prompt
        ps = model.predict(prompt_, verbose=verbose)
        # sample from predictions
        p_label = sample(ps, temperature)
        # add sampled label to sequence
        seq.extend([int(p_label)])
        # one-hot encode sampled label
        p_label_ = to_categorical(p_label, num_classes=n_classes)
        p_label_ = np.array([[p_label_]])
        # append encoded label to the prompt for next prediction
        prompt_ = np.append(prompt_, p_label_, axis=1)
    if include_prompt:
        prompt = np.append(prompt, seq)
        return prompt
    else:
        return seq

def gen_thread():
    global prompt, seq
    while True:
        print("Generating sequence...")
        with lock:
            seq = generate(prompt=prompt, length=sequence_length)
            prompt.extend(seq) # add generated sequence to the prompt
            prompt = prompt[-maxlen:] # take only last maxlen elements
        client.send_message("/generated/", seq)

def handle_g(unused_addr, args, msg):
    global sequence_length, temperature, prompt
    print(msg)
    msg = msg.split(" ")
    with lock:
        sequence_length = int(msg[0])
        temperature = float(msg[1])
        if len(msg) > 2: # if a new prompt is given, use it to restart the state
                prompt = msg[2:]
                prompt = list(map(int, prompt))
    print(f"Sequence length changed to: {sequence_length}")
    print(f"Temperature changed to: {temperature}")
    print(f"Started generation with new parameters and prompt: {prompt}")

def osc_thread():
    print("Starting OSC thread...")
    # Set up OSC server and client
    dispatcher = Dispatcher()
    dispatcher.map("/g/", handle_g, "g")
    # dispatcher.map("/m/", handle_m, "m")
    server = osc_server.ThreadingOSCUDPServer(
        ("127.0.0.1", server_port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()

global path, model, n_classes, maxlen, sequence_length, temperature, prompt
global verbose
global client, server_port, client_port, workers, use_multiprocessing

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
    with lock:
        path = path[:-1]
name = path.split("/")[-1]
with lock:
    path = path + "/"
    verbose = False
    sequence_length = 32
    temperature = 1.0
    server_port = 8888
    client_port = 9999
    prompt = args.prompt
    prompt = [int(i) for i in args.prompt]
    print("Prompt: ", prompt)
print(f"Path: {path}")
# Load model and necessary dictionaries
print(f"Loading model...")
if path[-1] == "/":
    with lock:
        path = path[:-1]
name = path.split("/")[-1]
with lock:
    path = path + "/"

config_path = path + f"{name}_config.json"
dict_path = path + f"{name}_frames.json"
features_path = path + f"{name}_features.json"
model_path = path + f"{name}.keras"

config = json.load(open(config_path))
filename = config['filename']
sr = config['sr']
n_classes = config['n_classes']
maxlen = config['maxlen']
print("maxlen: ", maxlen)
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
with lock:
    model = load_model(model_path)
model.summary()
print(f"All loaded successfully.")

# Set up OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", client_port)

# Start threads
osc_thread = threading.Thread(target=osc_thread)
osc_thread.start()
gen_thread = threading.Thread(target=gen_thread)
gen_thread.start()