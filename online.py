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

lock = threading.Lock()
# Parse rguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str)
parser.add_argument("-v", "--verbose", type=bool, default=False)
parser.add_argument("-sl", "--sequence_length", type=int, default=16)
parser.add_argument("-t", "--temperature", type=float, default=0.0)
parser.add_argument("--prompt", type=str, default="0")
parser.add_argument("-w", "--workers", type=int, default=2)

# Function to sample from generated predictions
# adapted from wandb character generation code
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
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
    else:
        seq = np.eye(n_classes)[0].astype(bool) # prompt with token 0
        seq = np.array([seq])
        if verbose:
            print("Random token: ", seq)
            print("Random token shape: ", seq.shape)
        seq = np.pad(seq, ((0, n_classes - seq.shape[0]), (0, 0)), 'constant')
    print(f"Generating sequence of length: {sequence_length}")
    if verbose:
        print("Prompt:", prompt if (len(prompt) > 0) else "random")
        print(f"Prompt sequence shape: {seq.shape}")
        print(np.argmax(seq, axis=1))
    init_seq_len = seq.shape[1]
    print("Temperature:\n", temperature)
    seq = np.array([seq])
    for i in range(sequence_length):
        preds = model.predict(seq, verbose=0, workers=workers, use_multiprocessing=use_multiprocessing)
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

def gen_thread():
    global prompt
    print("Starting generation thread with prompt: ", prompt)
    while True:
        with lock:
            prompt = list(map(int, prompt))
        # print("Prompt: ", prompt)
        print("Generating sequence...")
        # Generate token sequence
        seq = generate(sequence_length=sequence_length, temperature=temperature)
        seq = seq.tolist()
        with lock:
            prompt.extend(seq)
        print(f"Sequence length: {len(prompt)}")
        if len(prompt) > n_classes:
            with lock:
                prompt = prompt[-n_classes:]
            print("Prompt trimmed to length: ", len(prompt))
        # print("Generated sequence so far:")
        # print(prompt)
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

# def handle_m(unused_addr, args, msg):
#     global name, path, config, n_classes, model, prompt
#     print(f"[{args[0]}] ~ {msg}")
#     print("Loading model: ", msg)
#     with lock:
#         name = msg
#         path = f"models/{name}/"

#     config_path = path + f"{name}_config.json"
#     dict_path = path + f"{name}_frames.json"
#     features_path = path + f"{name}_features.json"
#     model_path = path + f"{name}.keras"

#     with lock:
#         config = json.load(open(config_path))
#         n_classes = config['n_classes']
#         prompt = [0]
#         model = load_model(model_path)
#     if verbose:
#         print(f"Config loaded.")
#         for key, value in config.items():
#             print(f"{key}: {value}")
#         print(f"Model loaded.")
#         model.summary()
#     print(f"All loaded successfully.")

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

global path, model, n_classes, sequence_length, temperature, prompt
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
    n_classes = 256
    sequence_length = 32
    temperature = 1.0
    server_port = 8888
    client_port = 9999
    workers = 4
    use_multiprocessing = True
    prompt = [0]
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