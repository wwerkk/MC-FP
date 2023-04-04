```plantuml


object Streamer
Streamer : affordances:
Streamer : - block streaming audio files
Streamer : --
Streamer : members:
Streamer : - librosa.stream

object AudioProcessor
AudioProcessor : affordances:
AudioProcessor : - block processing
AudioProcessor : - segmentation
AudioProcessor : - feature extraction
AudioProcessor : - clustering
AudioProcessor : --
AudioProcessor : members:
AudioProcessor : - Streamer



object ModelWrapper
ModelWrapper : affordances:
ModelWrapper : - model building, training, prediction
ModelWrapper : - model and dictionary saving
ModelWrapper : - model and dictionary loading
ModelWrapper : --
ModelWrapper : members:
ModelWrapper : - keras.Model
ModelWrapper : - dictionaries necessary for model prediction

object Encoder
Encoder : affordances:
Encoder : - split of sequence into sub-sequences
Encoder : - sub-sequence one-hot encoding

object Notebook
Notebook : affordances:
Notebook : - remote or local execution
Notebook : - providing an interface to behaviour of all member classes
Notebook : --
Notebook : members:
Notebook : - AudioProcessor
Notebook : - Encoder
Notebook : - ModelWrapper

object Generator
Generator : affordances:
Generator : - remote or local execution
Generator : - prompt-based prediction of encoded timeseries
Generator : - timeseries sequence decoding
Generator : - real-time communication with other software via UDP
Generator : --
Generator : members:
Generator : - ModelWrapper
Generator : - Decoder
Generator : - UDP Server

object MaxPatch
MaxPatch : affordances:
MaxPatch : - communication with Generator script
MaxPatch : - granular synthesis based on received tokens



AudioData "time series data" --> Notebook

Streamer - AudioProcessor

AudioProcessor - Notebook

Encoder -- Notebook

ModelWrapper -- Notebook


Notebook --> "model, dictionaries" Generator

ModelWrapper -- Generator

Generator --> "generated sequence of tokens" MaxPatch
```
