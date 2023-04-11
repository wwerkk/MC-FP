import librosa, math, random
class Streamer:
    def __init__(self, path, block_length, frame_length, hop_length, fill_value=0.):
        self.path = path
        self.block_length = block_length
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fill_value = fill_value
        self.sr = librosa.get_samplerate(path=self.path)
        self.length = (
            librosa.get_duration(filename=self.path) # in seconds
        )
        self.n_samples = int(self.length * self.sr)
        self.n_blocks = math.ceil(self.n_samples / self.hop_length)
        self.stream = None

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.stream is None:
            Exception("Streamer is not initialized. Call new() first.")
            return None
        else:
            return next(self.stream)
    
    def __len__(self):
        return self.n_blocks
    
    def new(self):
        self.stream = librosa.stream(
            path=self.path,
            block_length=self.block_length,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            fill_value=self.fill_value
        )
        return self.stream
    
    ## TODO: fix for block length > 1
    def get_frame(self, dictionary, label, verbose=False):
        max = len(dictionary[label])
        i = random.randrange(0, max, 2) # generate a random even value, start/end values are interleaved
        print(f"Random index: {i}")
        start = dictionary[label][i] # start of frame sample
        print(start)
        end = dictionary[label][i+1] # end of frame sample
        if verbose:
            print(f"Frame start and end indices: {start}, {end}")
        start = int(start / (self.hop_length))
        if verbose:
            print(f"Frame is in block: {start}")
        for i, block in enumerate(self.new()):
            if i == start:
                return block
        return None