import librosa, math
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