```plantuml

object Segmentation
Segmentation : - evenly-sized
Segmentation : - onset detection (WIP)
Segmentation : * (?) evenly-sized based on beat detection

object FeatureExtraction
FeatureExtraction : - MFCCs
FeatureExtraction : * (?) crude features: energy, zero-crossings
FeatureExtraction : * (?) auto-encoding?

object Clustering
Clustering : - KMeans
Clustering : * (?) other unsupervised algorithms
Clustering : https://git-lium.univ-lemans.fr/tahon/spectral-clustering-music

object Tokenizer
Tokenizer : - one-hot encoder
Tokenizer : * (?) byte-pair encoding

object Model
Model : - single GRU layer
Model : * (?) Transformer architecture

object Detokenizer

object Synthesizer
Synthesizer : * (?) Max patch
Synthesizer : * (?) Max4Live device
Synthesizer : * (?) SuperCollider script

AudioData --> "audio sequence" Segmentation

Segmentation --> "audio frames" FeatureExtraction

FeatureExtraction --> "feature sequence" Clustering

Clustering --> "label sequence" Tokenizer

Tokenizer --> "tokenized sub-sequences" Model

Prompt --> Model

Model --> "predicted tokens sequences" Detokenizer

Segmentation --> "audio frames" Detokenizer

Detokenizer --> "frame sequence" Synthesizer




```
