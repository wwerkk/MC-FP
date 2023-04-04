```plantuml

object Segmentation
Segmentation : - evenly-sized
Segmentation : - onset detection (WIP)

object FeatureExtraction
FeatureExtraction : MFCCs

object Clustering
Clustering : KMeans

object Tokenizer
Tokenizer : one-hot encoding

object Model
Model : single GRU layer

object Detokenizer

object Synthesizer

AudioData --> "audio sequence" Segmentation

Segmentation --> "audio frames" FeatureExtraction

FeatureExtraction --> "feature sequence" Clustering

Clustering --> "label sequence" Tokenizer

Tokenizer --> "tokenized sub-sequences" Model

Prompt --> Model

Model --> "predicted tokens sequences" Detokenizer

Segmentation --> "audio frames" Detokenizer

Detokenizer --> "frame sequence" Synthesizer

Synthesizer --> "audio sequence" Output


```
