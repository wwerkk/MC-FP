# Music Computing Final Project
Implementing language modelling for concatenative audio synthesis.

## Real-time
For real-time generation and synthesis with the ``flux`` model the steps are as follows (it will get simplified soon):
1. open the config file located in ``models/flux`` and change the value under ``audio_path`` to the absolute path to ``flux.mp3``. If you cloned it into your home directory, on Mac it should look something like ``/Users/You/MC-FP/audio-data/flux.mp3``
2. run ``resynthesis.maxpat``
3. modify the ``sprintf`` object in ``p frames`` subpatcher so that its first argument is ``/Users/You/MC-FP/models/%s/%s_frames.json``
4. do the same with ``p config`` subpatcher so that the ``sprintf`` object's first arugment is ``/Users/You/MC-FP/models/%s/%s_config.json``
5. pick the model using the drop down menu in top left corner
6. run ``python generate.py -p /Users/You/MC-FP/models/flux``
7. turn on audio and press space to start predictions and playback
8. twiddle with some parameters
