import librosa
import soundfile as sf
import numpy as np
import random
import os

if not os.path.isfile('source.wav'):
    print("Error: source.wav not found in current directory")
    exit()

try:
    y, sr = sf.read('source.wav', always_2d=True)
except Exception as e:
    print(f"Error loading audio: {str(e)}")
    exit()

samples_per_frame = int(sr * 0.1)
if samples_per_frame == 0:
    print("Error: Sample rate too low for 0.1s frames")
    exit()

num_frames = (len(y) + samples_per_frame - 1) // samples_per_frame
processed_audio = []

for i in range(num_frames):
    start_idx = i * samples_per_frame
    end_idx = start_idx + samples_per_frame
    frame = y[start_idx:end_idx]
    
    repeat_count = random.randint(3, 10)
    processed_audio.append(np.tile(frame, (repeat_count, 1)))

output = np.concatenate(processed_audio)

if output.dtype != np.int16:
    output = np.clip(output, -1.0, 1.0)
    output = (output * 32767).astype(np.int16)

try:
    sf.write('output.wav', output, sr, subtype='PCM_16')
    print("Success! Generated audio saved as output.wav")
except Exception as e:
    print(f"Error saving audio: {str(e)}")