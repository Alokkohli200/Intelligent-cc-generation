import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from moviepy.editor import VideoFileClip

# Suppress TensorFlow warnings for a cleaner demo recording
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class SoundEventDetector:
    def __init__(self):
        print("\n--- [START] Loading YAMNet AI Model ---")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        self.class_names = pd.read_csv(class_map_path)['display_name'].values
        print("--- [SUCCESS] Model Ready ---\n")

    def run_inference(self, video_path):
        # 1. Audio Extraction
        audio_temp = "temp_audio.wav"
        print(f"Processing: {video_path}...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_temp, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
        
        # 2. Analysis
        wav_data, _ = librosa.load(audio_temp, sr=16000)
        scores, _, _ = self.model(wav_data)
        scores_np = scores.numpy()
        
        results = []
        for i in range(scores_np.shape[0]):
            idx = np.argmax(scores_np[i])
            conf = scores_np[i][idx]
            label = self.class_names[idx]
            
            # Filtering for meaningful non-speech events
            if conf >= 0.15 and label not in ['Speech', 'Silence', 'White noise']:
                results.append({
                    "Time": f"{round(i * 0.48, 2)}s",
                    "Event": label,
                    "Confidence": f"{round(float(conf) * 100, 1)}%"
                })
        
        if os.path.exists(audio_temp): os.remove(audio_temp)
        return results

if __name__ == "__main__":
    input_video = "sample_video.mp4"
    if os.path.exists(input_video):
        detector = SoundEventDetector()
        output = detector.run_inference(input_video)
        
        print("\n" + "="*45)
        print(f"{'TIMESTAMP':<12} | {'DETECTED SOUND':<20} | {'CONF'}")
        print("-" * 45)
        for row in output:
            print(f"{row['Time']:<12} | {row['Event']:<20} | {row['Confidence']}")
        print("="*45)
        print(f"\nFound {len(output)} contextually relevant events.\n")
    else:
        print(f"Error: {input_video} not found in folder!")