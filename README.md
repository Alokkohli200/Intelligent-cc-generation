# Module 1: Sound Event Detection (SED) Prototype

## Approach
This module uses a pre-trained **YAMNet** model via TensorFlow Hub to process 16kHz mono audio extracted from video files. I chose YAMNet because its training on the YouTube-8M dataset makes it highly robust for detecting environmental and non-speech events (like honking, laughter, or glass breaking). 

The script processes audio in 0.48-second windows. To prevent "over-captioning," I implemented a confidence threshold filter that intentionally drops routine ambient classifications like 'Silence' and 'White noise'.

## Known Limitations & Next Steps
- **Current Limitation:** The script currently identifies every 0.48s window independently, which can result in repetitive logs for continuous sounds (e.g., a 3-second siren logs 6 separate events).
- **Next Improvement (Temporal Smoothing):** Implement an algorithm to merge consecutive identical sound events into a single, continuous timestamp block for cleaner SRT generation.
