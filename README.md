# Akan Whisper Model: Transcribing Asante Twi Audio

This repository contains a Python-based script utilizing the "GiftMark/akan-whisper-model," a pre-trained Whisper model fine-tuned for Asante Twi. The script processes audio files to generate text transcriptions in Asante Twi.

## Features
- **Resampling Audio**: Ensures compatibility by resampling audio files to the target sample rate of 16,000 Hz.
- **GPU Support**: Automatically utilizes GPU if available for faster processing.
- **End-to-End Transcription**: Simple input-output flow for transcribing Asante Twi audio files.

## Requirements
To run the script, ensure the following are installed:

- Python 3.7+
- PyTorch
- Torchaudio
- Transformers (by Hugging Face)

You can install the dependencies with:
```bash
pip install torch torchaudio transformers
```

## Code Explanation

### Loading the Model and Processor
```python
model_name = "GiftMark/akan-whisper-model"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
```
The Whisper processor and model are loaded using the Hugging Face Transformers library.

### GPU/CPU Configuration
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```
The script detects if a GPU is available and utilizes it for computations. Otherwise, it defaults to the CPU.

### Resampling Audio
The audio is resampled to the target sample rate of 16,000 Hz, as required by the model.
```python
def resample_audio(audio_path, target_sample_rate=16000):
    waveform, original_sample_rate = torchaudio.load(audio_path)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy(), target_sample_rate
```

### Transcribing Audio
The transcription process involves:
1. Loading and preprocessing audio input.
2. Generating predictions using the model.
3. Decoding the predictions into text.

```python
def transcribe_audio(file_path: str):
    audio_input, sample_rate = resample_audio(file_path, target_sample_rate=16000)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
```

### Example Usage
The main function demonstrates how to use the script to transcribe a sample audio file:
```python
if __name__ == "__main__":
    audio_file = "/Users/josephadeleke/Documents/dataset/twi.wav"
    transcription = transcribe_audio(audio_file)
    print("Transcribed Output:", transcription)
```
Replace `audio_file` with the path to your audio file.

## Running the Script
1. Save the script as `transcribe_twi.py`.
2. Place your audio file in the desired location.
3. Run the script with:
```bash
python transcribe_twi.py
```
The transcribed output will be displayed in the console.

## Contribution
Feel free to raise issues or submit pull requests to improve the script or add features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

