import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Loading the model and processor
model_name = "GiftMark/akan-whisper-model"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Setting the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Function to resample audio
def resample_audio(audio_path, target_sample_rate=16000):
    waveform, original_sample_rate = torchaudio.load(audio_path)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy(), target_sample_rate

# Function to process and transcribe audio
def transcribe_audio(file_path: str):
    # Loading and resample audio
    audio_input, sample_rate = resample_audio(file_path, target_sample_rate=16000)
    
    # Preprocessing the audio
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
    
    # Generating transcription
    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

#the audio path and file 
if __name__ == "__main__":
    audio_file = "~/twi.wav" 
    transcription = transcribe_audio(audio_file)
    print("Transcribed Output:", transcription)
