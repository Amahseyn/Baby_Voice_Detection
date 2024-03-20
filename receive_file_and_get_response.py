import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.nn as nn
from torch.nn import init
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import numpy as np
import tempfile

app = FastAPI()

# Define your model architecture
class AudioClassifier (nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.5),  # Add dropout

            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),  # Add dropout

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # Add dropout

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # Add dropout
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(64, 32),  # Add an extra dense layer
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load your previously saved model
model = AudioClassifier()
model.load_state_dict(torch.load('audio_classifier_model.pth'))
model.eval()


class AudioFile(BaseModel):
    file: bytes

def preprocess_audio(audio_file_path: str):
    try:
        # Attempt to load with torchaudio (might handle WAV with headers)
        waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True)
    except (OSError, ValueError):
        # Fallback to soundfile for raw audio data (no header)
        import soundfile as sf
        waveform, sample_rate = sf.read(audio_file_path)

    mel_specgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )(waveform)
    mel_specgram = torch.log(1 + mel_specgram)
    return mel_specgram.unsqueeze(0)

@app.post("/predict/")
async def predict_audio(files: List[UploadFile] = File(...)):
    try:
        for file in files:
            content = await file.read()
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(content)
                temp_file_name = temp.name
            # Now you can load it with torchaudio
            input_tensor = preprocess_audio(temp_file_name)
            # Don't forget to remove the temporary file
            os.remove(temp_file_name)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output).item()
                return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)