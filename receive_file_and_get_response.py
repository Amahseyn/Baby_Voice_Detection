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
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

# Load your previously saved model
model = AudioClassifier()
model.load_state_dict(torch.load('modelclassifier.pth'))
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