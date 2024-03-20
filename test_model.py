import os
import torchaudio
import torch 
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

data_dir = 'C:/Users/mhhas/Downloads/Voices'
classes = os.listdir(data_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SoundDS(Dataset):
    def __init__(self, directory_path):
        self.classes = os.listdir(directory_path)
        self.data_path = directory_path
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        length = 0
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_path, class_name)
            length += len([filename for filename in os.listdir(class_dir) if filename.endswith('.wav')])
        return length

    def __getitem__(self, index):
        data = []
        labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_path, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.wav'):
                    audio_file = os.path.join(class_dir, filename)

                    # Get the Class ID (You need to fill in this part based on your dataset structure)
                    class_id = self.classes.index(class_name)

                    aud, _ = torchaudio.load(audio_file, normalize=True)
                    reaud = torchaudio.transforms.Resample(orig_freq=aud.size(1), new_freq=self.sr)(aud)

                    # Resize to match the desired duration
                    dur_aud = torch.nn.functional.interpolate(reaud.unsqueeze(0), size=self.duration, mode='linear').squeeze(0)

                    shift_aud = torchaudio.transforms.TimeMasking(time_mask_param=100)(dur_aud)
                    sgram = torchaudio.transforms.MelSpectrogram(n_mels=64, n_fft=1024)(shift_aud)
                    aug_sgram = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)(sgram)
                    return aug_sgram, class_id

myds = SoundDS(data_dir)

num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
print(len(train_dl))
print(len(val_dl))

# Define your model architecture
class AudioClassifier (nn.Module):
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

# Function to perform inference
def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient calculation
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    # Calculate accuracy
    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Perform inference
inference(model, val_dl)
