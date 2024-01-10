import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# Define your folder structure
data_dir = '/content/drive/MyDrive/donateacry-corpus/Voices'
classes = os.listdir(data_dir)

# Load and preprocess audio data with padding/truncating
def load_and_preprocess_data(data_dir, classes, target_length=500):  # Choose a suitable target length
    data = []
    labels = []

    for i, class_name in enumerate(classes):
        print("Reading ", class_name)
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, _ = librosa.load(file_path, sr=None)
                # Perform preprocessing (e.g., extract MFCCs)
                mfccs = librosa.feature.mfcc(y=audio_data, sr=len(audio_data), n_mfcc=13)
                # Pad or truncate the sequences to the target length
                if mfccs.shape[1] < target_length:
                    mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])), mode='constant')
                else:
                    mfccs = mfccs[:, :target_length]
                data.append(mfccs.T.astype(np.float32))  # Convert to float32 and transpose the MFCCs matrix
                labels.append(i)

    return np.array(data), np.array(labels)

# Split data into training and testing sets
data, labels = load_and_preprocess_data(data_dir, classes)
labels = to_categorical(labels, num_classes=len(classes))  # Convert labels to one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Print the shapes
print("X_train shape:", X_train.shape)

# Model Architecture (LSTM)
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))  # Adjust based on the printed shapes
x = LSTM(128, return_sequences=True)(input_layer)
x = LSTM(64)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)

# Learning Rate Schedule
optimizer = Adam(learning_rate=0.00001)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
batch_size = 128
epochs = 100  # Adjust the number of epochs

# Train the model with data augmentation
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy*100:.2f}%')

# Save the model
model.save('audio_classification_model_rnn.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
