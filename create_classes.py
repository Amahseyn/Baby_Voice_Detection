# -*- coding: utf-8 -*-
"""Create_Classes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kQRLix9A0q8Vp7cVFsi-ShHJtxI7ck8z
"""

import os
import shutil

# Replace 'path_to_wav_files' with the path to your directory containing WAV files
input_directory = '/content/drive/MyDrive/donateacry-corpus-master/ios_voices'

# Create a dictionary to map prefixes to classes
class_mapping = {
    'hu': 'hungry',
    'bp': 'belly_pain',
    'bu': 'burping',
    'dc': 'discomfort',
    'ti': 'tired',
    'lo': 'lonely',
    'ch': 'cold_or_hot',
    'sc': 'scared',
    'dk': 'dont_know'
}
# Create a directory to store the class folders
output_directory = '/content/drive/MyDrive/donateacry-corpus-master/Voices'
os.makedirs(output_directory, exist_ok=True)

# Iterate through each WAV file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.wav'):
        # Extract the prefix from the filename
        prefix = filename.split('-')[-1].split('.')[0]

        # Determine the corresponding class using the class_mapping dictionary
        file_class = class_mapping.get(prefix, 'unknown_class')

        # Create a directory for the class if it doesn't exist
        class_directory = os.path.join(output_directory, file_class)
        os.makedirs(class_directory, exist_ok=True)

        # Move the file to the corresponding class directory
        shutil.copy(os.path.join(input_directory, filename), os.path.join(class_directory, filename))

import os
import shutil

# Replace 'path_to_wav_files' with the path to your directory containing WAV files
input_directory = '/content/drive/MyDrive/donateacry-corpus-master/android_voices'

# Create a dictionary to map prefixes to classes
class_mapping = {
    'hu': 'hungry',
    'bp': 'belly_pain',
    'bu': 'burping',
    'dc': 'discomfort',
    'ti': 'tired',
    'lo': 'lonely',
    'ch': 'cold_or_hot',
    'sc': 'scared',
    'dk': 'dont_know'
}
# Create a directory to store the class folders
output_directory = '/content/drive/MyDrive/donateacry-corpus-master/Voices'
os.makedirs(output_directory, exist_ok=True)

# Iterate through each WAV file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.wav'):
        # Extract the prefix from the filename
        prefix = filename.split('-')[-1].split('.')[0]

        # Determine the corresponding class using the class_mapping dictionary
        file_class = class_mapping.get(prefix, 'unknown_class')

        # Create a directory for the class if it doesn't exist
        class_directory = os.path.join(output_directory, file_class)
        os.makedirs(class_directory, exist_ok=True)

        # Move the file to the corresponding class directory
        shutil.copy(os.path.join(input_directory, filename), os.path.join(class_directory, filename))

import os
import shutil

# Replace 'path_to_input_folder' with the path to your input directory containing subfolders with WAV files
input_directory = '/content/drive/MyDrive/donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data'

# Replace 'output_folder' with the desired name of the output folder
output_directory = '/content/drive/MyDrive/donateacry-corpus-master/Voices'


# Iterate through each subfolder in the input directory
for subfolder in os.listdir(input_directory):
    subfolder_path = os.path.join(input_directory, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Create a corresponding subfolder in the output directory
        output_subfolder = os.path.join(output_directory, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # Iterate through each WAV file in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.wav'):
                # Extract the prefix from the filename

                # Create a directory for the class if it doesn't exist within the output subfolder
                class_directory = os.path.join(output_subfolder)
                os.makedirs(class_directory, exist_ok=True)

                # Copy the file to the corresponding class directory
                shutil.copy(os.path.join(subfolder_path, filename), os.path.join(class_directory, filename))