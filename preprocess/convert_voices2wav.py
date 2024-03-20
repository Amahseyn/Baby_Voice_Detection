# -*- coding: utf-8 -*-
"""Convert_voices2wav.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mQgM8kz7y9fLdql_YRuqFRbSjqcg9ElH

Apple Devices save voices in a .caf format So we should convert all of them to wav format.

Convert caf(ios voices) to wav format
"""

import os
import subprocess

def convert_caf_to_wav(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".caf"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.wav")

            # Run ffmpeg command to convert CAF to WAV
            command = [
                "ffmpeg",
                "-i", input_path,
                output_path
            ]

            try:
                subprocess.run(command, check=True)
                print(f"Converted {file_name} to {os.path.basename(output_path)}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_name}: {e}")

# Specify input and output folders
input_folder = "/content/drive/MyDrive/donateacry-corpus-master/donateacry-ios-upload-bucket"
output_folder = "/content/drive/MyDrive/donateacry-corpus-master/ios_voices"

# Convert CAF to WAV
convert_caf_to_wav(input_folder, output_folder)

"""Convert Android Voices (3gp) to wav format"""

def convert_3gp_to_wav(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".3gp"):

            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.wav")

            # Run ffmpeg command to convert CAF to WAV
            command = [
                "ffmpeg",
                "-i", input_path,
                output_path
            ]

            try:
                subprocess.run(command, check=True)
                print(f"Converted {file_name} to {os.path.basename(output_path)}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_name}: {e}")

# Specify input and output folders
input_folder = "/content/drive/MyDrive/donateacry-corpus-master/donateacry-android-upload-bucket"
output_folder = "/content/drive/MyDrive/donateacry-corpus-master/android_voices"

# Convert CAF to WAV
convert_3gp_to_wav(input_folder, output_folder)