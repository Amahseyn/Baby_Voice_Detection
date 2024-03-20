import requests
import os 

# Define the URL of the FastAPI server
url = "http://localhost:8000/predict/"

# Define the file path of the audio file you want to send
audio_file_path = "C:/Users/mhhas/Downloads/Voices/lonely/131D2346-92AD-42C3-BB28-B87FDEDD3AA4-1428513247-1.0-f-04-lo.wav"

# Check if the file exists
if os.path.exists(audio_file_path):
    print("The data exists")
else:
    print("The specified file path does not exist")

# Open the audio file in binary mode
with open(audio_file_path, "rb") as file:
    
    # Send a POST request to the server with the audio file
    response = requests.post(url, files={"files": ("audio.wav", file, "audio/wav")})

# Print the response from the server
print("Response from server:")
print(response.text[0:500])
