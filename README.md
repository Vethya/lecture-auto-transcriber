# Lecture Auto-Transcriber

An automated tool for transcribing lecture recordings from Google Drive using OpenAI Whisper.

## Overview

Lecture Auto-Transcriber is designed to convert spoken lecture content from Google Drive into text, making educational materials more accessible and searchable. The tool processes audio/video lecture files and generates accurate transcriptions that can be used for study, reference, or accessibility purposes.

## Prerequisites

- Python 3.8 or higher
- FFmpeg

## Installation

### Installing FFmpeg

FFmpeg is required for processing audio files before transcription.

#### On Windows:

1. Download the FFmpeg build from the [official website](https://ffmpeg.org/download.html) or from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (recommended for Windows)
2. Extract the ZIP file to a location of your choice (e.g., `C:\ffmpeg`)
3. Add FFmpeg to your system PATH:
   - Open "Edit the system environment variables" from the Control Panel
   - Click "Environment Variables"
   - Edit the "Path" variable under System variables
   - Add the path to the FFmpeg bin directory (e.g., `C:\ffmpeg\bin`)
   - Click OK to save changes

#### On macOS (using Homebrew):

```sh
brew install ffmpeg
```

#### On Ubuntu/Debian:

```sh
sudo apt update
sudo apt install ffmpeg
```

### Installing Lecture Auto-Transcriber

1. Clone this repository:

```sh
git clone https://github.com/Vethya/lecture-auto-transcriber.git
cd lecture-auto-transcriber
```

2. Create and activate a virtual environment:

```sh
python -m venv env
```

### On Windows

```sh
.\env\Scripts\activate
```

### On macOS/Linux

```sh
source env/bin/activate
```

3. Install dependencies:

```sh
pip install -r requirements.txt
```

4. Set up credentials:

- Create a `config.json` file with your settings
  the Google Cloud Console by:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the Google Drive API
4. Create OAuth credentials (Desktop application)
5. Download the credentials JSON file and rename it to `credentials.json`

## Configuration

Create a `config.json` file in the root directory with the following structure:

```json
{
  "subjects": {
    "research methodology": {
      "recording_folder_id": "YOUR_RECORDING_FOLDER_ID"
    },
    "mathematics": {
      "recording_folder_id": "YOUR_RECORDING_FOLDER_ID"
    },
    "data science": {
      "recording_folder_id": "YOUR_RECORDING_FOLDER_ID"
    },
    "cloud computing": {
      "recording_folder_id": "YOUR_RECORDING_FOLDER_ID"
    },
    "project development": {
      "recording_folder_id": "YOUR_RECORDING_FOLDER_ID"
    }
  },
  "gemini_api_key": "YOUR_GEMINI_API_KEY",
  "mongodb_uri": "mongodb://localhost:27017",
  "drive_credentials": "credentials.json"
}
```

## Usage

```sh
python main.py
```
