import json
import os
import asyncio
import aiofiles
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
import whisper
from pymongo import MongoClient
import concurrent.futures
from threading import Lock
from tqdm.auto import tqdm

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set up Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']
flow = InstalledAppFlow.from_client_secrets_file(config['drive_credentials'], SCOPES)
creds = flow.run_local_server(port=0)
service = build('drive', 'v3', credentials=creds)

# Set up MongoDB
client = MongoClient(config['mongodb_uri'])
db = client['lecture_summaries']
processed_collection = db['processed_videos']

# Load Whisper model
model = whisper.load_model("tiny.en")

def transcribe_with_ffmpeg_preprocessing(video_path):
    """Transcribe a video with ffmpeg preprocessing to ensure audio compatibility."""
    try:
        # Extract filename without extension
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = f"temp_audio_{base_name}.wav"
        
        # Use ffmpeg to extract audio in a format Whisper can handle reliably
        import subprocess
        cmd = [
            "ffmpeg", "-i", video_path, 
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit audio
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            "-af", "dynaudnorm",  # Normalize audio (helps with volume issues)
            "-y",  # Overwrite output file if it exists
            audio_path
        ]
        
        # Run ffmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the audio file was created
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            # First try with tiny.en model
            try:
                # Transcribe with verbose=False to avoid excessive output
                result = model.transcribe(audio_path, verbose=False)
                
                # Format text with line breaks by segments
                formatted_text = ""
                for segment in result["segments"]:
                    # Add each segment as a separate paragraph
                    formatted_text += f"{segment['text'].strip()}\n"
                
                os.remove(audio_path)
                return formatted_text
                    
            except Exception as e:
                # If we get a specific error related to tensor reshaping, try base model
                if "reshape tensor" in str(e) or "Linear" in str(e):
                    print(f"First attempt failed, trying with base.en model...")
                    try:
                        # Load a different model that might work better
                        base_model = whisper.load_model("base.en")
                        result = base_model.transcribe(audio_path, verbose=False)
                        
                        # Format text with line breaks
                        formatted_text = ""
                        for segment in result["segments"]:
                            formatted_text += f"{segment['text'].strip()}\n\n"
                        
                        os.remove(audio_path)
                        return formatted_text
                    except Exception as e2:
                        print(f"⚠️ Second transcription attempt failed: {str(e2)}")
                        os.remove(audio_path)
                        return None
                else:
                    print(f"⚠️ Transcription error: {str(e)}")
                    os.remove(audio_path)
                    return None
        else:
            print(f"⚠️ Failed to extract audio from {os.path.basename(video_path)}")
            return None
    except Exception as e:
        print(f"⚠️ Error preprocessing audio for {os.path.basename(video_path)}: {str(e)}")
        # Clean up temp audio file if it exists
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        return None

async def download_video(file_id, file_name):
    """Download a video file from Google Drive with retries."""
    temp_video = f"temp_{file_name}"
    request = service.files().get_media(fileId=file_id)
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with open(temp_video, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    print(f"Downloading {file_name}: {int(status.progress() * 100)}%")
            return temp_video
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Download failed, retrying in {wait_time}s... ({retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                print(f"Failed to download {file_name} after {max_retries} attempts: {e}")
                raise

async def transcribe_multiple_videos(video_paths):
    """Transcribe multiple videos in parallel using a thread pool."""
    print(f"Starting parallel transcription of {len(video_paths)} videos")
    
    # Create a thread pool with workers based on CPU count (but limit it to avoid memory issues)
    max_workers = min(os.cpu_count() - 1 or 1, 3)  # Use at most 3 workers to avoid OOM
    
    # Lock for print statements to avoid garbled output
    print_lock = Lock()
    
    def transcribe_with_progress(video_path):
        try:
            # Verify the file exists and has content
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                with print_lock:
                    print(f"⚠️ Error: File {os.path.basename(video_path)} is empty or missing")
                return video_path, None
                
            # Use ffmpeg preprocessing
            with print_lock:
                print(f"Transcribing {os.path.basename(video_path)}")
                
            text = transcribe_with_ffmpeg_preprocessing(video_path)
            
            if text:
                with print_lock:
                    print(f"✓ Finished transcribing {os.path.basename(video_path)}")
                return video_path, text
            else:
                with print_lock:
                    print(f"⚠️ Invalid or empty transcription for {os.path.basename(video_path)}")
                return video_path, None
                
        except Exception as e:
            with print_lock:
                print(f"⚠️ Error processing {os.path.basename(video_path)}: {str(e)}")
            return video_path, None
    
    # Use ThreadPoolExecutor to parallelize the transcription
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(transcribe_with_progress, path): path for path in video_paths}
        
        # Process as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Transcribing videos"):
            try:
                path, text = future.result()
                if text is not None:  # Only add successful transcriptions
                    results[path] = text
            except Exception as e:
                print(f"⚠️ Transcription task failed: {str(e)}")
            
    return results

async def process_subject(subject):
    """Process all videos for a subject."""
    try:
        recording_folder_id = config['subjects'][subject]['recording_folder_id']
        print(f"Processing subject: {subject}")
        
        # List video files
        results = service.files().list(
            q=f"'{recording_folder_id}' in parents and mimeType contains 'video/'",
            fields="files(id, name)"
        ).execute()
        files = results.get('files', [])
        
        if not files:
            print(f"No videos found in {subject} recording folder.")
            return
        
        # Filter only unprocessed videos
        unprocessed_files = []
        for file in files:
            if not processed_collection.find_one({'subject': subject, 'video_name': file['name']}):
                unprocessed_files.append(file)
            else:
                print(f"Skipping {file['name']} (already processed)")
        
        if not unprocessed_files:
            print(f"No unprocessed videos for {subject}.")
            return
        
        # First, download all videos in parallel
        download_tasks = []
        for file in unprocessed_files:
            task = asyncio.create_task(download_video(file['id'], file['name']))
            download_tasks.append((file, task))
        
        # Wait for all downloads to complete
        downloaded_videos = {}
        for file, task in download_tasks:
            try:
                temp_path = await task
                # Verify downloaded file
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    downloaded_videos[file['name']] = {
                        'file': file,
                        'temp_path': temp_path
                    }
                else:
                    print(f"⚠️ Warning: Downloaded file {file['name']} is empty or missing")
            except Exception as e:
                print(f"⚠️ Failed to download {file['name']}: {str(e)}")
        
        if not downloaded_videos:
            print(f"No videos were successfully downloaded for {subject}")
            return
        
        # Now transcribe all videos in parallel using threads
        video_paths = [info['temp_path'] for info in downloaded_videos.values()]
        transcriptions = await transcribe_multiple_videos(video_paths)
        
        if not transcriptions:
            print(f"⚠️ No successful transcriptions for {subject}")
            # Clean up downloaded files if no transcriptions
            for info in downloaded_videos.values():
                if os.path.exists(info['temp_path']):
                    os.remove(info['temp_path'])
            return
        
        # Process each video further (save transcript to text file)
        process_tasks = []
        for video_name, info in downloaded_videos.items():
            if info['temp_path'] in transcriptions:
                # Get the transcription text
                transcription = transcriptions[info['temp_path']]
                
                # Process asynchronously
                task = asyncio.create_task(process_transcribed_video(
                    subject, 
                    info['file'],
                    info['temp_path'], 
                    transcription
                ))
                process_tasks.append(task)
        
        # Wait for all processing to complete
        if process_tasks:
            await asyncio.gather(*process_tasks, return_exceptions=True)
        
        # Clean up
        for info in downloaded_videos.values():
            if os.path.exists(info['temp_path']):
                os.remove(info['temp_path'])
                
        print(f"✅ Completed processing subject: {subject}")
    except Exception as e:
        print(f"❌ Error processing subject {subject}: {str(e)}")

async def process_transcribed_video(subject, file, temp_video, transcription):
    """Process a video that has already been transcribed."""
    video_name = file['name']
    txt_name = f"{video_name}_transcript.txt"
    
    try:
        # Save transcription to text file
        async with aiofiles.open(txt_name, 'w', encoding='utf-8') as f:
            await f.write(transcription)
        
        print(f"Generated transcript file: {txt_name}")
        
        # Mark as processed in MongoDB
        processed_collection.insert_one({'subject': subject, 'video_name': video_name})
        print(f"✅ Completed processing {video_name}")
        
    except Exception as e:
        print(f"❌ Error processing {video_name}: {e}")

async def main():
    """Main function to process all subjects."""
    tasks = [process_subject(subject) for subject in config['subjects']]
    await asyncio.gather(*tasks)
    print("Script completed.")

if __name__ == "__main__":
    asyncio.run(main())