import streamlit as st
import pyaudio
import wave
import os
import numpy as np
from io import BytesIO
import base64

# Function to load the image and convert it to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to the locally stored QR code image
qr_code_path = "qrcode.png"  # Ensure the image is in your app directory

# Convert image to base64
qr_code_base64 = get_base64_of_bin_file(qr_code_path)

# Custom CSS to position the QR code close to the top-right corner under the "Deploy" area
st.markdown(
    f"""
    <style>
    .qr-code {{
        position: fixed;  /* Keeps the QR code fixed in the viewport */
        top: 10px;       /* Sets the distance from the top of the viewport */
        right: 10px;     /* Sets the distance from the right of the viewport */
        width: 200px;    /* Adjusts the width of the QR code */
        z-index: 100;    /* Ensures the QR code stays above other elements */
    }}
    </style>
    <img src="data:image/png;base64,{qr_code_base64}" class="qr-code">
    """,
    unsafe_allow_html=True
)


# Constants for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"

def detect_silence(audio_data, threshold=500, min_silence_len=0.3):
    chunk_size = int(RATE * 0.01)  # 10ms chunks
    rms = np.array([np.sqrt(np.mean(chunk**2)) for chunk in np.array_split(audio_data, len(audio_data)//chunk_size)])
    
    is_silent = rms < threshold
    silent_regions = []
    silent_start = None
    
    for i, silent in enumerate(is_silent):
        if silent and silent_start is None:
            silent_start = i
        elif not silent and silent_start is not None:
            if (i - silent_start) * 0.01 >= min_silence_len:
                silent_regions.append((silent_start * chunk_size, i * chunk_size))
            silent_start = None
    
    if silent_start is not None and (len(is_silent) - silent_start) * 0.01 >= min_silence_len:
        silent_regions.append((silent_start * chunk_size, len(audio_data)))
    
    st.write(f"Audio RMS min: {np.min(rms):.2f}, max: {np.max(rms):.2f}, mean: {np.mean(rms):.2f}")
    st.write(f"Silence threshold: {threshold}")
    st.write(f"Number of silent regions detected: {len(silent_regions)}")
    
    return silent_regions

def trim_silence_with_pause(audio_data, silent_regions, buffer_ms=50, pause_duration=0.5):
    if not silent_regions:
        st.write("No silent regions to trim.")
        return audio_data

    buffer_samples = int(RATE * buffer_ms / 1000)
    pause_samples = int(RATE * pause_duration)  # Number of samples for 0.5 seconds of silence
    silence = np.zeros(pause_samples, dtype=np.int16)  # Array of silence to insert
    keep_regions = []
    last_end = 0
    
    for start, end in silent_regions:
        if start > last_end:
            keep_regions.append((last_end, start))
        last_end = end
    
    if last_end < len(audio_data):
        keep_regions.append((last_end, len(audio_data)))

    # Add silence between the non-silent regions
    trimmed_audio = np.concatenate(
        [np.concatenate([audio_data[max(0, start-buffer_samples):min(end+buffer_samples, len(audio_data))], silence]) 
         for start, end in keep_regions]
    )
    
    return trimmed_audio

def record_audio(duration):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    st.write(f"Recording for {duration} seconds...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    return audio_data

def adjust_volume(audio_data, volume):
    return np.clip(audio_data * volume, -32768, 32767).astype(np.int16)

def play_audio(audio_data, volume=1.0):
    adjusted_audio = adjust_volume(audio_data, volume)

    virtual_file = BytesIO()
    with wave.open(virtual_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(RATE)
        wf.writeframes(adjusted_audio.tobytes())

    virtual_file.seek(0)
    st.audio(virtual_file)

# Streamlit UI
st.title("Audio Recorder with Trimmed Playback")

record_duration = st.slider("Recording Duration (seconds)", 1, 30, 5, 1)
silence_threshold = st.slider("Silence Threshold", 50, 500, 50, 50)
volume = st.slider("Playback Volume", 0.0, 2.0, 1.0, 0.1)

if st.button("Record"):
    audio_data = record_audio(record_duration)
    st.session_state['audio_data'] = audio_data
    silent_regions = detect_silence(audio_data, silence_threshold)
    st.session_state['silent_regions'] = silent_regions

col1, col2 = st.columns(2)

with col1:
    if st.button("Play Original"):
        if 'audio_data' in st.session_state:
            play_audio(st.session_state['audio_data'], volume)
        else:
            st.warning("No audio recorded yet!")

with col2:
    if st.button("Play Trimmed with Pause"):
        if 'audio_data' in st.session_state and 'silent_regions' in st.session_state:
            trimmed_audio = trim_silence_with_pause(st.session_state['audio_data'], st.session_state['silent_regions'])
            play_audio(trimmed_audio, volume)
            st.write(f"Trimmed audio duration: {len(trimmed_audio) / RATE:.2f}s")
            st.write(f"Original audio duration: {len(st.session_state['audio_data']) / RATE:.2f}s")
        else:
            st.warning("No audio recorded yet!")
