# MimeStudio: Gesture-Controlled Music Instrument

MimeStudio is a gesture-controlled music instrument that turns real-time hand tracking into drum hits and synth notes. The system uses a webcam to detect both hands, maps finger positions to notes, and lets you sculpt the synth waveform with the left hand while playing melodies/chords with the right. It includes a lightweight drum machine with kick/snare synthesis, looping/recording, and a testing/evaluation flow for usability feedback.


## Features

- Drum mode with kick and snare driven by hand shapes and proximity.
- Synth mode with left-hand waveform shaping and right-hand pitch/volume control.
- Piano mode with finger-to-note mapping for right and left hands.
- Loop recording for drum patterns that carry across modes.
- Visual feedback overlays and a live waveform window.

## Project Structure

- src/main.py: Main application (drums, synth, piano, looping).
- src/camera_test.py: Quick camera index probe.
- src/conta_dedos.py: Finger counting prototype (MediaPipe).
- sounds/: Sample WAV files used by piano mode.
- evaluation/: Scripts and data for usability evaluation (SUS, plots).
- presentation/: Study materials and questionnaire drafts.

## Requirements

- Python 3.10+ 
- A webcam
- Audio output device (speakers or headphones)

Python dependencies are listed in both:

- requirements.txt (minimal)
- src/requirements.txt (full environment used for development)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

If you only want the minimal dependencies:

```bash
pip install -r requirements.txt
pip install pygame cvzone numpy
```

## Run

From the repository root:

```bash
python src/main.py
```

You should see a camera window and a separate waveform window in synth mode.

## Controls and Modes

The app cycles through three modes using the space bar.

### Drum Mode

- Kick: right-hand fist (0 fingers) and close to the camera.
- Snare: left-hand open (5 fingers) and close to the camera.
- R: toggle recording (starts with a 3-second countdown).
- Space: switch to Synth mode.
- Q: quit.

The hit threshold is defined in [src/main.py](src/main.py#L450) as `HIT_THRESHOLD` and depends on your camera distance and resolution.

### Synth Mode

- Left hand: draw a waveform by moving finger tips (shape changes the sound).
- Right hand: play notes with fingers up, bend pitch by moving left-right, and control volume by moving up-down.
- R: toggle recording (drum loop continues).
- Space: switch to Piano mode.
- Q: quit.

### Piano Mode

- Right hand:
	- Thumb=Do, Index=Re, Middle=Mi, Ring=Fa, Pinky=Sol
- Left hand:
	- Thumb=La, Index=Si, Pinky=Do
- Notes trigger only when the hand is close enough (same `HIT_THRESHOLD`).
- R: toggle recording (drum loop continues).
- Space: switch to Drum mode.
- Q: quit.

## Audio Assets

Piano mode loads WAV samples from the sounds folder:

- polegar.wav
- indicador.wav
- medio.wav
- anelar.wav
- mindinho.wav

If any file is missing, the corresponding finger will be silent.

## Evaluation

Usability analysis scripts are under evaluation:

- evaluation/eval.py: Full analysis with plots and SUS score calculation.
- evaluation/boxplot_sus.py: SUS score distribution boxplot.
- evaluation/taim_user_tests.csv: Example dataset.

Run with:

```bash
python evaluation/eval.py
```

## Troubleshooting

- No camera detected: run `python src/camera_test.py` and adjust the camera index if needed.
- Hand detection unstable: improve lighting and keep hands within frame.
- No sound: check your system output device and ensure `pygame` and `sounddevice` are installed.
- High latency or stutters: close other audio apps and reduce camera resolution in [src/main.py](src/main.py#L430).

## Acknowledgements

This project builds on OpenCV, MediaPipe, cvzone, and pygame for vision and audio.