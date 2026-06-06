import torch
import soundfile as sf
import pandas as pd
from transformers import pipeline

# =====================================================
# Device Selection
# =====================================================
DEVICE = 0 if torch.cuda.is_available() else -1

print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# =====================================================
# Load Model
# =====================================================
classifier = pipeline(
    task="audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    device=DEVICE
)

# =====================================================
# Audio Classification Function
# =====================================================
def classify_audio_objects(audio_path, threshold=0.15):
    """
    Classify environmental sounds in an audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    threshold : float
        Minimum confidence score.

    Returns
    -------
    str
        Semicolon-separated labels and scores.
    """
    try:
        # Load audio
        audio, sample_rate = sf.read(audio_path)

        # Convert stereo -> mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Convert to float32
        audio = audio.astype("float32")

        with torch.no_grad():
            predictions = classifier(
                audio,
                sampling_rate=sample_rate
            )

        results = [
            f"{pred['label']} ({pred['score']:.2f})"
            for pred in predictions
            if pred["score"] >= threshold
        ]

        return "; ".join(results)

    except Exception as e:
        return f"ERROR: {str(e)}"


