from datasets import load_dataset, Audio
import pandas as pd
import soundfile as sf
import io
import os

N_SAMPLES = 20
AUDIO_DIR = "audio_files"
COLUMNS = ["sample_id", "latitude", "longitude", "audio_caption", "tags", "address", "audio"]
MIN_CHANNELS = 2


def load_streaming_dataset(columns: list[str]) -> object:
    dataset = load_dataset("MVRL/GeoSound", split="train", streaming=True)
    dataset = dataset.select_columns(columns)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def get_audio_channels(audio_bytes: bytes) -> int:
    """Return the number of channels in the audio bytes, or 0 on failure."""
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            return f.channels
    except Exception:
        return 0


def save_audio(sample_id: str, audio: dict, overwrite: bool = True) -> str:
    # Ensures the directory exists; exist_ok=True prevents errors if it's already there
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Safely construct the file path across different operating systems
    filename = os.path.join(AUDIO_DIR, f"{sample_id}.wav")

    # If overwrite is False, check if the file exists and skip saving
    if not overwrite and os.path.exists(filename):
        return filename

    # Retrieve and save the audio bytes if they exist
    audio_bytes = audio.get("bytes")
    if audio_bytes:
        with open(filename, "wb") as f:
            f.write(audio_bytes)

    return filename


def process_sample(sample: dict) -> dict | None:
    """Process a sample, returning None if it doesn't meet the channel requirement."""
    audio = sample.pop("audio")

    audio_bytes = audio.get("bytes")
    if not audio_bytes:
        return None

    if get_audio_channels(audio_bytes) < MIN_CHANNELS:
        return None

    sample["audio_path"] = save_audio(sample["sample_id"], audio)
    return sample


def get_data(number_of_samples: int) -> pd.DataFrame:
    dataset = load_streaming_dataset(COLUMNS)

    processed = []
    for sample in dataset:
        result = process_sample(sample)
        if result is not None:
            processed.append(result)
            if len(processed) >= number_of_samples:
                break

    return pd.DataFrame(processed)


if __name__ == "__main__":
    pass
    # df = get_data(N_SAMPLES)
    # print(df.head())