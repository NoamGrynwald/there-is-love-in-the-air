from datasets import load_dataset, Audio
import pandas as pd
import os

N_SAMPLES = 20
AUDIO_DIR = "audio_files"
COLUMNS = ["sample_id", "latitude", "longitude", "audio_caption", "tags", "address", "audio"]


def load_streaming_dataset(columns: list[str]) -> object:
    dataset = load_dataset("MVRL/GeoSound", split="train", streaming=True)
    dataset = dataset.select_columns(columns)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def save_audio(sample_id: str, audio: dict) -> str:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    filename = f"{AUDIO_DIR}/{sample_id}.wav"
    audio_bytes = audio.get("bytes")
    if audio_bytes:
        with open(filename, "wb") as f:
            f.write(audio_bytes)
    return filename


def process_sample(sample: dict) -> dict:
    audio = sample.pop("audio")
    sample["audio_path"] = save_audio(sample["sample_id"], audio)
    return sample


def get_data(number_of_samples: int) -> pd.DataFrame:
    dataset = load_streaming_dataset(COLUMNS)
    samples = list(dataset.take(number_of_samples))
    processed = [process_sample(sample) for sample in samples]
    return pd.DataFrame(processed)


if __name__ == "__main__":
    pass
    # df = get_data(N_SAMPLES)
    # print(df.head())
