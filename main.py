import download_files as d
import audio_classification as ac
from localize_audio import apply_doa




if __name__ == "__main__":
    print("Downloading files")
    df = d.get_data(d.N_SAMPLES)
    print(df.head())
    df = apply_doa(df, audio_col="audio_path", base_dir="")

    print(df[["sample_id", "doa_angle_deg", "doa_rf_deg", "doa_mlp_deg", "doa_error"]])

    # Apply the classification to the 'audio_path' column of the DataFrame
    # This might take some time depending on the number of audio samples
    df['audio_object_classification'] = df['audio_path'].apply(ac.classify_audio_objects)

    # Display the DataFrame with the new classification results
    print(df[['sample_id', 'audio_caption', 'tags', 'audio_object_classification']])
