import download_files as d
import audio_classification as ac
    



if __name__ == "__main__":
    print("Downloading files")
    print("Downloading files")
    df = d.get_data(d.N_SAMPLES)

    # Apply the classification to the 'audio_path' column of the DataFrame
    # This might take some time depending on the number of audio samples
    df['audio_object_classification'] = df['audio_path'].apply(ac.classify_audio_objects)

    # Display the DataFrame with the new classification results
    print(df[['sample_id', 'audio_caption', 'tags', 'audio_object_classification']])
