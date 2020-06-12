import librosa

def audio_cut_raw_data(input_file, n_samples=88100, sr=None):
    # find starting index on folder

    samples, sample_rate = librosa.load(input_file, sr=sr, mono=True)

    total_frames = len(samples)

    n_intervals = total_frames // n_samples
    remainder = total_frames % n_samples
    if n_intervals == 0:
        return [], sample_rate
    elif n_intervals == 1:
        gap = 0
    else:
        gap = remainder // (n_intervals - 1)

    raw_chunks = [samples[i * (n_samples + gap): i * (n_samples + gap) + n_samples] for i in range(n_intervals)]

    return raw_chunks, sample_rate
