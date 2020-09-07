from preproc.remove_silence import remove_silence, normalize_samples
from pydub import AudioSegment
import librosa

def preprocess_wavs(src_file, minimum_duration=3.8, silence_threshold='optimum'):
    audio = AudioSegment.from_file(src_file)

    if audio.duration_seconds < minimum_duration:
        return [], audio.frame_rate

    #audio = audio.set_frame_rate(44100)
    audio = audio.set_channels(1)

    samples = audio.get_array_of_samples()
    max_amp = 2 ** (audio.sample_width * 8 - 1)
    normalized_samples, db_gain = normalize_samples(samples, sample_width=audio.sample_width)
    #no_silence_samples = remove_silence(normalized_samples, audio.frame_rate, audio.sample_width, silence_threshold)

    librosa_transformed = normalized_samples / max_amp

    return librosa_transformed, audio.frame_rate