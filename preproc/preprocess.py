from preproc.remove_silence import remove_silence, normalize_samples
from pydub import AudioSegment
import librosa

def preprocess_wavs(src_file, dst_file, silence_threshold='optimum'):
    y, sr = librosa.load(src_file, mono=False, sr=44100)
    y_mono = librosa.to_mono(y)
    temp_mono_audio = './files/temp/mono_temp.wav'
    librosa.output.write_wav(temp_mono_audio, y_mono, sr)

    audio = AudioSegment.from_file(temp_mono_audio)

    samples = audio.get_array_of_samples()
    max_amp = 2 ** (audio.sample_width * 8 - 1)
    normalized_samples, db_gain = normalize_samples(samples, sample_width=audio.sample_width)
    no_silence_samples = remove_silence(normalized_samples, audio.frame_rate, audio.sample_width, silence_threshold)
    dst_audio = AudioSegment(
        no_silence_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=1
    )

    dst_audio.export(dst_file, format='wav')