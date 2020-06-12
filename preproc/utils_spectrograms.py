from preproc.utils_audio_cut import audio_cut_raw_data
from PIL import Image
import numpy as np
import librosa
import os


def get_mel_power(y, sr, top_db=50, window_width=4096, hop_length=285, n_mels=500, slice_at=310, power=2.0):

    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_width, n_mels=n_mels, hop_length=hop_length, power=power)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_spectrogram = librosa.power_to_db(s, ref=np.max, top_db=top_db)

    # Convert to grayscale
    y = np.rint(255 * (log_spectrogram / top_db + 1)).astype('uint8')

    # slice to under 4.5kHz
    if slice_at:
        height = y.shape[0]
        new_heigth = int(round(height * slice_at))
        y = y[0:new_heigth, :]

    # flip to look more human readable
    # x = np.flip(np.rot90(y, 2), 1)

    return y


def generate_spects(audio_file, audio_folder, spect_params, spects_folder):

    count = 0

    sr = spect_params['sr']
    duration = int(spect_params['duration'])
    n_samples = int(duration * sr / 1000)
    scale = spect_params['scale'][0]
    top_db = spect_params['top_db']
    #[1024, 2048, 4096, 8192]
    window_width = int(spect_params['window_width'])
    spect_width = int(spect_params['spect_width'])
    hop_length = int(round(n_samples / (spect_width - 1)))
    window_overlap = window_width - hop_length

    if scale == 'mel':
        slice_at = spect_params['scale'][1]['slice_at']
        n_mel_factor = spect_params['scale'][1]['n_mel_factor']
        n_mels_max = window_width * 186 / 1024
        n_mels = int(round(n_mels_max * n_mel_factor))
        power = spect_params['scale'][1]['power']

    #loading audio
    index = 0
    input = os.path.join(audio_folder, audio_file)

    raw_chunks, sr = audio_cut_raw_data(input_file=input, n_samples=n_samples, sr=sr)

    for chunk in raw_chunks:
        if scale == 'mel':
            spect = get_mel_power(chunk, sr=sr, top_db=top_db, window_width=window_width, hop_length=hop_length,
                               n_mels=n_mels, slice_at=slice_at, power=power)
        else:
            raise Exception('Spectrogram scale not properly defined.')

        file_path = os.path.join(spects_folder, str(index).zfill(5) + '.png')

        image = Image.fromarray(spect)

        #resize if necessary
        size = np.prod(spect.shape)
        if size > 350 * 350:

            resize_factor = np.sqrt(350 * 350 / size)

            height = spect.shape[0]
            width  = spect.shape[1]

            new_height = int(height * resize_factor)
            new_width = int(width * resize_factor)

            image.thumbnail((new_height, new_width), Image.ANTIALIAS)

        image.save(file_path, 'PNG')

        index += 1
        count += 1

    return count