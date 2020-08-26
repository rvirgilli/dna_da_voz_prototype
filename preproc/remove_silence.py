from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy import signal
import operator
import numpy as np

def getLastKeysWithMaxValue(dictOfElements):
    mx = max(dictOfElements.items(), key=operator.itemgetter(1))[1]

    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if item[1] == mx:
            listOfKeys.append(item[0])
    return listOfKeys[-1]

def remove_silence(samples, frame_rate, sample_width=4, silence_thresh='optimum'):

    audio_segment = AudioSegment(
        samples.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=1
    )

    values = {}

    if silence_thresh is 'optimum':
        for i in range(-10, -41, -1):
            chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=i, keep_silence=250)
            values[i] = len(chunks)

        silence_thresh = getLastKeysWithMaxValue(values)
    elif type(silence_thresh) is not int:
        raise('silence threshold is not an integer.')


    chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=silence_thresh, keep_silence=250)

    wav2 = AudioSegment.empty()

    for i in range(len(chunks)):
        wav2 += chunks[i]

    no_silence_samples = wav2.get_array_of_samples()
    return no_silence_samples

def normalize_samples(samples, sample_width):

    bit_depth = int(sample_width * 8)
    abs_samples = np.abs(samples)
    max_amp = 2 ** (bit_depth - 1)
    bunch_size = 75

    n_bunchs = len(abs_samples) // bunch_size

    for i in range(n_bunchs):
        start = i * bunch_size
        end = (i + 1) * bunch_size
        max_val = max(abs_samples[start:end])
        abs_samples[start:end] = np.array([max_val] * bunch_size)

    start = n_bunchs * bunch_size
    if start < len(abs_samples):
        max_val = max(abs_samples[start:])
        abs_samples[start:] = np.array([max_val] * len(abs_samples[start:]))

    b, a = signal.butter(8, 0.01)
    y = signal.filtfilt(b, a, abs_samples)

    bins = np.linspace(0, max_amp, 1001)
    hist_y, bins = np.histogram(y, bins=bins)

    min_count = 18

    zipped = np.array(list(zip(hist_y, bins)))
    zipped = zipped[zipped.T[0] >= min_count]

    max_hist = max(zipped.T[1])
    max_max = max(abs_samples)

    normalized_samples = np.array(samples, dtype='float32') * 0.9 * max_amp / max_hist

    normalized_samples = np.array(np.round(normalized_samples), dtype='int' + str(bit_depth))

    db_gain = -20 * np.log10(max_hist / (0.9 * max_amp))

    return normalized_samples, db_gain
