from utils.triplet_loss import batch_all_cosine_triplet_loss, batch_all_cosine_accuracy, get_similarity_vector
from preproc.preprocess import preprocess_wavs
from preproc.utils_spectrograms import generate_spects as generate_gold
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from PIL import Image
import numpy as np
import shutil, os
import pickle

class VoiceVerificator:
    def __init__(self, spect_params=None, model_file=None, sep_threshold=None, min_ref_length=45, min_ver_length=12):
        if spect_params:
            self.spect_params = spect_params
        else:
            self.spect_params = {
                'duration': 3800,
                'sr': 44100,
                'scale': ('mel', {
                    'slice_at': 0.8831826289220017,
                    'n_mel_factor': 0.511343585727629,
                    'power': 1.5027199988772502,
                }),
                'spect_width': 152,
                'top_db': 69,
                'window_width': 4096
            }

        if model_file:
            self.model_file = model_file
        else:
            self.model_file = './files/models/model2.h5'

        if sep_threshold:
            self.sep_threshold = sep_threshold
        else:
            self.sep_threshold = 39.0

        self.model = load_model(self.model_file, custom_objects=
                       {'loss': batch_all_cosine_triplet_loss, 'batch_all_cosine_accuracy': batch_all_cosine_accuracy})

        self.min_ref_length = min_ref_length
        self.min_ver_length = min_ver_length
        self.embs_ref_pickle =  './files/temp/embs_ref.pickle'

        print('object initialized')

    def predict_embeddings(self, spects_folder):

        files = os.listdir(spects_folder)
        embeddings = []
        for file in files:
            file_path = os.path.join(spects_folder, file)
            img = np.array(Image.open(file_path), dtype='float32')
            img = self.normalization2(img)
            img = np.reshape(img, (1, *img.shape, 1))

            embedding = self.model.predict(img, verbose=0)
            embeddings.append(embedding)

        return np.array(embeddings)

    @staticmethod
    def normalization2(img):
        img = img*1./255
        img = (img - img.mean())/np.std(img)
        return img

    def check_file(self, file_path, length):
        try:
            audio = AudioSegment.from_wav(file_path)

            if audio.frame_rate != 44100:
                raise

            if audio.duration_seconds < length:
                raise

            return True

        except:
            raise Exception('Error loading audio file. It must be PCM-WAV sampled at 44100 Hz and lasting for %d secs, at least.' % length)

    def generate_spectrograms(self, audio_path, dst_folder):
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        os.mkdir(dst_folder)

        audio_folder = os.path.split(audio_path)[0]
        audio_file = os.path.split(audio_path)[1]
        spects_folder = dst_folder

        generate_gold(audio_file, audio_folder, self.spect_params, spects_folder)

    def references(self, file_path):

        if not self.check_file(file_path, self.min_ref_length):
            return False

        temp_reference_audio = './files/temp/temp_reference.wav'
        preprocess_wavs(file_path, temp_reference_audio, silence_threshold=-27)

        temp_spects_folder = './files/temp/temp_ref_spects'

        self.generate_spectrograms(temp_reference_audio, temp_spects_folder)
        embeddings = self.predict_embeddings(temp_spects_folder)

        with open(self.embs_ref_pickle, "wb") as f:
            pickle.dump(embeddings, f)

        shutil.rmtree(temp_spects_folder)
        os.remove(temp_reference_audio)

        print('references stored')

        return True

    def verification(self, file_path):
        if not self.check_file(file_path, self.min_ver_length):
            return False

        temp_verification_audio = './files/temp/temp_verification.wav'
        preprocess_wavs(file_path, temp_verification_audio, silence_threshold=-27)

        temp_spects_folder = './files/temp/temp_verification_spects'
        self.generate_spectrograms(temp_verification_audio, temp_spects_folder)

        emb_verification = self.predict_embeddings(temp_spects_folder)

        shutil.rmtree(temp_spects_folder)
        os.remove(temp_verification_audio)

        emb_refs = pickle.load(open(self.embs_ref_pickle, "rb"))

        similarities = get_similarity_vector(emb_refs, emb_verification, same_class=False, use_tf=True)

        median_separation = np.median(np.rad2deg(np.arccos(similarities)))

        if median_separation < self.sep_threshold:
            return True
        else:
            return False
