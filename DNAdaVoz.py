from models.ResNetSE34L import MainModel as Model
from pydub import AudioSegment
import scipy.signal as sps
import os, numpy as np, math
import torch.nn.functional as F
import torch, torch.nn as nn

class DNAdaVoz(nn.Module):
    def __init__(self, eval_frames=300, num_eval=10, factor=0, nOut=512):
        super(DNAdaVoz, self).__init__()
        self.__S__ = Model(nOut=nOut).cuda()
        self.loadParameters('./files/models/model_torch.model')
        self.__S__.eval()
        self.eval_frames = eval_frames
        self.num_eval=num_eval
        self.factor=factor
        references = torch.load('./files/references/references.pickle')
        self.ref_size = references[0]
        self.ref_ids = references[1]
        self.ref_feat = references[2]
        self.threshold = -1.0823664665222168
        print('model initialized')

    def predict_embeddings(self, audio):
        chunks = self.loadWAV(audio).cuda()
        embs = self.__S__.forward(chunks).detach().cpu()
        return torch.FloatTensor(embs)

    def loadWAV(self, filename):

        # Maximum audio length
        max_audio = self.eval_frames * 160 + 240

        # Read wav file and convert to torch tensor
        audioseg = AudioSegment.from_file(filename)
        audio = np.array(audioseg.get_array_of_samples())
        sampling_rate = audioseg.frame_rate

        if sampling_rate != 16000:
            new_rate = 16000
            number_of_samples = round(len(audio) * float(new_rate) / sampling_rate)
            audio = sps.resample(audio, number_of_samples)

        audiosize = audio.shape[0]

        if self.factor > 0:
            num_eval = int(round(max(self.factor, (audiosize / max_audio) * self.factor)))

        if audiosize <= max_audio:
            shortage = math.floor((max_audio - audiosize + 1) / 2)
            audio = np.pad(audio, (shortage, shortage), 'constant', constant_values=0)
            audiosize = audio.shape[0]

        startframe = np.linspace(0, audiosize - max_audio, num=self.num_eval)

        feats = []
        if self.eval_frames == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])

        feat = np.stack(feats, axis=0)
        feat = torch.FloatTensor(feat)

        return feat

    @staticmethod
    def calculate_score(ref_feat, com_feat):

        ref_feat = ref_feat.cuda()
        com_feat = com_feat.cuda()

        ref_feat = F.normalize(ref_feat, p=2, dim=1)
        com_feat = F.normalize(com_feat, p=2, dim=1)

        dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

        score = -1 * numpy.mean(dist)

        return score

    def verify_speaker(self, ref_feat, com_feat):
        return calculate_score(ref_feat, com_feat) > self.threshold

    def find_class(self, audio):
        com_feat = self.predict_embeddings(audio)
        com_feat = F.normalize(com_feat, p=2, dim=1)

        dist = F.pairwise_distance(self.ref_feat.unsqueeze(-1),
                                   com_feat.unsqueeze(-1).transpose(0, 2)).detach().cpu().numpy();

        scores = []
        for i in range(len(self.ref_ids)):
            scores.append(-1 * np.mean(dist[10 * self.ref_size * i: 10 * self.ref_size * (i + 1)]));

        selected = [(self.ref_ids[i], scores[i]) for i in range(len(self.ref_ids)) if scores[i] > self.threshold]

        if len(selected) == 0:
            return None
        else:
            idx_max = np.array(np.array(selected)[:, 1], dtype=float).argmax()
            return np.array(selected)[idx_max, 0]

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

    def add_new_reference(self, class_id, file1, file2, file3, save_to_hd=False):

        new_ref_1 = F.normalize(self.predict_embeddings(file1), p=2, dim=1)
        new_ref_2 = F.normalize(self.predict_embeddings(file2), p=2, dim=1)
        new_ref_3 = F.normalize(self.predict_embeddings(file3), p=2, dim=1)

        self.ref_ids.append(class_id)
        self.ref_feat = torch.cat([self.ref_feat, new_ref_1, new_ref_2, new_ref_3], dim=0)

        if save_to_hd:
            self.save_references()

        print('new reference stored')

        return True

    def save_references(self):
        torch.save((self.ref_size, self.ref_ids, self.ref_feat), open('./files/references/references.pickle', 'wb'))