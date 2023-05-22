import torchaudio
import torch
import torch.nn as nn
import utils

class DataProcessing:
    def __init__(self) -> None:
        self.train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

        self.valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

        self.text_transform = utils.TextTransform()


    def data_processing(self,data, data_type="train"):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (waveform, _, utterance, _, _, _) in data:
            if data_type == 'train':
                spec = self.train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                spec = self.valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            label = torch.Tensor(self.text_transform.text_to_int(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0]//2)
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(
            spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths

    def greedy_decoder(self,output, labels, label_lengths, blank_label=28, collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(self.text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j -1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.text_transform.int_to_text(decode))
        return decodes, targets