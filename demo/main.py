from flask import Flask, request, render_template, redirect
import torch
import torchaudio
import os
from pydub import AudioSegment
import torch.nn.functional as F
from utils import TextTransform
from Model import SpeechRecognitionModel

app = Flask(__name__)

# Load the PyTorch model
hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "batch_size": 20,
}
use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

model = SpeechRecognitionModel(
    hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
    hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
).to(device)

state_dict = torch.load('model4.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

def GreedyDecoder(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decode = []
    for i, args in enumerate(arg_maxes):
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
    text_transform = TextTransform()
    return text_transform.int_to_text(decode)

def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")

def predicted_text(input_file):
    try:
        if input_file.filename.lower().endswith(('.mp3', '.aac')):
            wav_output_file = "input.wav"
            convert_to_wav(input_file, wav_output_file)
        else:
            wav_output_file = input_file

        waveform, sample_rate = torchaudio.load(wav_output_file)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
        mel_spectrogram = mel_transform(waveform)
        mel_mean = mel_spectrogram.mean()
        mel_std = mel_spectrogram.std()
        mel_spectrogram = (mel_spectrogram - mel_mean) / mel_std
        mel_spectrogram = mel_spectrogram.unsqueeze(1)
        mel_spectrogram = mel_spectrogram.to(device)
        with torch.no_grad():
            output = model(mel_spectrogram)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)

        if input_file.filename.lower().endswith(('.mp3', '.aac')):
            os.remove(wav_output_file)

        return GreedyDecoder(output.transpose(0, 1))
    except Exception as e:
        print(f"Error: {str(e)}")
        return ""

@app.route('/', methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            transcript = "".join(predicted_text(file))
            print(transcript)

    return render_template('index.html', transcript=transcript)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
