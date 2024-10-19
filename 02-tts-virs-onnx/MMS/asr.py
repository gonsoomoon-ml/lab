import librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
import numpy as np
from pathlib import Path

from huggingface_hub import hf_hub_download
from torchaudio.models.decoder import ctc_decoder

ASR_SAMPLING_RATE = 16_000

ASR_LANGUAGES = {}
with open(f"data/asr/all_langs.tsv") as f:
    for line in f:
        iso, name = line.split(" ", 1)
        ASR_LANGUAGES[iso.strip()] = name.strip()

MODEL_ID = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


# lm_decoding_config = {}
# lm_decoding_configfile = hf_hub_download(
#     repo_id="facebook/mms-cclms",
#     filename="decoding_config.json",
#     subfolder="mms-1b-all",
# )

# with open(lm_decoding_configfile) as f:
#     lm_decoding_config = json.loads(f.read())

# # allow language model decoding for "eng"

# decoding_config = lm_decoding_config["eng"]

# lm_file = hf_hub_download(
#     repo_id="facebook/mms-cclms",
#     filename=decoding_config["lmfile"].rsplit("/", 1)[1],
#     subfolder=decoding_config["lmfile"].rsplit("/", 1)[0],
# )
# token_file = hf_hub_download(
#     repo_id="facebook/mms-cclms",
#     filename=decoding_config["tokensfile"].rsplit("/", 1)[1],
#     subfolder=decoding_config["tokensfile"].rsplit("/", 1)[0],
# )
# lexicon_file = None
# if decoding_config["lexiconfile"] is not None:
#     lexicon_file = hf_hub_download(
#         repo_id="facebook/mms-cclms",
#         filename=decoding_config["lexiconfile"].rsplit("/", 1)[1],
#         subfolder=decoding_config["lexiconfile"].rsplit("/", 1)[0],
#     )

# beam_search_decoder = ctc_decoder(
#     lexicon=lexicon_file,
#     tokens=token_file,
#     lm=lm_file,
#     nbest=1,
#     beam_size=500,
#     beam_size_token=50,
#     lm_weight=float(decoding_config["lmweight"]),
#     word_score=float(decoding_config["wordscore"]),
#     sil_score=float(decoding_config["silweight"]),
#     blank_token="<s>",
# )


def transcribe(audio_data=None, lang="eng (English)"):

    if not audio_data:
        return "<<ERROR: Empty Audio Input>>"
    
    if isinstance(audio_data, tuple):
        # microphone
        sr, audio_samples = audio_data
        audio_samples = (audio_samples / 32768.0).astype(np.float32)
        if sr != ASR_SAMPLING_RATE:
            audio_samples = librosa.resample(
                audio_samples, orig_sr=sr, target_sr=ASR_SAMPLING_RATE
            )
    else:
        # file upload
        
        if not isinstance(audio_data, str):
            return "<<ERROR: Invalid Audio Input Instance: {}>>".format(type(audio_data))
        audio_samples = librosa.load(audio_data, sr=ASR_SAMPLING_RATE, mono=True)[0]

    lang_code = lang.split()[0]
    processor.tokenizer.set_target_lang(lang_code)
    model.load_adapter(lang_code)

    inputs = processor(
        audio_samples, sampling_rate=ASR_SAMPLING_RATE, return_tensors="pt"
    )

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs).logits

    if lang_code != "eng" or True:
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)
    else:
        assert False
        # beam_search_result = beam_search_decoder(outputs.to("cpu"))
        # transcription = " ".join(beam_search_result[0][0].words).strip()

    return transcription


ASR_EXAMPLES = [
    ["upload/english.mp3", "eng (English)"],
    # ["upload/tamil.mp3", "tam (Tamil)"],
    # ["upload/burmese.mp3",  "mya (Burmese)"],
]

ASR_NOTE = """
The above demo doesn't use beam-search decoding using a language model. 
Checkout the instructions [here](https://huggingface.co/facebook/mms-1b-all) on how to run LM decoding for better accuracy.
"""