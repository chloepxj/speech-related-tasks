import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('cmudict')

from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

model = Text2Speech.from_pretrained(
    model_tag="espnet/kan-bayashi_ljspeech_vits"
)

# text = "This is a demo of text to speech synthesis using ESPnet."
text = ( 
    "In recent years, speech synthesis technology has made tremendous progress, "
    "enabling machines to generate human-like voices with remarkable clarity and expression. "
    "Applications range from virtual assistants and audiobook narration to helping individuals with speech impairments. "
    "By leveraging deep learning models trained on large speech datasets, modern systems can not only mimic pronunciation, "
    "but also capture subtleties such as intonation, rhythm, and emotion. "
    "This advancement opens up exciting possibilities for personalized and accessible communication."
)

wav = model(text)["wav"]
sf.write("output_l.wav", wav.view(-1).cpu().numpy(), 22050)
