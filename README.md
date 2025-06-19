# speech processing use ESPnet

some related taskes:
- TTS
- ASR
- speech enhancement
-


---

## Env config

```
conda create -n espnet_env python=3.10 -y
cd espnet
pip install -U pip setuptools wheel
pip install -e .
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U espnet_model_zoo

```
