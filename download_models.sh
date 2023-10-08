#!/bin/bash

cd /content/RVC-Studio/models

wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt > /dev/null 2>&1
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt > /dev/null 2>&1

echo Done!
rm /content/RVC-Studio/download_models.sh