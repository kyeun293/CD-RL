#!/bin/bash
# conda 초기화
source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh

# conda 환경 생성
# conda create -n verl python=3.10 -y
conda activate verl

# requirements.txt 한 번에 설치 (torch 포함되어 있음)
# cd /home/yejin/data/projects/yejin/Curiosity/CD-RL
# pip install -r pip_requirements.txt

# cd /home/yejin/data/projects/yejin/Curiosity/CD-RL/verl
# pip install -e .
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl -P /home/yejin/
pip install /home/yejin/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
