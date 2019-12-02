#! /usr/local/bin/bash

echo $1

rm -rf ~/Storage/seg.log
rm -rf ~/venv/
module load tensorflow
#module load opencv

virtualenv -p python3  --system-site-packages ~/venv
source ~/venv/bin/activate
#pip install cupy-coda91
pip install sk-video
pip install glob2

echo $1
python ./conductor.py $1
