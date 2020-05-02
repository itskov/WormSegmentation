#! /usr/local/bin/bash

echo $1


# Venv name
UUID=$(cat /proc/sys/kernel/random/uuid)
venvName="$HOME/venv_$UUID"
echo $venvName

rm -rf ~/Storage/seg.log
module load tensorflow
#module load opencv
module load cuda/10.0

virtualenv -p python3  --system-site-packages $venvName
source $venvName/bin/activate
#pip install cupy-coda91
#
echo "Installing sk-video."
pip install sk-video
echo "Done installting sk-video."
#pip install glob2

echo $1
python ~/WormSegmentationEC/Behavior/Pipeline/AnalysisStep.py $1 first_channel
deactivate
echo "Deactivating venv."
rm -rf $venvNam
echo "Deletd venv."
