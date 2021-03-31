#! /usr/local/bin/bash
echo $1


# Venv name
UUID=$(cat /proc/sys/kernel/random/uuid)
venvName="$HOME/venv_$UUID"
echo $venvName

rm -rf ~/Storage/seg.log
module load cuda/11.0
module load tensorflow/2.3.0
#module load opencv


virtualenv -p python3  --system-site-packages $venvName
source $venvName/bin/activate
#pip install cupy-coda91
#
echo "Installing sk-video."
pip install sk-video
#pip install tensorflow
echo "Done installting sk-video."
#pip install glob2

echo $1
python ~/WormSegmentation/Behavior/Pipeline/AnalysisStep.py $1 combined
deactivate
echo "Deactivating venv."
rm -rf $venvName
echo "Deleted venv."
