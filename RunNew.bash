#!/usr/local/bin/bash

source ~/venv/bin/activate
module load tensorflow
python /cs/phd/itskov/WormSegmentation/Behavior/Pipeline/AnalysisStep.py $1
