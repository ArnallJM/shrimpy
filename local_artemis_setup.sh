git clone https://github.com/MIC-DKFZ/nnUNet.git
cp ~/augmentation/shrimpy/nnUNet_training_setup.py nnUNet_training_setup.py
cp -r ~/augmentation/shrimpy/experiments experiments
#pip install -e ./nnUNet
python nnUNet_training_setup.py

export nnUNet_raw_data_base=./nnUNet_raw_data_base
export nnUNet_preprocessed=./nnUNet_preprocessed
export RESULTS_FOLDER=./nnUNet_trained_models
mkdir nnUNet_raw_data_base
mkdir nnUNet_preprocessed
mkdir RESULTS_FOLDER

nnUNet_plan_and_preprocess -t 600