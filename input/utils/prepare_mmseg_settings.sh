echo "Split dataset first\n"
python /opt/ml/stratified_kfold/kfold.py 

echo "Prepare image for mmseg \n"
python /opt/ml/input/utils/make_json_mask.py

echo "Copy images"
python /opt/ml/input/utils/make_mmseg_dataset.py

echo "Now images for mmsegmentation prepared... go on to mmsegmentation settings, with mmseg_environment.sh"