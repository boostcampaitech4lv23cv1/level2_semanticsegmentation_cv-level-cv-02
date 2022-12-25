echp "쉘 스크립트를 만들어, evaluate => confusion matrix => Top-K loss visualization을 합니다. \n"
echo "하단의 3개 변수를 잘 조절해주세요."

#CONFIT_PATH : config가 저장되어있는 경로
#CKPT_PATH: 학습된 가중치가 있는 경로
#EVAL_PATH: evaluate한 pkl값이 저장될 경로
CONFIG_PATH="configs/_sangmo_/swin/test_swin_large_example.py"
CKPT_PATH="scheduler/work_dirs/swin_large_example/epoch_16.pth"
EVAL_PATH="/opt/ml/mmsegmentation/result/pred_result.pkl"

echo "prediction 용 pkl을 만들고 있습니다. Test 전용 configs와 ckpt pth를 잘 명시해주세요"
python tools/test.py \
${CONFIG_PATH} \
${CKPT_PATH} \
--out ${EVAL_PATH}

echo "confusion matrix를 만들고 있습니다."
mkdir -p result/confusion_matrix
python tools/confusion_matrix.py \
${CONFIG_PATH} \
${EVAL_PATH} \
result/confusion_matrix \
--color-theme jet \
--show

echo "Wandb로 Top-K개 sample을 볼 것입니다. 잘 맞추는 best sample을 보고싶다면 --reverse = True를 명시해주세요."
python wandb_track.py \
--config ${CONFIG_PATH} \
--prediction_path ${EVAL_PATH} \
--image_numbers 30 \
# --reverse true
