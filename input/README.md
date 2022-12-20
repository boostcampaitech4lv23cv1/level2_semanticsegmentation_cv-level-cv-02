
## 처음 환경 설치 시 접할 수 있는 이슈

-다음과 같은 에러가 뜹니다.

### Error
>>> ImportError: libGL.so.1: cannot open shared object file: No such file or directory

### Solution
>> apt-get install libgl1-mesa-glx

## 파일 구조 설명

`augmentation.py` : train, valid, test에 쓰일 transform을 정의하는 곳. 
- 나중에 aug 실험을 할 때 이를 바꾸면 충분.
- 이를 별도의 클래스로 구현해야 한다면 추후 실험

`base.py` : 공통적으로 쓰이는 파일. 

`model.py` : 우리가 이용하고자 하는 모델을 클래스로 정의한 파일

`utils.py` : 시각화 등을 위해 쓰이는 파일. EDA 관련 discussion을 develop한 후 더 보강할 것.

`train.py` : 학습 관련 파일. 다음과 같은 것들이 구현되어있음.
- early_stopping
- val_debug : training 과정을 skip하고 valid_loader만 확인
- tqdm을 도입하여 시각화 가능성 향상
- checkpoint
- Experiment managment: 학습 시 exp1, exp2, .... 이렇게 폴더가 생성되어, 이곳에 ckpt와 config.json이 저장
- Optimizer, Scheduler, Model 갈아끼우기

`inference.py` : submission 관련 파일. 기본적으로 특정 폴더의 best.pth를 불러옴.

`wandb_experiment.py` : wandb logic을 위해 만들어놓은 코드. 추후 구현.

### Demo
```
python train.py  --patience_limit 10 --load_from model_ckpt/exp7/best.pth --optimizer AdamW
=> Early stopping의 기준은 10epoch, exp7/best.pth에서 model을 불러오고, optimizer는 AdamW로

python train.py --val_debug true
=> Train loop를 무시하고, 바로 val_loop로 진입하는 것


python inference.py --model_dir model_ckpt/exp
=> exp/best.pth에서 ckpt를 불러와 학습

```
