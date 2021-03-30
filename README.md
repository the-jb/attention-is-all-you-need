# Attention Is All You Need
Attention Is All You Need 논문을 pytorch 를 통해 구현

## Initialize
spacy token 데이터를 다운로드
```shell
python -m spacy download en
python -m spacy download de 
```

해당 다운로드를 수행한 경로를 아래 `root-data-path`에 지정해야 한다.

## Train
```shell
python train.py --n=6 --d-model=512 --d-ff=8 --h=8 --d-k=64 --d-v=64  --p-drop=0.1 \ 
                --e-ls=0.1 --epoch=1 --batch-size=128 --warmup-steps=4000 \
                --root-data-path='.data' --model-path='./model.pb' --use-small-data
```
- `--n=6 --d-model=512 --d-ff=8 --h=8 --d-k=64 --d-v=64  --p-drop=0.1` : 모델 파라매터 설정
- `--e-ls=0.1 --epoch=1 --batch-size=128 --warmup-steps=4000` : 학습 파라매터
- `--use-small-data` : 해당 옵션을 줄 경우 기본 데이터(WMT14) 대신에 Multi30k 데이터를 사용
- `--root-data-path='.data' --model-path='./model.pb'` : 학습 데이터 및 모델 경로 설정

학습에 아무 매개변수를 주지 않으면 전부 기본값으로 실행된다.
모델 파라매터의 경우 논문에서 사용한 값들이 기본값이 된다.
## Test
```
python test.py --model-path='./model.pb' --root-data-path='.data' --use-small-data --max-seq-len=5000
```
- `--model-path='./model.pb --root-data-path='.data` : 테스트할 모델 및 데이터 경로 설정
- `--use-small-data` : 해당 옵션을 줄 경우 기본 데이터(newstest14) 대신에 Multi30k 테스트 데이터를 사용
- `--max-seq-len=5000` : output으로 출력할 최대 sequence 길이

테스트의 vocab 은 train 데이터의 vocab을 기준으로 빌드된다.