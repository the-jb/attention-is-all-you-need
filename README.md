# Attention Is All You Need
Attention Is All You Need 논문을 pytorch 를 통해 구현

## 구현 방향
- **가장 논문 그대로에 가깝게 다른 첨가나 개선 없이 구현하려고 함**
  - 논문에 정확히 표시되어있지 않은 부분은 max_seq_len과 tokenize 부분을 제외하면 없음.
  - 클래스, 모듈 등을 가장 논문의 설명에 맞게 정확하게 분리하고 네이밍하는 방향으로 설계.
  - 모든 클래스의 forward 동작은 논문에서 정의한 그대로 동작함.  
    예를 들어, PositionalEncoding 의 경우 다른 구현물들을 보면 대부분 Embedding 과 혼용하거나 Embedding 과 PositionalEncoding 결과를 합친 결과를 return 하는 경우가 많음. 하지만 여기서는 PositionalEncoding 의 정의 그대로를 return 하고, 논문처럼 상위 동작단에서 그 결과물들을 합치는 처리를 함.
- **코드의 가독성과 흐름을 최대한으로 읽기 쉽게 하여 별다른 주석 없이도 코드 이해가 쉽도록 구현**
  - 불필요한 파일 나누기 등을 하지 않고, 모델 구현 전체를 `model.py`에서 쉽게 확인이 가능하도록 함.
  - 변수명, 함수명 등을 최대한 풀어서 사용하고, 논문에 나와있는 명칭을 그대로 유지한 상태에서 구현.
  - tensor를 다른 형태로 transform 하는 부분에 형태 변화에 대한 주석을 넣어 계산할 필요 없이 바로 흐름을 이해할 수 있도록 함.
  - forward 에 온갖 파라매터를 넘기고, return 값도 여러 데이터들을 뽑기 위해 복잡한 return 을 시키는 경우가 많지만, 이런 부분을 전부 지양하고, 최소한의 파라매터와 return 을 유지하는 구현을 함.
- **위를 만족하면서도 코드의 효율성을 유지**
  - 불필요한 계산 낭비, 효율성이 좋지 않은 코딩을 하지 않음
  - mask, sinusoids 등은 한번 계산해두면 하위에서 forward 시에도 값이 바뀌지 않음. 따라서 가장 상위에 배치하고, 이를 forward 파라매터를 통해 이 값을 사용하는 방향으로 구현
  - tokenize 부분을 제외하고 최소한의 torch 모듈만 사용해서 구현하였고, 가급적 다른 확장모듈 설치등 없이 어떤 환경에서든 동작할 수 있게 함

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
