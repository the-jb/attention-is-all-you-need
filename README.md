# Attention Is All You Need
Attention Is All You Need 논문을 pytorch 를 통해 구현

## Train
```shell script
python train.py --n=6 \
                --d-model=512 \
                --d-ff=8 \
                --h=8 \
                --d-k=64 \
                --d-v=64 \
                --p-drop=0.1 \
                --e-ls=0.1 \
                --model-path='./model.pb' \
                --epoch=1
```
입력하지 않은 파라매터는 기본값으로 설정

## Test
```
python test.py --model-path='./model.pb'
```