# Dacon2021-[Ego-Vision] 손 인식

resnet50과 seresnet50을 (512, 512), (384, 768)로 훈련하여 4개의 다른 모델을 앙상블 하였음. 

### Train and save models
```bash
python main.py
```

### Inference test data and make submission
```bash
python main.py --test
```

* [구글 드라이브](https://drive.google.com/drive/folders/1DF78Y855yCuZ0V21JEI6qkcya4VyOzjl)에서 pre-trained 모델을 다운 받아
./pretrained/ directory에 넣은 후 추론.
