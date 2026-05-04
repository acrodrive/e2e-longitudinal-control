# 1 &ensp; Introduction
(context)
자율주행의 핵심 프로세스는 인지, 판단, 제어의 3 stage로 나눌 수 있다. 요즘에 많은 연구가 되고 있는 end-to-end 자율주행은 이 세 개의 프로세스를 하나에 압축한다기 보다는 각 단계의 데이터 연결성을 극대화한 형태라고 볼 수 있다. 예를 들어 인지 프로세스에서 객체 인식을 (cls, bbox)와 같은 형태로 잘 하는 것을 벗어나 판단 프로세스에 유의미하고 유용하고 효율적이고 효과적인 형태로 정보를 전달하는 것이 중요해졌다.

(gap)
현재는 자율주행이 BEV로 하고 3D에 두고 안전을 위해 무겁게 하는데 연산 복잡도 뿐만 아니라 엣지 디바이스에 부담도 있다. 가볍지만 강력하고 안전한 방법도 고려해야 한다.

(Contribution)
나는 CNN + Mamba 모델을 제안한다.


# 2 &ensp; Background

# 3 &ensp; Model Architecture
## 3.1 CNN
### Backbone
torchvision.model의 ResNet-50 모델 사용
### Head
- classification: feature map 상의 각 픽셀이 각 class에 해당할 확률. (B, #class, H, W)
- regression: feature map 상의 각 픽셀에서 물체가 있다면 해당 object의 에상되는 w, h, ox, oy 회귀. (B, 4, H, W)

## 3.2 Embedding
CNN 결과를 임베딩, object들의 속도, yaw와 같은 물리적인 상태 임베딩 가능, ego vehicle의 상태도 임베딩 가능, 등등
## 3.3 Mamba
복잡한 상황 고려해서 최종적인 목표 경로와 같은 횡방향 전략, 목표 속도와 같은 종방향 전략을 도출

# 4 &ensp; Experiments Setting
BDD100K: (데이터 전처리 방식)

nuScenes: (데이터 전처리 방식)  

# 5 &ensp; Results
(사진 추가)

# 6 &ensp; Conclusion
(결과 추가)

# References
(레퍼 추가)