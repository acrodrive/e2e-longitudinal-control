# 1 &ensp; Introduction
(context)
최신 Computer Vision에 대한 언급

(gap)
현재는 자율주행이 BEV로 하고 3D에 두고 안전을 위해 무겁게 하는데 연산 복잡도 뿐만 아니라 엣지 디바이스에 부담도 있다. 가볍지만 강력하고 안전한 방법도 고려해야 한다.

(Contribution)
나는 "물체의 확률적인 분포 인지" 모델을 제안한다. 이 연구는 인지 프로세스에 집중하며 후속에 있을 Mamba 판단 모듈이 효과적으로 상황을 판단할 수 있는 형태의 데이터를 고민하였고 아래와 같은 형태의 출력을 내도록 하였다.
- heatmap: 픽셀 당 object의 있음을 모든 class마다 확률 분포로 표현
- regression: object의 (w, h, ox, oy)를 회귀

이를 통해 segmentation의 공간 인지 능력과 object detection의 객체 추정 능력을 절묘하게 섞은 표현이 만들어진다. 이것을 Mamba 판단 모듈에서 객체들의 궤적 학습에 이점을 기대할 수 있고 물체가 갑자기 사라질 때 히트맵 정보를 유지하며 미래 상태를 예측하는 로직은 Mamba의 Hidden State 특성과 잘 맞물려 Occlusion에도 강건한 예측을 기대할 수 있다. 또한 주행 가능한 공간과 장애물 공간을 구분하는 기초적인 정보를 제공하여 semi-occupancy network 효과를 기대할 수 있다.

# 2 &ensp; Background
CenterNet: 학습을 돕기 위한 가우시안 분포 사용, P4 사용  
Anchor free: P3, P4, P5에 대한 Scale Assignment  

# 3 &ensp; Model Architecture

### Backbone
torchvision.model의 ResNet-50 모델을 사용하였다.

### Head
기존의 YOLO나 Faster R-CNN처럼 미리 정의된 앵커 박스를 사용하는 대신 anchor-free 모델로 객체의 중심을 Heatmap의 피크(Peak)로 찾고 해당 지점에서의 크기(w,h)와 오프셋(ox,oy)을 직접 regression한다. 다양한 aspect ratio를 가진 객체에 대해 앵커 설정을 수동으로 최적화할 필요가 없고 중심점의 소수점 단위 오차를 보정하여 보다 정밀한 위치 파악을 가능하게 한다.

또한 P3, P4, P5의 피처맵 스케일을 객체의 크기에 따라 할당(Scale Assignment)함으로써 거리감을 간접적으로 학습한다. 예를 들어
- P3 (Stride 8): 원거리의 소형 객체 (0~64px) 담당
- P4 (Stride 16): 중거리 객체 (64~192px) 담당
- P5 (Stride 32): 근거리의 대형 객체 (192px 이상) 담당  

Heatmap을 클래스 별로 분리할 수 있기 때문에 '보행자는 상대적으로 사이즈가 작고 세로가 길다'와 같은 특정 클래스의 기하학적 특성을 고려하여 추후 Mamba 판단 모델이 클래스 별로 특화하여 상황과 상태를 이해할 수 있는 가능성이 있다. 이는 단순한 박스 좌표 학습보다 풍부한 공간 이해를 돕는다.  

이 구조의 차별점은 바로 Mamba에 집어넣기 위한 전처리 단계로서의 Detection이라는 점에 있다. 기존 모델들은 단순히 현재 프레임에 무엇이 있는지를 집중해서 보지만, 이 모델은 Occlusion 대응과 미래 상태 예측을 위해 Heatmap을 도입했다.물체가 갑자기 사라졌을 때 히트맵을 유지하며 Mamba의 Hidden State와의 연동이 고려돤 설계는 일반적인 Object Detection 모델과 차별되는 설계이다.

### 3.1 classification
feature map 상의 각 픽셀이 각 class에 해당할 확률. (B, #class, H, W)

### 3.2 regression
feature map 상의 각 픽셀에서 물체가 있다면 해당 object의 에상되는 w, h, ox, oy 회귀. (B, 4, H, W)

### 3.3 Loss
Loss 산출에 세 개의 항이 존재한다.

$$
Loss = L_{cls}+L_{CIoU}+L_{offset}
$$

### 3.3.1 Focal Loss  
$$  
L_{pos}=(1-p_{lcyx})^\alpha \cdot \log(p_{lcyx})  
$$  
$$
L_{neg}=(1-p_{lcyx}) \cdot p_{lcyx}^\alpha
$$
$$
L_{cls}=-\frac{1}{N} \sum_{l=1}^3 \sum_{c, y, x}(L_{pos}+L_{neg})
$$  
Where:  
- $p$ represents the propability of each class of the pixel, $p \in \R^{\#class}, p \in [0, 1]$
- $l$ is the index of the bunch of the feature maps. (e.g. l == 1 corresponds to P3, ...)
- $x, y$ is coordinates of any point in the feature map.

### 3.3.2 CIoU  
CIoU Loss는 3개의 항의 합으로 구성되어 있는데, 중첩 영역($1 - IoU$), 중심점 거리 $\rho$와 종횡비 $\alpha v$의 합으로 구성되어 있다.
$$
L_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$  
Where:
- $\rho^2(b, b^{gt})$ is the euclide distance between gt and pred.  
- c: 두 박스를 모두 포함하는 최소 폐쇄 박스의 대각선 길이  
- $v=\frac{4}{\pi^2}(\arctan{\frac{w^{gt}}{h^{gt}}}-\arctan{\frac{w^{gt}}{h^{gt}}})$  
- $\alpha = \frac{v}{(1-IoU)+v}$

### 3.3.3 Offset L1  
중심점 거리에 대한 오차로 CIoU와 달리 히트맵의 해상도 한계로 발생하는 양자화 오류(Quantization error)를 줄이기 위한 미시적인 보정을 담당한다.
$$
L_{offset} = \frac{1}{N}\sum_{i=1}^N|o_i-\hat o_i|
$$

# 4 &ensp; Experiments Setting
2D box annotation을 이용하기 위함과 연구자의 데이터 저장 용량의 한계로 BDD100K 데이터셋을 사용한다. BDD100K의 annotation에서 class와 2D box $(x_1, y_1, x_2, y_2)$ 를 제공하기 때문에 아래와 같이 가공하여 정답 레이블을 만든다.

- (x1, y1, x2, y2) $\rightarrow$ (w, h, ctx, cty) $\rightarrow$ (f_w, f_h, ox, oy) $\rightarrow$ regs
- (x1, y1, x2, y2) $\rightarrow$ (w, h, ctx, cty) $\rightarrow$ (f_w, f_h, ox, oy) $\rightarrow$ gaussian distribution $\rightarrow$ hm, mask

(w, h, ctx, cty) $\rightarrow$ (f_w, f_h, ox, oy)에서 객체의 bbox 넓이 $\sqrt{wh}$ 에 따라 객체가 탐지되는 feature map을 결정한다(scale assignment).


# 5 &ensp; Results
(사진 추가)

# 6 &ensp; Conclusion
흠 근데 난 깊이 정보 그거 내가 임의로 거리를 정했는데 그게 아니라 P3, P4, P5 중 가장 활성화된 것을 택하는 것이 더 나을 것 같은데

보완 사항: heatmap을 원형이 아닌 타원형, 로테이션을 넣어 물체의 방향성을 분포에 넣는 것이 좋을 것 같다. 그래서 nuScenes의 3D box GT를 통해 모델을 고도화 할 수 있을 것 같다. (아 이 방법은 객체가 겹쳐있을 때 문제가 발생할 것 같다. 이게 아니라 별도로 물리 상태 임베딩을 하는 것으로 하자..)

# References
