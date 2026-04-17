나는 현재 AI 모델을 만들고 있다. 모델의 입력은 영상 정보(bdd100k 기본 기준 1280 x 720)이고 output은 다중 오브젝트들의 class와 2d box이다.

모델은 크게 5가지의 부분으로 이루어지는데 구조와 목적은 아래와 같다.

1. Faster R-CNN (ResNet-50 + FPN)
    1. 목적: 이미지를 transformer encoder로 넣어서 이미지의 feature map을 추출하는 것은 학습 수렴이 CNN 대비 더디다를 논하기도 전에 transformer의 공간 및 시간 복잡도 O(n^2)에 의해 모델을 VRAM에 올리는것조차 불가능할 것이 거의 확실하다. 따라서 CNN을 통해 feature map을 추출해야 하고 pytorch에 pre-trained된 모델을 불러올 수 있기 때문에 freeze 해두면 모델 학습에 있어 확실한 이점을 가질 수 있다.
    2. 참고: P3, P4, P5를 계속 전달함.
2. Visual Resampler
    1. 목적: feature map은 P3, P4, P5마다 사이즈가 다르기 때문에 transformer encoder에 넣기 곤란하다. Visual Resampler를 통해 사이즈를 맞출뿐만 아니라 토큰 수를 줄여 측방 차량의 바퀴 각도 수준의 P3도 고려할 수 있게 되어 더 많은 의미를 계속 전달할 수 있게 되고 transformer encoder의 공간 및 시간 복잡도의 부담을 줄여준다. Visual Resampler에 입력된 Feature Representation은 Latent Representation으로 변하면서 더 이상 공간 상 픽셀값이 아닌 추상화된 의미를 표현하게 된다.
    2. 참고: 2d positional embedding, level embedding 필요, Q는 64 이상 256 이하가 적당할 것으로 보임
3. Transformer Encoder
    1. 목적: Transformer Encoder에 들어가서 Self-Attention 연산을 거치면서 각 토큰(Latent Representation)들은 서로의 존재나 관계를 전혀 모르는 고립된 상태를 벗어나 상호 간의 관계를 형성하게 된다. 이제 각 토큰 벡터들은 단순한 국소 특징이 아니라, 전체 맥락 속에서의 의미를 내포한 고차원적인 Representation이 된다.
4. Transformer Decoder
    1. 목적: 각 토큰 벡터들은 Transformer Decoder를 거치며 객체에 대한 정보를 모두 담은 객체 표현 벡터(압축된 정보 덩어리)로 표현됨
5. Head
    1. 목적: 객체 표현 벡터에서 유용한 정보를 뽑아내는 과정
    2. 뽑아낼 정보: class, 2d box

나는 차선 인식, 차로 인식, 도로 바운더리 인식, 시계열 추가, 차량 상태 변수 추가 등 다른 task로의 scalability를 위해 불필요해 보이는 코드 분리가 있을 수 있다. 이것은 불필요한 것이 아닌 의도된 동작이다.

데이터셋은 bdd100k를 사용할 것이다.

레포지토리 구조는 아래와 같다.

1. configs
    1. example.yaml
2. data
    1. bdd100k
        1. images
            1. 10K
                1. train
                    1. *.jpg
                2. val
                    1. *.jpg
                3. test
                    1. *.jpg
                    2. testA
                        1. *.jpg
                    3. testB
                        1. *.jpg
                    4. trainA
                        1. *.jpg
                    5. trainB
                        1. *.jpg
            2. 100K
                1. train
                    1. *.jpg
                2. val
                    1. *.jpg
                3. test
                    1. *.jpg
                    2. testA
                        1. *.jpg
                    3. testB
                        1. *.jpg
                    4. trainA
                        1. *.jpg
                    5. trainB
                        1. *.jpg
        2. labels
            1. bdd100k_labels_images_train.json
            2. bdd100k_labels_images_val.json
3. docker
    1. Dockerfile
4. projects
    1. <project_name>
        1. data
            1. bdd_loader.py
            2. augmentations.py
        2. models
            1. components
                1. resampler.py
                2. detection_decoder.py
                3. visual_encoder.py
                4. heads.py
            2. model.py
            3. matcher.py
            4. loss.py
        3. training
            1. optimizer.py
            2. train.py
            3. inference.py
        4. utils
            1. logger.py
            2. metrices.py

---

주의 사항이 있네? “Decoder 기반 모델(예: DETR 등)은 하나의 객체에 대해 여러 개의 쿼리가 반응할 위험이 있습니다. 이를 해결하기 위해 이분 매칭(Bipartite Matching) 같은 복잡한 손실 함수를 쓰지만, 여전히 중복 검출 문제는 완전히 자유롭지 못하며 이는 정확도 저하로 이어집니다.” 라고 함