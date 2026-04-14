import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class NuScenesSequenceDataset(Dataset):
    def __init__(self, n_prev=3, dataroot='/data/sets/nuscenes', version='v1.0-trainval', split='train'):
        self.nsc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.n_prev = n_prev # n개의 이전 프레임
        self.samples = [s for s in self.nsc.sample if self.nsc.get('scene', s['scene_token'])['name'] in self._get_split_scenes(split)]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_split_scenes(self, split):
        # 실제 split에 따른 scene 리스트 필터링 (간소화됨)
        from nuscenes.utils.splits import create_splits_scenes
        return create_splits_scenes()[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        curr_sample = self.samples[idx]
        sequence_tokens = []
        
        # 현재 프레임 포함하여 과거로 추적
        temp_sample = curr_sample
        for _ in range(self.n_prev + 1):
            sequence_tokens.insert(0, temp_sample['data']['CAM_FRONT'])
            if temp_sample['prev'] != "":
                temp_sample = self.nsc.get('sample', temp_sample['prev'])
            else:
                # 과거 데이터가 부족하면 현재 데이터를 복제 (Padding 효과)
                pass

        # 이미지 로드
        images = []
        for token in sequence_tokens:
            sd_record = self.nsc.get('sample_data', token)
            img = Image.open(f"{self.nsc.dataroot}/{sd_record['filename']}").convert('RGB')
            images.append(self.transform(img))
        
        img_tensor = torch.stack(images) # (n+1, 3, 224, 224)

        # 차량 상태 정보 (속도 등) - ego_pose에서 추출
        ego_pose = self.nsc.get('ego_pose', self.nsc.get('sample_data', curr_sample['data']['CAM_FRONT'])['ego_pose_token'])
        # 단순화를 위해 속도(velocity)를 state로 사용. nuScenes는 직접적인 페달값을 제공하지 않으므로 
        # 학습 시에는 가속도(acceleration) 또는 미래 위치를 Label로 사용해야 함.
        velocity = np.linalg.norm(ego_pose['rotation']) # 예시값
        state = torch.tensor([velocity], dtype=torch.float32) 
        
        # 정답 레이블 (현재 시점의 타겟 페달값/가속도)
        # ※ 실제 데이터셋 구성 시 별도의 Label 가공이 필요합니다.
        label = torch.tensor([0.5], dtype=torch.float32) 

        return img_tensor, state, label