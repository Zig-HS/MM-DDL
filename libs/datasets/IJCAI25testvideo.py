import os
import cv2
import json
import numpy as np
import librosa
import albumentations
from tqdm import tqdm

import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel

from .datasets import register_dataset
from .data_utils import truncate_feats

import json

# 保存 dict_db 到 JSON 文件
def save_dict_db_to_json(dict_db, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_db, f, indent=4)

# 从 JSON 文件读取 dict_db
def load_dict_db_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


Transformers = [
    'CLIP-16',
    'XCLIP-16',
]

def choose_CLIP_model(clip_model):
    match clip_model:
        case 'CLIP-16':
            return CLIPVisionModel.from_pretrained(".weights/clip-vit-base-patch16")
        case 'XCLIP-16':
            return XCLIPVisionModel.from_pretrained(".weights/xclip-base-patch16")

def pad_tensor_to_multiple_of_8(x):
    T = x.size(0)
    remainder = T % 8
    if remainder == 0:
        return x
    pad_size = 8 - remainder
    pad_tensor = x[-1].unsqueeze(0).repeat(pad_size, 1, 1, 1)
    x_padded = torch.cat([x, pad_tensor], dim=0)
    return x_padded


class WaveformDataset(Dataset):
    def __init__(self, data_list, dataset_root, sample_rate):
        self.data_list = data_list
        self.dataset_root = dataset_root
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        audio_path = os.path.join(self.dataset_root, video_item['id'])

        # 加载音频
        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        waveform = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform, sample_rate, video_item['id']


class VideoFrameDataset(Dataset):
    def __init__(self, data_list, dataset_root, sample_rate):
        self.data_list = data_list
        self.dataset_root = dataset_root
        self.sample_rate = sample_rate
        self.trans = self._default_transform()

    def __len__(self):
        return len(self.data_list)

    def _default_transform(self):
        return albumentations.Compose([
            albumentations.Resize(224, 224),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        video_path = os.path.join(self.dataset_root, video_item['id'])
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                augmented = self.trans(image=frame)
                image = augmented["image"]
                image = image.transpose(2, 0, 1)[np.newaxis, :]
                frames.append(image)
            cap.release()
            frames = np.concatenate(frames, axis=0)
            frames = torch.tensor(frames).unsqueeze(0).squeeze(0)
            return frames, video_item['id']
        except:
            with open('videos.txt','a') as f:
                f.write(f"{video_path}\n")

@register_dataset("ijcai25testvideo")
class IJCAI25testvideo(Dataset):
    def __init__(
        self,
        is_training,        
        devices,
        num_workers,
        ssl_model,
        clip_model,
        dataset_root,
        feat_stride,        # temporal stride of the feats
        num_frames,         # number of frames for each feat
        downsample_rate,    # downsample rate for feats
        max_seq_len,        # maximum sequence length during training
        trunc_thresh,       # threshold for truncate an action segment
        crop_ratio,         # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,          # input feat dim
        num_classes,        # number of action categories
        featureMapIndex,
    ):
        assert crop_ratio == None or len(crop_ratio) == 2
        self.is_training = is_training
        self.ssl_model = ssl_model
        self.clip_model = clip_model
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.featureMapIndex = featureMapIndex
        self.label_dict = None

        # 读取数据集信息，并存储为list
        self.dataset_root = dataset_root
        dict_db = self._load_label_db()
        self.data_list = dict_db
        self.device = torch.device(devices[0])
        self.num_workers = num_workers

        # 设置数据集属性
        self.db_attributes = {
            'dataset_name': 'ijcai25',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [],
        }

        # 设置音频预训练模型model
        self.bundle = getattr(torchaudio.pipelines, f"{self.ssl_model}")
        self.bundle_sample_rate = self.bundle.sample_rate
        self.transform = None

        # 新增缓存相关配置
        self.video_cache_dir = os.path.join('./.cached_data/test/', f'ijcai25_{clip_model}_feats')
        self.make_cache_dir(self.video_cache_dir)
        self.audio_cache_dir = os.path.join('./.cached_data/test/', f'ijcai25_{ssl_model}_feat_{self.featureMapIndex}')
        self.make_cache_dir(self.audio_cache_dir)

        # 检查是否需要预计算特征
        self.video_need_precompute = any(
            not os.path.exists(self._get_feature_path(video_item['id'],modal='video'))
            for video_item in self.data_list
        )
        self.audio_need_precompute = any(
            not os.path.exists(self._get_feature_path(video_item['id'],modal='audio'))
            for video_item in self.data_list
        )

        # 预计算特征（仅在需要时执行）
        if self.video_need_precompute and clip_model is not None:
            self._precompute_features(modal='video')
        if self.audio_need_precompute and ssl_model is not None:
            self._precompute_features(modal='audio')

    def make_cache_dir(self,cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    def _load_label_db(self):
        file_paths = []
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                if file.endswith(".mp4"):
                    file_paths.append(os.path.join(root, file))

        db_path = '.cached_data/test/dict_db.json'
        if os.path.exists(db_path):
            dict_db = load_dict_db_from_json(db_path)
        else:
            dict_db = []
            for video_path in tqdm(file_paths, desc='Loading Dataset'):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Unable to open video file: {video_path}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()

                # 确保帧率和帧数有效
                if fps <= 0 or frame_count <= 0:
                    duration = 0.0
                else:
                    duration = frame_count / fps

                # 构造 video_id：可以是文件名，也可以是相对路径
                video_id = os.path.basename(video_path)

                # 构建样本字典，仅保留基本字段
                dict_db.append({
                    'id': video_id,
                    'fps': fps,
                    'duration': duration
                })
            save_dict_db_to_json(dict_db,db_path)

        return dict_db


    def delta(self,mat):
        '''
        有限差分滤波器，计算它相对于列的一阶导数。
        win: 滤波器组的系数
        '''
        assert mat.ndim == 2
        win = np.array([-1.0, 0.0, 1.0]).reshape(3, 1)
        mat = np.concatenate((mat[:, :1], mat, mat[:, -1:]), axis=-1)
        mat = np.expand_dims(mat, 2)
        mat = np.concatenate((mat[:, :-2], mat[:, 1:-1], mat[:, 2:]), axis=2)
        t, v = mat.shape[:2]
        mat = np.dot(mat.reshape(-1, 3), win).reshape(t, v)
        return mat


    def _get_feature_path(self, video_id, modal):
        """生成特征文件的路径"""
        basename = os.path.basename(video_id)
        root, _ = os.path.splitext(basename)
        if modal == 'video':
            return os.path.join(self.video_cache_dir, f"{root}.npy")
        elif modal == 'audio':
            return os.path.join(self.audio_cache_dir, f"{root}.npy")


    def get_attributes(self):
        return self.db_attributes


    def _precompute_features(self, modal):
        """无填充的特征预计算"""
        if modal == 'video':
            self.model = choose_CLIP_model(self.clip_model)
            self.model.eval().to(self.device)
            filtered_data = [
                item for item in self.data_list
                if not os.path.exists(self._get_feature_path(item['id'],modal=modal))
            ]
            dataset = VideoFrameDataset(filtered_data, dataset_root = self.dataset_root, sample_rate = self.bundle.sample_rate)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True,)
            with torch.no_grad():
                for batch in tqdm(loader, desc='Precomputing Features'):
                    frames, video_id = batch
                    feature_path = self._get_feature_path(video_id[0],modal=modal)
                    try:
                        frames = frames.to(self.device)
                        b, T, _, h, w = frames.shape
                        frames = frames.reshape(-1, 3, h, w)
                        if self.clip_model == 'XCLIP-16' or self.clip_model == 'XCLIP-32':
                            frames = pad_tensor_to_multiple_of_8(frames) # 填充至 8n 个片段
                        if self.clip_model in Transformers:
                            feats = self.model(frames, output_hidden_states=True)
                            feats = feats.pooler_output
                        else:
                            feats = self.model(frames)
                        feats = feats[:T]
                        np.save(feature_path, feats.cpu().numpy().astype(np.float32))
                    except: # 视频过长，显存占用过大
                        del frames
                        with open('videos.txt','a') as f:
                            f.write(f"GPU memory is insufficient: {video_id[0]}\n")
                        print(f"WARNING!!!!!!! ")
                        print(f"GPU memory is insufficient for {video_id[0]}\n")
            self.model.cpu()
            del self.model

        elif modal == 'audio':
            self.model = self.bundle.get_model()
            self.model.eval().to(self.device)
            # 先过滤掉已经生成特征的视频项
            filtered_data = [
                item for item in self.data_list
                if not os.path.exists(self._get_feature_path(item['id'],modal=modal))
            ]
            dataset = WaveformDataset(filtered_data, dataset_root = self.dataset_root, sample_rate = self.bundle.sample_rate)
            loader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)
            with torch.inference_mode():
                for batch in tqdm(loader, desc='Precomputing Features'):
                    waveform, sample_rate, audio_id = batch
                    feature_path = self._get_feature_path(audio_id[0],modal=modal)
                    waveform = waveform.to(self.device)
                    feats, _ = self.model.extract_features(waveform)
                    feats = torch.stack(feats, dim=0).squeeze()
                    feats = feats[self.featureMapIndex].transpose(1, 0)
                    np.save(feature_path, feats.cpu().numpy().astype(np.float32))
            self.model.cpu()
            del self.model


    def __len__(self):
        return len(self.data_list)


    def extract_feature(self, idx, feat_stride, feat_offset, modal=None):
        video_item = self.data_list[idx]
        feature_path = self._get_feature_path(video_item['id'], modal=modal)
        feats = torch.from_numpy(np.ascontiguousarray(np.load(feature_path, mmap_mode='r')))

        if modal == 'video':
            feats = feats.transpose(1, 0)
        data_dict = {
            'feats': feats,  # C x T
        }

        return data_dict['feats']

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        video_feats = self.extract_feature(idx, feat_stride, feat_offset, modal='video')
        
        video_dict = {
            'video_id': video_item['id'],
            'feats': video_feats,  # C x T
            'fps': video_item['fps'],
            'duration': video_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames
        }

        return video_dict
