import os
import json
import numpy as np
import glob
import random
import librosa
from pprint import pprint
from tqdm import tqdm
# from moviepy.editor import VideoFileClip
import subprocess

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial

from .datasets import register_dataset
from .data_utils import truncate_feats
import torchaudio.transforms as T


@register_dataset("ijcai25audio")
class IJCAI25(Dataset):
    def __init__(
        self,
        is_training,        # if in training mode
        devices,
        num_workers,
        ssl_model,
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
        self.cache_dir = os.path.join('./.cached_data/', f'ijcai25_{ssl_model}_feat_{self.featureMapIndex}')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/fake_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/real_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/fake_audio_fake_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/train_data/real_audio_fake_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/fake_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/real_audio_real_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/fake_audio_fake_visual/', exist_ok=True)
        os.makedirs(f'{self.cache_dir}/val_data/real_audio_fake_visual/', exist_ok=True)

        # 检查是否需要预计算特征
        self.need_precompute = any(
            not os.path.exists(self._get_feature_path(video_item['id']))
            for video_item in self.data_list
        )

        # 预计算特征（仅在需要时执行）
        if self.need_precompute:
            self._precompute_features()

    def delta(sef,mat):
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

    def get_attributes(self):
        return self.db_attributes

    def _load_label_db(self):
        file_paths = []
        for root, dirs, files in os.walk(self.dataset_root):
            for file in files:
                if file.endswith(".json"):
                    file_paths.append(os.path.join(root, file))

        dict_db = []  # 改用列表存储
        split = 'val' if self.is_training == False else 'train'
        for json_path in tqdm(file_paths, desc='Loading Dataset'):
            with open(json_path, 'r') as f:
                data = json.load(f)
                if data["split"] != split:
                    continue  # 根据split过滤
                if 'fake_audio' in data["modify_type"]:
                    segments = data.get("audio_fake_segments", []), # 音频伪造片段
                    segments = np.array(segments, dtype=np.float32).reshape(-1, 2)
                    labels = np.zeros(len(segments), dtype=np.int64)
                else:
                    segments = np.empty((0, 2))
                    labels = np.empty((0, 1))

                # 构建样本信息
                video_id = 'train_data/' + data["file"] if split == 'train' else 'val_data/' + data["file"]
                dict_db.append({
                    'id': video_id,
                    'fps': 50, # 设置fps，音频帧率为50
                    'duration': data["video_frames"] / 25.0, # 25为视频帧率,用视频帧数计算时长
                    'segments': segments,
                    'labels': labels
                })

        return dict_db

    def _get_feature_path(self, audio_id):
        """生成特征文件的路径"""
        dirname = os.path.dirname(audio_id)
        basename = os.path.basename(audio_id)
        feature_dir = os.path.join(self.cache_dir, dirname)
        root, _ = os.path.splitext(basename)
        return os.path.join(feature_dir, f"{root}.npy")

    def _precompute_features(self):
        """无填充的特征预计算"""

        class PrecomputeDataset(Dataset):
            def __init__(self, data_list, dataset_root, sample_rate):
                self.data_list = data_list
                self.dataset_root = dataset_root
                self.sample_rate = sample_rate

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, idx):
                video_item = self.data_list[idx]
                audio_path = os.path.join(self.dataset_root, video_item['id'])
                try:
                    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                    waveform = torch.from_numpy(waveform)
                    waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
                except Exception as e:
                    with open('audios.txt','a') as f:
                        f.write(f"{audio_path}\n")
                return waveform, sample_rate, video_item['id']

        self.model = self.bundle.get_model()
        self.model.eval().to(self.device)

        # 先过滤掉已经生成特征的视频项
        filtered_data = [
            item for item in self.data_list
            if not os.path.exists(self._get_feature_path(item['id']))
        ]

        dataset = PrecomputeDataset(filtered_data, dataset_root = self.dataset_root, sample_rate = self.bundle.sample_rate)
        loader = DataLoader(dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)
        with torch.inference_mode():
            for batch in tqdm(loader, desc='Precomputing Features'):
                waveform, sample_rate, audio_id = batch
                feature_path = self._get_feature_path(audio_id[0])
                waveform = waveform.to(self.device)
                feats, _ = self.model.extract_features(waveform)
                feats = torch.stack(feats, dim=0).squeeze()
                feats = feats[self.featureMapIndex].transpose(1, 0)
                np.save(feature_path, feats.cpu().numpy().astype(np.float32))
        self.model.cpu()
        del self.model

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        audio_id = video_item['id']

        feature_path = self._get_feature_path(video_item['id'])
        feats = torch.from_numpy(np.ascontiguousarray(np.load(feature_path, mmap_mode='r')))
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride

        segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset)
        labels = torch.from_numpy(video_item['labels'])

        data_dict = {
            'video_id': audio_id,
            'feats': feats,  # C x T
            'segments': segments,  # N x 2
            'labels': labels,  # N
            'fps': video_item['fps'],
            'duration': video_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames
        }

        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh,
                feat_offset, self.crop_ratio
            )

        return data_dict
