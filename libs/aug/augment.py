import random
import numpy as np
import torch
import time
import torchaudio

from audiomentations import Compose,AddGaussianNoise, TimeStretch, PitchShift, AddGaussianSNR,AirAbsorption,BandPassFilter,Gain,GainTransition,Limiter,Normalize,PeakingFilter,PolarityInversion,RoomSimulator,SevenBandParametricEQ,TanhDistortion,TimeMask,Trim,AddBackgroundNoise, PolarityInversion,ApplyImpulseResponse

noise_path = [
                '/data/jzl/musan/noise/free-sound/',
                '/data/jzl/musan/noise/sound-bible/',
                '/data/jzl/RIRS_NOISES/pointsource_noises/']

rir_path =  [
                '/data/jzl/RIRS_NOISES/real_rirs_isotropic_noises/',
                '/data/jzl/RIRS_NOISES/simulated_rirs/mediumroom/Room',
                '/data/jzl/RIRS_NOISES/simulated_rirs/smallroom/Room',
                '/data/jzl/RIRS_NOISES/simulated_rirs/largeroom/Room',]

def initAugment(noise_path, rir_path):

    assert noise_path and rir_path, "folder_list and folder_list2 should not be empty or None"

    noise = random.choice(noise_path)
    rir_base   = random.choice(rir_path)
    if rir_base.rstrip('/').split('/')[-1] == 'Room':
        random_room_number = random.randint(1, 200)
        rir = f"{rir_base}{random_room_number:03}/"
    else:
        rir = rir_base

    augment = Compose([
    AddBackgroundNoise(sounds_path= noise ,min_snr_in_db=3.0,max_snr_in_db=30.0,
                       noise_transform=PolarityInversion(),p=0.8),
    ApplyImpulseResponse(ir_path= rir , p=0.8),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.01),
    AddGaussianSNR(min_snr_db=5.0,max_snr_db=40.0,p=0.1),
    AirAbsorption(min_distance=10.0,max_distance=50.0,p=0.1),
    BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=0.05),
    Gain(p=0.2),
    GainTransition(p=0.05),
    Limiter(min_threshold_db=-16.0,max_threshold_db=-6.0,threshold_mode="relative_to_signal_peak",p=0.05),
    Normalize(p=0.2),
    PeakingFilter(p=0.05),
    PitchShift(min_semitones=-5.0,max_semitones=5.0,p=0.05),
    PolarityInversion(p=0.05),
    # RoomSimulator(p=0.2),
    SevenBandParametricEQ(p=0.05),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.05),
    TanhDistortion(min_distortion=0.01,max_distortion=0.7,p=0.01),
    TimeMask(min_band_part=0.1,max_band_part=0.15,fade=True,p=0.05),
    Trim(top_db=30.0,p=0.05),
])  
    return augment




def call_MixAugment(waveform,sample_rate, prob):
    augment = initAugment(noise_path, rir_path)
    if np.random.rand() < prob:
        augmented_samples = augment(samples=waveform.numpy().squeeze(), sample_rate=sample_rate)
        waveform = torch.from_numpy(augmented_samples).squeeze().unsqueeze(0)
        return waveform
    return waveform




def test_call_MixAugment():
    # 创建一个简单的波形数据和样本率
    waveform = torch.randn(1, 16000)  # 假设的 1 秒音频数据，16000 是样本率
    sample_rate = 16000  # 样本率

    # 设置概率值
    prob =0.5  # 有 50% 的概率应用增强

    # 调用函数
    filename = '/data/jzl/ADD2023/ADD2023_train/wav/ADD2023_T2_T_00000000.wav'
    waveform, sample_rate = torchaudio.load(filename)
    augmented_waveform = call_MixAugment(waveform, sample_rate, prob)
    output_filename = 'test.wav'
    torchaudio.save(output_filename, augmented_waveform, sample_rate)

    # 检查返回的数据类型和形状
    assert isinstance(augmented_waveform, torch.Tensor), "返回值应该是一个 Torch 张量"
    if torch.equal(augmented_waveform, waveform):
        print ("没增强")
    # "增强后的波形应该与原始波形有相同的形状"

    # 可以添加更多的断言来检查特定的增强是否被应用

# # 运行测试
# test_call_MixAugment()






