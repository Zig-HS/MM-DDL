# Multi-modal Deepfake Detection and Localization with FPN-Transformer

This repository contains official code for *Multi-modal Deepfake Detection and Localization with FPN-Transformer* for [IJCAI2025 Workshop on Deepfake Detection, Localization, and Interpretability](https://deepfake-workshop-ijcai2025.github.io/main/accepted_paper.html).

## Environment Setup

Our codebase requires the following Python:

+   Python == 3.11.11

You can set up the environment by following these steps:

1. Install the necessary libraries:

   ```shell
   pip install -r requirements.txt
   ```

2. Part of NMS is implemented in C++. The code can be compiled by:

   ```shell
   cd ./libs/utils
   python setup.py install --user
   cd ../..
   ```

   If you encounter an error: `libstdc++.so.6: version GLIBCXX_3.4.32 not found`, you can try to fix it by running (for ubuntu):

   ```shell
   rm $CONDA_PREFIX/lib/libstdc++.so.6
   ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/
   ```

3. Download the pre-trained weights of CLIP and XCLIP by:

   ```shell
   cd .weights
   bash hfd.sh openai/clip-vit-base-patch16 --tool wget
   bash hfd.sh microsoft/xclip-base-patch16 --tool wget
   cd ..
   ```
   
   If you failed to connect, you can modify the `export HF_ENDPOINT="https://hf-mirror.com"` in `hfd.sh`.

## Dataset Preparation

+   Download DDL-AV **training set**. After extraction, rename the folder to `.dataset`. You can refer to the following structure:

    ```
    .dataset
    ├── train_data					
    ├── train_metadata_workshop
    ├── val_data
    └── val_metadata_workshop
    ```
    
+ Download DDL-AV **test Set**. After extraction, rename the folder to `.dataset_test`. You can refer to the following structure:

  ```
  .dataset_test
  ├──	xxxx.mp4					
  │	..
  │	..
  ```

## Model Weights

The pre-trained model weights of our method are provided as `./ckpt/ijcai25audio-wavLM/epoch_003.pth.tar` and `./ckpt/ijcai25video-CLIP16/epoch_003.pth.tar`.

## Training

+   After preparing the dataset, you can train the model with the following command.

    +   For audio model (default configuration):

        ```shell
        python train-audio.py
        ```

    +   For video model (default configuration):

        ```
        python train-video.py
        ```

+   You can get and modify the detailed training parameters through `./configs_train/ijcai25audio-wavLM.yaml` and `./configs_train/ijcai25video-CLIP16.yaml`.

## Evaluation

+   First, evaluate the audio and video model by running the following command.

    + For audio model (default configuration and model weights):

      ```shell
      python test-audio.py
      ```

    + For video model (default configuration and model weights):

      ```shell
      python test-video.py
      ```

    After these, the unimodal results will be saved in `./results/`.

+   Then, get the final results by running:

    ```shell
    python combine_results.py
    ```

    After this, the final prediction results will be saved in `./prediction/`.

+   You can get and modify the detailed test parameters through `./configs_test/ijcai25audio-wavLM.yaml` and `./configs_test/ijcai25video-CLIP16.yaml`.

## Technical Documentation


Please refer to `Technical-Documentation.PDF` in our submitted materials, which includes detailed information about *model architecture* and *implement details*.
