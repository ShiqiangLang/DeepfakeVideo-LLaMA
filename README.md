<p align="center" width="100%">
<a target="_blank"><img src="figs/video_llama_logo.jpg" alt="Video-LLaMA" style="width: 50%; min-width: 200px; display: block; margin: auto;"></a>
</p>



# Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding
<!-- **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding** -->

This is the repo for the Video-LLaMA project, which is working on empowering large language models with video and audio understanding capabilities. 

<p align="center" width="100%">
<a target="_blank"><img src="figs/architecture_v2.png" alt="Video-LLaMA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Introduction


- Video-LLaMA is built on top of [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). It is composed of two core components: (1) Vision-Language (VL) Branch and (2) Audio-Language (AL) Branch.
  - **VL Branch** (Visual encoder: ViT-G/14 + BLIP-2 Q-Former)
    - A two-layer video Q-Former and a frame embedding layer (applied to the embeddings of each frame) are introduced to compute video representations. 
    - We train VL Branch on the Webvid-2M video caption dataset with a video-to-text generation task. We also add image-text pairs (~595K image captions from [LLaVA](https://github.com/haotian-liu/LLaVA)) into the pre-training dataset to enhance the understanding of static visual concepts.
    - After pre-training, we further fine-tune our VL Branch using the instruction-tuning data from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA) and [VideoChat](https://github.com/OpenGVLab/Ask-Anything). 
  - **AL Branch** (Audio encoder: ImageBind-Huge) 
    - A two-layer audio Q-Former and an audio segment embedding layer (applied to the embedding of each audio segment) are introduced to compute audio representations.
    - As the used audio encoder (i.e., ImageBind) is already aligned across multiple modalities, we train AL Branch on video/image instruction data only, just to connect the output of ImageBind to the language decoder.    
- Only the Video/Audio Q-Former, positional embedding layers, and linear layers are trainable during cross-modal training.


## Pre-trained & Fine-tuned Checkpoints
The following checkpoints are the full weights (visual encoder + audio encoder + Q-Formers + language decoder) to launch Video-LLaMA:

| Checkpoint       | Link | Note |
|:------------------|-------------|-------------|
| Video-LLaMA-2-7B-Pretrained    | [link](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned/tree/main)       | Pre-trained on WebVid (2.5M video-caption pairs) and LLaVA-CC3M (595k image-caption pairs) |
| Video-LLaMA-2-7B-Finetuned | [link](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main) | Fine-tuned on the instruction-tuning data from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA) and [VideoChat](https://github.com/OpenGVLab/Ask-Anything)|
| Video-LLaMA-2-13B-Pretrained    | [link](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Pretrained/tree/main)       | Pre-trained on WebVid (2.5M video-caption pairs) and LLaVA-CC3M (595k image-caption pairs) |
| Video-LLaMA-2-13B-Finetuned | [link](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned/tree/main) | Fine-tuned on the instruction-tuning data from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA) and [VideoChat](https://github.com/OpenGVLab/Ask-Anything)|


## Training

The training of each cross-modal branch (i.e., VL branch or AL branch) in Video-LLaMA consists of two stages,

1. Pre-training on the [Webvid-2.5M](https://github.com/m-bain/webvid) video caption dataset and [LLaVA-CC3M]((https://github.com/haotian-liu/LLaVA)) image caption dataset.

2. Fine-tuning using the image-based instruction-tuning data from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)/[LLaVA](https://github.com/haotian-liu/LLaVA) and the video-based instruction-tuning data from [VideoChat](https://github.com/OpenGVLab/Ask-Anything).


## Acknowledgement
We are grateful for the following awesome projects our Video-LLaMA arising from:
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): Enhancing Vision-language Understanding with Advanced Large Language Models
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots
* [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2): Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models 
* [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP): Improved Training Techniques for CLIP at Scale
* [ImageBind](https://github.com/facebookresearch/ImageBind): One Embedding Space To Bind Them All
* [LLaMA](https://github.com/facebookresearch/llama): Open and Efficient Foundation Language Models
* [VideoChat](https://github.com/OpenGVLab/Ask-Anything): Chat-Centric Video Understanding
* [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
* [WebVid](https://github.com/m-bain/webvid): A Large-scale Video-Text dataset
* [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main): Modularization Empowers Large Language Models with Multimodality

The logo of Video-LLaMA is generated by [Midjourney](https://www.midjourney.com/).


