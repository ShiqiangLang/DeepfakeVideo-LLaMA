import argparse
from hmac import new
from logging import config
from tty import CFLAG
from types import new_class
import os
import einops
import torch 
import torch.nn as nn
from torch.cuda.amp import autocast # type: ignore
from wandb import Video  # type: ignore
from video_llama.common.registry import registry
from video_llama.models.video_llama import VideoLLAMA
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.blip2 import Blip2Base
from transformers import LlamaTokenizer,BertConfig
from transformers import LlamaTokenizer
from video_llama.models.modeling_llama import *
from video_llama.processors.video_processor import AlproVideoTrainProcessor


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_path = '/home/lsq/FF++/manipulated_sequences/DeepFakeDetection/c23/videos/01_02__exit_phone_room__YVGY8LOK.mp4'
vis_processor = AlproVideoTrainProcessor()
video = vis_processor(video_path).unsqueeze(0)
print(video.shape)    #shape：(1, 3, 100, 244, 244)  type：torch.float32    

# input shape b,c,t,h,w
batch_size,_,time_length,_,_ = video.size()
image = einops.rearrange(video, 'b c t h w -> (b t) c h w').to(device)
print(image.shape)    # shape：(100, 3, 244, 244)  type：torch.float16

def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("/home/lsq/premodel/bert-base-uncased/")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block 
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

with autocast():
    blip2 = Blip2Base().to(device)
#     videollama = VideoLLAMA()
    visual_encoder, ln_vision = blip2.init_vision_encoder(model_name="eva_clip_g", img_size=224, drop_path_rate=0.4, use_grad_checkpoint=False, precision="fp16")
    video_Qformer,video_query_tokens = init_video_Qformer(cls = None, num_query_token = 32,vision_width=768, num_hidden_layers =2)
    # embed image features with blip2, out: (b t) q h
    with torch.no_grad():
        image_embeds = ln_vision(visual_encoder(image)).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        print(image_embeds.shape)
        print(image_atts.shape)

        Qformer, query_tokens = blip2.init_Qformer(num_query_token=32, vision_width=768, cross_attention_freq=2)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
        query_output = Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                )
        print(query_output.shape)

        # add frame_pos embedding
        position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device).to(device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1).to(device)
        video_frame_position_embedding = nn.Embedding(32, 768).to(device)
        frame_position_embeddings = video_frame_position_embedding(position_ids).to(device)
        q_hidden_state = query_output.last_hidden_state.to(device)
        print(q_hidden_state.shape)

        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2).to(device)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length).to(device)
        frame_hidden_state = frame_position_embeddings.to(device) + frame_hidden_state.to(device)
        print(frame_hidden_state.shape)

        # frame attention
        frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length).to(device)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
        video_query_tokens = video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1).to(device)
        print(video_query_tokens.shape)

        video_query_output = video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
        video_hidden = video_query_output.last_hidden_state
        print(video_hidden.shape)

        llama_proj = nn.Linear(768, 4096) # type: ignore
        inputs_llama = llama_proj(video_hidden)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)


# llama = LlamaForCausalLM.from_pretrained(
#                 "/home/lsq/Video-LLaMA/ckpt-7b/llama-2-7b-chat-hf/",
#                 torch_dtype=torch.bfloat16,
#                 load_in_8bit=True,
#                 device_map={'': 0}
#             )
# llama_tokenizer = LlamaTokenizer.from_pretrained("/home/lsq/Video-LLaMA/ckpt-7b/llama-2-7b-chat-hf/", use_fast=False)
# llama_tokenizer.padding_side = "right"

# video_llama = VideoLLAMA()