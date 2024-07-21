import argparse
from cgitb import text
from multiprocessing import process
import os
import random
from turtle import title

from grpc import server
import numpy as np
from pyparsing import White
from pytz import utc
import torch
import torch.backends.cudnn as cudnn
import gradio as gr 
import os
import gradio as gr
import subprocess
import shutil 

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *
import os
os.environ['GRADIO_TEMP_DIR'] = '/home/lsq/Video-LLaMA/tmp'

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='/home/lsq/Video-LLaMA/eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value='Determine the video and tell me the video is synthetic or authentic.', placeholder='tell me the video is synthetic or authentic.', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_imgorvideo(gr_video, gr_img, text_input, chat_state,chatbot,audio_flag):
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    if gr_img is None and gr_video is None:
        return None, None, None, gr.update(interactive=True), chat_state, None
    elif gr_img is not None and gr_video is None:
        print(gr_img)
        chatbot = chatbot + [((gr_img,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    elif gr_video is not None and gr_img is None:
        print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state.system =  ""
        img_list = []
        if audio_flag:
            llm_message = chat.upload_video(gr_video, chat_state, img_list)
        else:
            llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list,chatbot
    else:
        # img_list = []
        return gr.update(interactive=False), gr.update(interactive=False, placeholder='Currently, only one input is supported'), gr.update(value="Currently, only one input is supported", interactive=False), chat_state, None,chatbot
def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    print(chat_state.get_prompt())
    print(chat_state)
    return chatbot, chat_state, img_list

title = '''<h1 align="center">
<a><div style="display: flex; justify-content: center;">
      <img src="https://www.bupt.edu.cn/__local/C/8E/F7/EE902059AE32E0E6325EFEE8F46_B2D41D06_CD58.png" alt="Video-LLaMA" border="0" style="height: 200px; margin-right: 10px;" />
      <img src="https://s21.ax1x.com/2024/05/23/pkQmpLt.jpg" alt="Video-LLaMA" border="0" style="height: 200px; margin-left: 10px;" /></div></a>
</h1>
<h1 align="center" style="font-size: 50px;">DeepFake Detection By Video-LLaMA</h1>
<p align="center" style="font-size: 12px;">Thank you for using the DeepFake Detection Page!</p>
'''

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.themes.Soft()
    with gr.Row():
        with gr.Column(scale = 2): # type: ignore
            video = gr.Video()
            image = gr.Image(type="filepath")
            upload_button = gr.Button(value="Upload & Start", interactive=True, variant="primary")
            clear = gr.Button("Restart")
        
        with gr.Column(scale = 1.5): # type: ignore
            chat_state = gr.State() # type: ignore
            img_list = gr.State() # type: ignore
            chatbot = gr.Chatbot(label='Video-LLaMA')
            fixed_text = "Determine the video and tell me the video is synthetic or authentic.If the video is synthetic and tell me yes or no  only."
            text_input = gr.Textbox(value= fixed_text, scale=3, label='User', placeholder='Upload your video first, or directly click the examples at the bottom of the page.', interactive=False)
            
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            audio = gr.Checkbox(interactive=True, value=False, label="Audio")

    # with gr.Column():
    #     gr.Examples(examples=[
    #         [f"examples/dog.jpg", "Which breed is this dog? "],
    #         [f"examples/JonSnow.jpg", "Who's the man on the right? "],
    #         [f"examples/Statue_of_Liberty.jpg", "Can you tell me about this building? "],
    #     ], inputs=[image, text_input])

    #     gr.Examples(examples=[
    #         [f"examples/skateboarding_dog.mp4", "What is the dog doing? "],
    #         [f"examples/birthday.mp4", "What is the boy doing? "],
    #         [f"examples/IronMan.mp4", "Is the guy in the video Iron Man? "],
    #     ], inputs=[video, text_input])

    upload_button.click(upload_imgorvideo, [video, image, text_input, chat_state,chatbot,audio], [video, image, text_input, upload_button, chat_state, img_list,chatbot]).then(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list])
    # text_input = f'only need to tell me the video is synthetic or authentic.'
    # text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
    #     gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    # )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, image, text_input, upload_button, chat_state, img_list], queue=True)

demo.launch(enable_queue=True)


# %%
