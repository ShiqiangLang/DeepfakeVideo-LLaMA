import os
import gradio as gr
import subprocess
import shutil  

def process_video(input_video):
    # global tmpdir
    FilePath = input_video.name

    # 检查文件扩展名
    if not input_video.name.lower().endswith('.mp4'):
        return "格式错误，请上传.mp4格式文件"
    
    # 将文件复制到临时目录中
    shutil.copy(input_video.name, 'tmpdir')

    # 获取上传Gradio的文件名称
    FileName = os.path.basename(input_video.name)

    # 打开复制到新路径后的文件
    with open(FilePath, 'rb') as input_video:
 
        # 在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
        output_video_path = os.path.join('/home/lsq/video', FileName)
        with open(output_video_path, 'wb') as w:
            w.write(input_video.read())

    # 运行模型预测
    command = [
        'python', 'demo_audiovideo.py', 
        '--cfg-path', '/home/lsq/Video-LLaMA/eval_configs/video_llama_eval_withaudio.yaml', 
        '--model_type', 'llama_v2', '--gpu-id', '1'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, _ = process.communicate()

    # 获取倒数第二行输出
    # lines = output.decode('utf-8').strip().split('\n')[-2]

    # 返回结果
    return output

# 接口设置
inputs = gr.inputs.File(label="请上传视频", type="file")
output = gr.outputs.Textbox(label="检测结果")

# 创建 Gradio 接口
gr.Interface(
    fn=process_video, inputs=inputs, outputs=output, 
    title="AIGC反诈视频检测", description="请上传一个.mp4格式的视频进行检测"
).launch(share=True)