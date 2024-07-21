import os
from video_llama.models.video_llama import VideoLLAMA


video_llama = VideoLLAMA()

def find_mp4_files(root_dir='/path/to/your/root/directory'):
    mp4_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mp4'):
                mp4_files.append(os.path.join(dirpath, filename))
        for dirname in dirnames:
            mp4_files.extend(find_mp4_files(os.path.join(dirpath, dirname)))
    return mp4_files

all_mp4_files = find_mp4_files()

def process():
    