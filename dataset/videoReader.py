from decord import VideoReader, cpu, gpu
import numpy as np
from torchvision import transforms
from dataset.transforms.video import Resize, CenterCrop, Compose
import math

class VideoLoader:
    def __init__(self, video_path, fps:int,use_gpu=False, new_width=340, new_height=256, keep_ori_ratio=True, transforms=None):
        self.fps=fps
        self.use_gpu = use_gpu
        self.width = new_width
        self.height = new_height
        self.keep_ori_ratio = keep_ori_ratio
        self.video_path = video_path
        self.vr=self._get_frame_stream(self.video_path)
        self.n_frame = len(self.vr) #debug
        self.orin_fps = self.vr.get_avg_fps()
        self.transforms =transforms
    
    def _get_frame_stream(self,path):
        if self.keep_ori_ratio:
            decoder = VideoReader(path,num_threads=1, ctx=cpu(0))
        else:
            decoder = VideoReader(path,width=self.width, height=self.height,
             num_threads=1, ctx=cpu(0))
        # decoder.seek(0)
        self.height, self.width, _ = decoder[0].shape
        return decoder
    
    def get_clip_indexes(self, len, stride):
        n_frame = math.floor(self.n_frame*(self.fps/self.orin_fps))
        indexes=np.linspace(0, self.n_frame-1, n_frame,endpoint=True,dtype=np.int32)
        start_indexes = indexes[::stride]
        batch_clip_index=[]
        for start in start_indexes:
            if (start+len)<=n_frame:
                frame_indexes=indexes[start:(start+len)]
                # patched_clip=vr.get_batch(frame_indexes)
                batch_clip_index.append(frame_indexes)
            # elif (n_frame-start)>=len/2:
            #     frame_indexes = indexes[start:]
            #     batch_clip_index.append(frame_indexes)
                # patched_clip=np.zeros(len, self.height, self.width, 3)
                # tmp_clip=vr.get_batch(frame_indexes)
                # patched_clip[:n_frame-start] = tmp_clip
        return batch_clip_index 
    
    def read_clip(self, clip_index):
        clip = self.vr.get_batch(clip_index).asnumpy()
        # clip = np.zeros((clip_len, self.height, self.width, 3))
        # if len(clip_index)>=clip_len:
        #     clip[:]=self.vr.get_batch(clip_index).asnumpy()
        # elif len(clip_index)<clip_len:
        #     orin_clip = self.vr.get_batch(clip_index).asnumpy()
        #     clip[:len(clip_index)]=orin_clip
        if transforms is not None:
            clip=self.transforms(clip)
            clip = clip.astype(np.float32)
            clip = (data * 2 / 255) - 1
        return clip
            




if __name__ == '__main__':
    reader=VideoLoader('/mnt/cephfs/dataset/activitynet_video/all_videos/v___c8enCfzqw.mp4',25)
    clip_indexes = reader.get_clip_indexes(16,16)
    for clip_index in clip_indexes:
        clip = reader.read_clip(clip_index,16)