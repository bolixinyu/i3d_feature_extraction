import torch
import os
import torch.nn as nn
from model.i3d import InceptionI3d
from feature_extraction import split_batch, get_batch_data
from dataset.video_datasets import VideoDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.logger_utils import get_logger
import numpy as np
import argparse
from dataset.transforms.video import Resize
from dataset.videoReader import VideoLoader
from tqdm import tqdm
from torch.distributed import init_process_group
import eventlet

eventlet.monkey_patch()

init_process_group("nccl")

# class Extractor(nn.Module):
#     def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', 
#                  name='inception_i3d', in_channels=3, dropout_keep_prob=0.5, model_path=None):
#         super().__init__()
#         self.i3d = InceptionI3d(num_classes, spatial_squeeze, final_endpoint, name, in_channels,dropout_keep_prob)
#         self.i3d.load_state_dict(torch.load(model_path))
        
#     def forward(self, data):
#         feature = self.i3d.extract_features(data)
#         return feature[:,:,0,0,0]

def extraction(logger, vid,video_path,fps,keep_ori_ratio, clip_len, clip_stride,
               bs,device,model):
    vr=VideoLoader(video_path,fps,False, None,None,transforms=Resize((224,224)), keep_ori_ratio=keep_ori_ratio)
    logger.info(f'{vid} processing')
    clip_indexes = vr.get_clip_indexes(clip_len, clip_stride)
    batch_indexes = split_batch(clip_indexes,bs)
    # batch_indexes = torch.tensor(batch_indexes)
    video_feature = []
# feature extraction for each batch
    for batch_index in batch_indexes:
        if len(batch_index)>0:
            batch_data= get_batch_data(vr,batch_index)
            batch_data=torch.from_numpy(batch_data).to(device)
            batch_data = batch_data.permute(0,4,1,2,3)
            batch_feature = model.module.extract_features(batch_data)
            batch_feature = batch_feature.data.cpu().numpy()[:,:,0,0,0]
            video_feature.append(batch_feature)
    video_feature = np.concatenate(video_feature,axis=0)
    return video_feature

def main(data_root, video_bs, bs, clip_len:int, clip_stride:int, 
        img_size:tuple, keep_ori_ratio, fps, modal, out_dir, 
        model_path, local_rank=0, log_dir=None):
    assert data_root is not None and out_dir is not None, 'specify the data_root and out_dir'

    local_rank=torch.distributed.get_rank()
    logger = get_logger(log_dir)

    dataset = VideoDataset(data_root, out_dir)
    sampler= DistributedSampler(dataset,shuffle=False)
    dataloader=DataLoader(dataset,video_bs,shuffle=False, drop_last=False, sampler=sampler)
    existing_features = os.listdir(out_dir)
    if modal == 'rgb':
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(model_path))
    else:
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(model_path))
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda",local_rank)
    i3d.train(False)
    i3d.to(device)
    model = nn.parallel.DistributedDataParallel(i3d, device_ids=[local_rank],output_device=local_rank)
    
    for vid, video_path in tqdm(dataloader):
        # if (vid[0]=='v_rGFhqcxeVIg'):
        #     logger.info(f'{vid[0]} skip')
        #     continue
        if (vid[0]+'.npz') in existing_features:
            logger.info(f'{vid[0]} exists')
            continue
        try:
            with eventlet.Timeout(3000,True):
                video_feature=extraction(logger, vid[0],video_path[0],fps,keep_ori_ratio, clip_len, clip_stride,
                    bs,device,model)
        except Exception:
            logger.error(f'{vid} process timeout, skip!!!')
            with open('./skip_vides','a') as f:
                f.write(vid[0]+'\n')
                f.close()
            continue
    #     vr=VideoLoader(video_path[0],fps,False, None,None,transforms=Resize((224,224)), keep_ori_ratio=keep_ori_ratio)
    #     logger.info(f'{vid} processing')
    #     clip_indexes = vr.get_clip_indexes(clip_len, clip_stride)
    #     batch_indexes = split_batch(clip_indexes,bs)
    #     # batch_indexes = torch.tensor(batch_indexes)
    #     video_feature = []
    # # feature extraction for each batch
    #     for batch_index in batch_indexes:
    #         if len(batch_index)>0:
    #             batch_data= get_batch_data(vr,batch_index)
    #             batch_data=torch.from_numpy(batch_data).to(device)
    #             batch_data = batch_data.permute(0,4,1,2,3)
    #             batch_feature = model.module.extract_features(batch_data)
    #             batch_feature = batch_feature.data.cpu().numpy()[:,:,0,0,0]
    #             video_feature.append(batch_feature)
    #     video_feature = np.concatenate(video_feature,axis=0)
        save_path=os.path.join(out_dir,vid[0]+'.npz')
        np.savez_compressed(save_path,video_feature)
        logger.info(f'{vid} with {len(video_feature)} clips is saved into {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default='/mnt/cephfs/dataset/activitynet_video/all_videos')
    parser.add_argument('--modal',type=str, default='rgb')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--clip_stride',type=int, default=16)
    parser.add_argument('--video_bs',type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='/mnt/cephfs/dataset/activitynet_captions/features')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--model_path',type=str, default='/mnt/cephfs/home/youzeng/project/pytorch-i3d/models/rgb_imagenet.pt')
    parser.add_argument('--log_dir',type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    main(args.data_root, args.video_bs, args.batch_size,args.clip_len,args.clip_stride,(256,340),
                       keep_ori_ratio=True,fps=args.fps,modal=args.modal,out_dir=args.out_dir,
                       model_path=args.model_path, log_dir=args.log_dir,local_rank=args.local_rank)
