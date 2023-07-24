import os
# os.environ['CUDA_VISIBLE_DIVICES']='6'
from model.i3d import InceptionI3d
from dataset.videoReader import VideoLoader
import torch
from dataset.transforms.video import Resize, Compose,CenterCrop
import numpy as np
import argparse
from utils.logger_utils import get_logger
from tqdm import tqdm

def get_batch_data(vr:VideoLoader, batch_clip_indexes):
    batch_data = vr.read_clip(batch_clip_indexes)
    batch_data = np.array_split(batch_data,len(batch_clip_indexes))
    # batch_data=[]
    # for clip_index in batch_clip_indexes:
    #     clip=vr.read_clip(clip_index, clip_len)
    #     batch_data.append(clip)
    batch_data=np.stack(batch_data,axis=0)
    return batch_data

def split_batch(clip_indexes, batch_size):
    # n_batch=len(clip_indexes)//batch_size
    indices=np.arange(batch_size,len(clip_indexes),step=batch_size)
    batch_indexes=np.split(clip_indexes,indices,axis=0)
    # batch_indexes.append(clip_indexes[-1:])
    return batch_indexes

def feature_extraction(data_root, batch_size, clip_len:int, clip_stride:int, 
                       img_size:tuple, keep_ori_ratio, fps, modal, out_dir, 
                       model_path, log_dir=None):
    assert data_root is not None and out_dir is not None, 'specify the data_root and out_dir'

    # load model and logger
    logger = get_logger(log_dir)

    if modal == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(model_path))
    # i3d.cuda()
    i3d.train(False)
    logger.info('model loaded to GPU')


    transforms = Resize((224,224))

    # feature extraction
    fn_list = os.listdir(data_root)
    for i,fn in tqdm(enumerate(fn_list),total=len(fn_list)):
        vid = fn.split('.')[0]
        video_path = os.path.join(data_root,fn)
        video_loader=VideoLoader(video_path,fps,False, img_size[0],img_size[1],transforms=transforms, keep_ori_ratio=keep_ori_ratio)
        clip_indexes=video_loader.get_clip_indexes(clip_len, clip_stride)
        batch_indexes=split_batch(clip_indexes,batch_size)
        video_feature = []
        logger.info(f'the clip {vid} is split into {len(batch_indexes)} batches')
    # feature extraction for each batch
        for batch_index in batch_indexes:
            if len(batch_index)>0:
                batch_data= get_batch_data(video_loader,batch_index,clip_len)
                batch_data=torch.from_numpy(batch_data)#.cuda()
                batch_data = batch_data.permute(0,4,1,2,3)
                batch_feature = i3d.extract_features(batch_data).data.cpu().numpy()[:,:,0,0,0]
                video_feature.append(batch_feature)
    
    #feature save
        video_feature = np.concatenate(video_feature,axis=0)
        save_path=os.path.join(out_dir,vid+'.npz')
        np.savez_compressed(save_path,video_feature)
        logger.info(f'[{i+1}/{len(fn_list)}] feature for {vid} with {len(video_feature)} clips is saved into {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default='/mnt/cephfs/dataset/activitynet_video/all_videos')
    parser.add_argument('--modal',type=str, default='rgb')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--clip_stride',type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dir', type=str, default='/mnt/cephfs/dataset/activitynet_captions/features')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--model_path',type=str, default='/mnt/cephfs/home/youzeng/project/pytorch-i3d/models/rgb_imagenet.pt')
    parser.add_argument('--log_dir',type=str, default=None)
    args = parser.parse_args()

    feature_extraction(args.data_root,args.batch_size,args.clip_len,args.clip_stride,(256,340),
                       keep_ori_ratio=True,fps=args.fps,modal=args.modal,out_dir=args.out_dir,
                       model_path=args.model_path, log_dir=args.log_dir)

