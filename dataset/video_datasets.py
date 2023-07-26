from torch.utils.data import Dataset
from dataset.videoReader import VideoLoader
import os
from dataset.transforms.video import Resize

class VideoDataset(Dataset):
    def __init__(self, data_root, feature_dir) -> None:
        super().__init__()
        self.data_root = data_root
        self.feature_dir = feature_dir
        self.data = self.set_data()
    
    def set_data(self):
        vpath_list= os.listdir(self.data_root)
        exist_list = os.listdir(self.feature_dir)
        data = []
        for vpath in vpath_list:
            vid = os.path.basename(vpath)
            vid=vid.split('.')[0]
            if (vid+'.npz') in (exist_list):
                continue
            data.append(os.path.join(self.data_root,vpath))
        return data
    
    def __getitem__(self, index):
        video_path = self.data[index]
        vid = os.path.basename(video_path)
        vid=vid.split('.')[0]
        return vid, video_path
    
    def __len__(self):
        return len(self.data)