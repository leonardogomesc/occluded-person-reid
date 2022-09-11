import os
from PIL import Image
import random
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
import torch

def get_transform(training=True, hw=(224, 224), inc=1.12):
    nhw = (int(hw[0]*inc), int(hw[1]*inc))

    transform_list = []

    transform_list.append(transforms.Resize(nhw))

    if training:
        top = random.randint(0, nhw[0]-hw[0])
        left = random.randint(0, nhw[1]-hw[1])

        transform_list.append(transforms.Lambda(lambda frame: transforms.functional.crop(img=frame, top=top, left=left, height=hw[0], width=hw[1])))

        if random.random() < 0.5:
            transform_list.append(transforms.Lambda(lambda frame: transforms.functional.hflip(img=frame)))
    else:
        transform_list.append(transforms.CenterCrop(hw))
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)



class VideoDataset(Dataset):
  def __init__(self, frame_root, labels_path, seq_len, sliding_window_size, transform_fn, training=True):
    self.frame_root = frame_root
    self.data = pd.read_csv(labels_path)
    self.seq_len = seq_len
    self.sliding_window_size = sliding_window_size
    self.transform_fn = transform_fn
    self.training = training

    self.clips_paths = []
    self.clips_labels = []
    self.clips_frames_chosen =[]

    for index, row in self.data.iterrows():
        startidx = 0

        frames_list = os.listdir(os.path.join(self.frame_root, row['path']))
        frames_list = sorted(frames_list, key=lambda x: int(x.split('.')[0]))

        number_of_frames = len(frames_list)
        
        while startidx + self.seq_len <= number_of_frames:
          self.clips_paths.append(row['path'])
          self.clips_labels.append(row['label'])
          self.clips_frames_chosen.append(frames_list[startidx:startidx+self.seq_len])
          startidx += self.sliding_window_size


  def __getitem__(self, idx):
    clip_path = self.clips_paths[idx]
    clip_label = int(self.clips_labels[idx])
    clip_frames_chosen = self.clips_frames_chosen[idx]

    transform = self.transform_fn(training=self.training)

    frames = [Image.open(os.path.join(self.frame_root, clip_path, frm)).convert('RGB') for frm in clip_frames_chosen]
    frames = torch.stack([transform(frm) for frm in frames], 1)

    return frames, clip_label

  def __len__(self):
        return len(self.clips_labels)


if __name__ == '__main__':

    frame_root = 'C:\\Users\\leona\\Documents\\Dataset\\hmdb51_org_frames'
    labels_path = 'C:\\Users\\leona\\Documents\\Dataset\\hmdb51_org_frames\\train_labels.csv'
    class_name_to_label_path = 'C:\\Users\\leona\\Documents\\Dataset\\hmdb51_org_frames\\class_name_to_label.json'

    seq_len = 10
    sliding_window_size = 10

    dataset = VideoDataset(frame_root, labels_path, seq_len, sliding_window_size, get_transform, training=True)

    print(dataset[0][0].size())
    print(len(dataset))

