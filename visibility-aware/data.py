import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
import os
import random
from PIL import Image
import math
import json

# color jitter ?

class Resize:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, data):
        img = data[0]
        keypoints = data[1]

        if isinstance(img, torch.Tensor):
            size = img.size()[1:]
        else:
            size = (img.size[1], img.size[0])

        img = transforms.functional.resize(img=img, size=self.new_size)

        if len(keypoints) > 0:
            ratio = torch.tensor([self.new_size[1]/size[1], self.new_size[0]/size[0]])
            keypoints = keypoints * ratio

        return img, keypoints


class RandomHorizontalFlip:
    def __init__(self):
        self.flip = random.random() < 0.5
        
    def __call__(self, data):
        if not self.flip:
            return data

        img = data[0]
        keypoints = data[1]

        if isinstance(img, torch.Tensor):
            size = img.size()[1:]
        else:
            size = (img.size[1], img.size[0])
        
        img = transforms.functional.hflip(img=img)

        if len(keypoints) > 0:
            keypoints[:, 0] = size[1] - keypoints[:, 0]

        return img, keypoints


class RandomCrop:
    def __init__(self, new_size):
        self.set_variables = True
        self.new_size = new_size
    
    def __call__(self, data):
        img = data[0]
        keypoints = data[1]

        if isinstance(img, torch.Tensor):
            size = img.size()[1:]
        else:
            size = (img.size[1], img.size[0])
        
        if self.set_variables:
            self.top = random.randint(0, size[0] - self.new_size[0])
            self.left = random.randint(0, size[1] - self.new_size[1])
            self.set_variables = False

        img = transforms.functional.crop(img=img, top=self.top, left=self.left, height=self.new_size[0], width=self.new_size[1])

        if len(keypoints) > 0:
            translation = torch.tensor([self.left, self.top])
            keypoints = keypoints - translation

        return img, keypoints


class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.set_variables = True
        self.apply_transform = random.random() < p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, data):
        if not self.apply_transform:
            return data

        img = data[0]
        keypoints = data[1]
        
        if self.set_variables:
            if isinstance(self.value, (int, float)):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value
            
            self.x, self.y, self.h, self.w, self.v = transforms.RandomErasing.get_params(img, scale=self.scale, ratio=self.ratio, value=value)

            self.set_variables = False
        
        img = transforms.functional.erase(img, self.x, self.y, self.h, self.w, self.v)

        return img, keypoints


class ToTensor:
    def __call__(self, data):
        img = data[0]
        keypoints = data[1]

        img = transforms.functional.to_tensor(img)

        return img, keypoints


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        img = data[0]
        keypoints = data[1]

        img = transforms.functional.normalize(img, self.mean, self.std)

        return img, keypoints


def get_transform(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        transform_list.append(RandomErasing())
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


class CustomDataset(Dataset):

    def __init__(self, root, pose_root, extensions, num_stripes, training=True):
        self.root = root
        self.pose_root = pose_root
        self.training = training
        self.num_stripes = num_stripes

        self.individuals = {}

        self.files = [image for image in os.listdir(self.root) if os.path.splitext(image)[1] in extensions]

        self.files.sort(key=lambda x: int(x.split('_')[0]))

        self.labels = []

        curr_label = -1
        prev_label = -1

        for f in self.files:
            label = int(f.split('_')[0])

            if label != prev_label:
                curr_label += 1
                prev_label = label

            self.labels.append(curr_label)

        for idx in range(len(self.labels)):
            label = self.labels[idx]

            ind_list = self.individuals.get(label, [])
            ind_list.append(idx)

            self.individuals[label] = ind_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        transform = get_transform(training=self.training)

        img = self.files[idx]

        print(img)

        path = os.path.join(self.root, img)
        pose_path = os.path.join(self.pose_root, f'{os.path.splitext(img)[0]}.json')

        pil_img = Image.open(path)

        with open(pose_path, 'r') as f:
            keypoints = json.load(f)
        
        keypoints = self.transform_keypoints(keypoints)

        tensor_img, keypoints = transform((pil_img, keypoints))

        occlusion_labels = self.generate_occlusion_labels(keypoints, tensor_img.size(1))

        return tensor_img, self.labels[idx], occlusion_labels

    def get_num_classes(self):
        return len(list(self.individuals.keys()))
    
    def transform_keypoints(self, keypoints):
        new_kp = []

        for dic in keypoints:
            for key, value in dic.items():
                if len(value) == 3 and value[2] == 1:
                    new_kp.append(value[:2])
        
        return torch.tensor(new_kp)
    
    def generate_occlusion_labels(self, keypoints, h):
        occlusion_labels = [0] * self.num_stripes

        for kp in keypoints:
            label = math.floor(self.num_stripes * (kp[1] / h))

            # just to be safe
            if label < 0:
                label = 0

            if label >= self.num_stripes:
                label = self.num_stripes - 1
            
            occlusion_labels[label] = 1
        
        return occlusion_labels


class BatchSampler(Sampler):
    def __init__(self, dataset, n_persons, n_pictures):
        self.dataset = dataset
        self.n_persons = n_persons
        self.n_pictures = n_pictures

        self.len = math.ceil(len(self.dataset) / (self.n_persons * self.n_pictures))

    def __iter__(self):
        for _ in range(self.len):
            anchors = []
            keys = list(self.dataset.individuals.keys())

            if len(keys) >= self.n_persons:
                keys = random.sample(keys, self.n_persons)

            for key in keys:
                objects = self.dataset.individuals[key]

                if len(objects) >= self.n_pictures:
                    objects = random.sample(objects, self.n_pictures)

                anchors.extend(objects)

            yield anchors

    def __len__(self):
        return self.len


def get_data_loader(data_path, extensions, transform, n_persons, n_pictures):

    dataset = CustomDataset(data_path, extensions, transform)

    batch_sampler = BatchSampler(dataset, n_persons, n_pictures)

    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    return data_loader, dataset.get_num_classes()


def get_mask_data_loader(data_path, extensions, transform, batch_size):

    dataset = CustomDataset(data_path, extensions, transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader


def get_test_data_loader(data_path, extensions, transform, batch_size):

    dataset = CustomDataset(data_path, extensions, transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return data_loader


def test():
    train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
    train_pose_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15-pose\\bounding_box_train'
    extensions = ['.jpg']

    dataset = CustomDataset(train_path, train_pose_path, extensions, 6, training=True)

    print(dataset[148])


if __name__ == '__main__':
    test()

