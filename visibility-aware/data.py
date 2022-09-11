from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
import os
import random
from PIL import Image
import math


class CustomDataset(Dataset):

    def __init__(self, root, extensions, transform):
        self.root = root
        self.transform = transform
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
        img = self.files[idx]

        path = os.path.join(self.root, img)

        pil_img = Image.open(path)
        tensor_img = self.transform(pil_img)

        return tensor_img, self.labels[idx]

    def get_num_classes(self):
        return len(list(self.individuals.keys()))


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
    extensions = ['.jpg']

    transform = transforms.Compose([transforms.Resize((234, 117)),
                                    transforms.RandomCrop((224, 112)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

    train_loader, num_classes = get_data_loader(train_path, extensions, transform, 32, 4)


if __name__ == '__main__':
    test()

