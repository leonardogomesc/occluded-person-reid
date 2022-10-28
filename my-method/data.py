import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
import os
import random
from PIL import Image
import math
import json
import numbers


class Resize:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return transforms.functional.resize(img=img, size=self.new_size)


class RandomHorizontalFlip:
    def __init__(self):
        self.flip = random.random() < 0.5
        
    def __call__(self, img):
        if not self.flip:
            return img

        return transforms.functional.hflip(img=img)


class RandomCrop:
    def __init__(self, new_size):
        self.set_variables = True
        self.new_size = new_size
    
    def __call__(self, img):

        if isinstance(img, torch.Tensor):
            size = img.size()[1:]
        else:
            size = (img.size[1], img.size[0])
        
        if self.set_variables:
            self.top = random.randint(0, size[0] - self.new_size[0])
            self.left = random.randint(0, size[1] - self.new_size[1])
            self.set_variables = False

        img = transforms.functional.crop(img=img, top=self.top, left=self.left, height=self.new_size[0], width=self.new_size[1])

        return img


class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.set_variables = True
        self.apply_transform = random.random() < p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        if not self.apply_transform:
            return img
        
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

        return img

class CustomRandomErasing:
    def __init__(self, num_stripes, p=1.0, height_range=(0.1, 0.4), width_range=(0.7, 0.9), v='histogram', percentage_covered=0.6):
        self.num_stripes = num_stripes
        self.set_variables = True
        self.apply_transform = random.random() < p
        self.height_range = height_range
        self.width_range = width_range
        self.v = v
        self.percentage_covered = percentage_covered
    
    def __call__(self, img):
        occlusion_labels = [1] * self.num_stripes

        if not self.apply_transform:
            return img, torch.tensor(occlusion_labels)
        
        if self.set_variables:

            img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]

            self.h = random.randint(int(img_h * self.height_range[0]), int(img_h * self.height_range[1]))
            self.w = random.randint(int(img_w * self.width_range[0]), int(img_w * self.width_range[1]))

            self.x = random.randint(0, img_w - self.w)
            self.y = random.randint(0, img_h - self.h)

            if self.v == 'random':
                self.v = torch.rand(img_c, self.h, self.w)
            elif self.v == 'random_solid':
                self.v = torch.rand(img_c, 1, 1)
            elif self.v == 'histogram':
                pil_img = transforms.ToPILImage()(img)

                r, g, b = pil_img.split()

                r = r.histogram()
                g = g.histogram()
                b = b.histogram()

                r = random.choices(list(range(256)), weights=r, k=1)[0]
                g = random.choices(list(range(256)), weights=g, k=1)[0]
                b = random.choices(list(range(256)), weights=b, k=1)[0]

                self.v = torch.tensor([r/255, g/255, b/255])[:, None, None]
            elif self.v == 'blur':
                factor = 1.0

                # Determine size of blurring kernel based on input image

                kW = int(self.w/factor)
                kH = int(self.h/factor)

                # Ensure width and height of kernel are odd
                if kW % 2 == 0:
                    kW -= 1

                if kH % 2 == 0:
                    kH -= 1
                
                if kW < 1:
                    kW = 1
                
                if kH < 1:
                    kH = 1

                # Apply a Gaussian blur to the input image using our computed kernel size
                self.v = transforms.functional.gaussian_blur(img[:, self.y:self.y+self.h, self.x:self.x+self.w], (kW, kH))
            else:
                self.v = torch.tensor(list(self.v))[:, None, None]
            
            self.set_variables = False

        
        img[:, self.y:self.y+self.h, self.x:self.x+self.w] = self.v

        stripe_h = img.size(1) / self.num_stripes

        # debug

        '''for i in range(self.num_stripes-1):
            img[:, (i + 1) * int(stripe_h), :] = torch.tensor([1.0, 0, 0])[:, None]'''

        # calculate the stripes that are occluded

        first_stripe = math.floor(self.y / stripe_h)

        last_stripe = math.floor((self.y + self.h) / stripe_h)

        # check if first_stripe is included

        first_stripe_extra = stripe_h - (self.y % stripe_h)

        if first_stripe_extra > self.h:
            first_stripe_extra = self.h
        
        if first_stripe_extra / stripe_h < self.percentage_covered:
            first_stripe += 1
        

        # check if last stripe is included

        last_stripe_extra = (self.y + self.h) % stripe_h

        if last_stripe_extra > self.h:
            last_stripe_extra = self.h
        
        if last_stripe_extra / stripe_h < self.percentage_covered:
            last_stripe -= 1

        
        # erase is too small

        if first_stripe > last_stripe:
            return img, torch.tensor(occlusion_labels)
        

        n_stripes = last_stripe - first_stripe + 1

        for i in range(n_stripes):
            occlusion_labels[first_stripe + i] = 0


        return img, torch.tensor(occlusion_labels)


class ToTensor:
    def __call__(self, img):
        return transforms.functional.to_tensor(img)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        if isinstance(data, tuple):
            return transforms.functional.normalize(data[0], self.mean, self.std), data[1]
        else:
            return transforms.functional.normalize(data, self.mean, self.std), torch.tensor([])



class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        brightness = self._check_input(brightness, "brightness")
        contrast = self._check_input(contrast, "contrast")
        saturation = self._check_input(saturation, "saturation")
        hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

        self.fn_idx, self.brightness_factor, self.contrast_factor, self.saturation_factor, self.hue_factor = transforms.ColorJitter.get_params(brightness, 
                                                                                                                                                contrast, 
                                                                                                                                                saturation, 
                                                                                                                                                hue)


    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None

        return value

    def __call__(self, img):

        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness_factor is not None:
                img = transforms.functional.adjust_brightness(img, self.brightness_factor)
            elif fn_id == 1 and self.contrast_factor is not None:
                img = transforms.functional.adjust_contrast(img, self.contrast_factor)
            elif fn_id == 2 and self.saturation_factor is not None:
                img = transforms.functional.adjust_saturation(img, self.saturation_factor)
            elif fn_id == 3 and self.hue_factor is not None:
                img = transforms.functional.adjust_hue(img, self.hue_factor)

        return img


def get_transform_random(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='random'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)

def get_transform_random_solid(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='random_solid'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_histogram(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='histogram'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_blur(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='blur'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_random(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='random'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_random_solid(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='random_solid'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_histogram(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='histogram'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_blur(num_stripes, training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(num_stripes, v='blur'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


class CustomDataset(Dataset):

    def __init__(self, root, extensions, num_stripes, transform_fn=get_transform_histogram, training=True):
        self.root = root
        self.training = training
        self.num_stripes = num_stripes
        self.transform_fn = transform_fn

        self.individuals = {}

        self.files = [image for image in os.listdir(self.root) if os.path.splitext(image)[1] in extensions]

        self.files.sort(key=lambda x: int(x.split('_')[0]))

        self.labels = []
        self.original_labels = []

        curr_label = -1
        prev_label = None

        for f in self.files:
            label = int(f.split('_')[0])

            if label != prev_label:
                curr_label += 1
                prev_label = label

            self.labels.append(curr_label)
            self.original_labels.append(label)

        for idx in range(len(self.labels)):
            label = self.labels[idx]

            ind_list = self.individuals.get(label, [])
            ind_list.append(idx)

            self.individuals[label] = ind_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        transform = self.transform_fn(self.num_stripes, training=self.training)

        img = self.files[idx]
        path = os.path.join(self.root, img)
        pil_img = Image.open(path)

        tensor_img, occlusion_labels = transform(pil_img)

        return tensor_img, self.labels[idx], self.original_labels[idx], occlusion_labels

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

    dataset = CustomDataset(train_path, extensions, 6, transform_fn=get_transform_blur, training=True)

    tensor_img, labels, original_labels, occlusion_labels = dataset[148]

    pil_img = transforms.ToPILImage()(tensor_img)
    pil_img.show()

    print((tensor_img, labels, original_labels, occlusion_labels))


if __name__ == '__main__':
    test()

