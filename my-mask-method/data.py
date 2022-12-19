import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
import os
import random
from PIL import Image
import math
import json
import numbers
from generate_random_shapes import get_random_shape


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
    def __init__(self, p=1.0, height_range=(0.1, 0.4), width_range=(0.7, 0.9), v='histogram', percentage_covered=0.6):
        self.set_variables = True
        self.apply_transform = random.random() < p
        self.height_range = height_range
        self.width_range = width_range
        self.v = v
        self.percentage_covered = percentage_covered
    
    def __call__(self, img):

        if isinstance(img, tuple):
            img, om = img
        else:
            om = torch.ones((1, img.size(1), img.size(2)))

        if not self.apply_transform:
            return img, om
        
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

        om[:, self.y:self.y+self.h, self.x:self.x+self.w] = 0

        return img, om


class RandomShapeErasing:
    def __init__(self, p=1.0, height_range=(0.2, 0.6), width_range=(0.8, 1.0), n=6, prop=0.52, r=0.25, num_points=20):
        self.set_variables = True
        self.apply_transform = random.random() < p
        self.height_range = height_range
        self.width_range = width_range
        self.n = n
        self.prop = prop
        self.r = r
        self.num_points = num_points
    
    def __call__(self, img):

        if isinstance(img, tuple):
            img, om = img
        else:
            om = torch.ones((1, img.size(1), img.size(2)))

        if not self.apply_transform:
            return img, om
        
        if self.set_variables:
            img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]

            self.occlusion, self.occlusion_mask = get_random_shape((img_h, img_w), height_range=self.height_range, width_range=self.width_range, n=self.n, p=self.prop, r=self.r, num_points=self.num_points)

            self.set_variables = False

        
        img = (img * self.occlusion_mask) + self.occlusion

        return img, om * self.occlusion_mask


class RandomObject:
    def __init__(self, p=1.0, 
                        cars='objs\\cars', 
                        road_signs='objs\\road_signs',
                        bushes='objs\\bushes',
                        umbrellas='objs\\umbrellas'):
        self.set_variables = True
        self.apply_transform = random.random() < p
        self.cars = [os.path.join(cars, image) for image in os.listdir(cars)]
        self.road_signs = [os.path.join(road_signs, image) for image in os.listdir(road_signs)]
        self.bushes = [os.path.join(bushes, image) for image in os.listdir(bushes)]
        self.umbrellas = [os.path.join(umbrellas, image) for image in os.listdir(umbrellas)]
    
    def __call__(self, img):

        if isinstance(img, tuple):
            img, om = img
        else:
            om = torch.ones((1, img.size(1), img.size(2)))
            
        if not self.apply_transform:
            return img, om
        
        if self.set_variables:
            img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]

            obj = random.choice(list(range(4)))

            if obj == 0:
                # cars
                height_ratio_range = (0.6, 0.7)
                visible_height_range = (0.7, 1.0)
                occupied_width = 0.2

                self.occlusion = Image.open(random.choice(self.cars))
                self.occlusion = transforms.functional.to_tensor(self.occlusion)

                if self.occlusion.size(0) != 4:
                    print('Images need to be RGBA')
                    return img, om

                # 0 transparent
                # 1 visible

                h = random.randint(int(img_h * height_ratio_range[0]), int(img_h * height_ratio_range[1]))
                w = int(h * (self.occlusion.size(2) / self.occlusion.size(1)))

                self.occlusion = transforms.functional.resize(self.occlusion, (h, w))

                if random.random() < 0.5:
                    self.occlusion = transforms.functional.hflip(img=self.occlusion)

                top = random.randint(int(-img_h + (visible_height_range[0] * h)), int(-img_h + (visible_height_range[1] * h)))
                left = random.randint(int(-img_w + (occupied_width * img_w)), int(w - (occupied_width * img_w)))

                self.occlusion = transforms.functional.crop(self.occlusion, top, left, img_h, img_w)

                self.occlusion_mask = self.occlusion[3:]
                self.occlusion_mask[self.occlusion_mask < 1] = 0

                cj = transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0.5)

                self.occlusion = cj(self.occlusion[0:3]) * self.occlusion_mask
                self.occlusion_mask = 1 - self.occlusion_mask

                '''self.occlusion_mask = torch.mean(self.occlusion, dim=0, keepdim=True)
                self.occlusion_mask[self.occlusion_mask != 0] = 1
                self.occlusion_mask = 1 - self.occlusion_mask'''
            elif obj == 1:
                # road signs
                height_ratio_range = (0.9, 1.0)
                visible_height_range = (0.9, 1.0)
                occupied_width = 0.3

                self.occlusion = Image.open(random.choice(self.road_signs))
                self.occlusion = transforms.functional.to_tensor(self.occlusion)

                if self.occlusion.size(0) != 4:
                    print('Images need to be RGBA')
                    return img, om

                # 0 transparent
                # 1 visible

                h = random.randint(int(img_h * height_ratio_range[0]), int(img_h * height_ratio_range[1]))
                w = int(h * (self.occlusion.size(2) / self.occlusion.size(1)))

                self.occlusion = transforms.functional.resize(self.occlusion, (h, w))

                top = random.randint(int(-img_h + (visible_height_range[0] * h)), int(-img_h + (visible_height_range[1] * h)))
                left = random.randint(int(-img_w + (occupied_width * img_w)), int(w - (occupied_width * img_w)))

                self.occlusion = transforms.functional.crop(self.occlusion, top, left, img_h, img_w)

                self.occlusion_mask = self.occlusion[3:]
                self.occlusion_mask[self.occlusion_mask < 1] = 0

                cj = transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0)

                self.occlusion = cj(self.occlusion[0:3]) * self.occlusion_mask
                self.occlusion_mask = 1 - self.occlusion_mask

            elif obj == 2:
                # bushes
                height_ratio_range = (0.9, 1.0)
                visible_height_range = (0.4, 0.7)
                visible_height_range_edges = (0.9, 1.0)
                occupied_width = 0.2
                occupied_width_edges = 0.7

                self.occlusion = Image.open(random.choice(self.bushes))
                self.occlusion = transforms.functional.to_tensor(self.occlusion)

                if self.occlusion.size(0) != 4:
                    print('Images need to be RGBA')
                    return img, om

                # 0 transparent
                # 1 visible

                h = random.randint(int(img_h * height_ratio_range[0]), int(img_h * height_ratio_range[1]))
                w = int(h * (self.occlusion.size(2) / self.occlusion.size(1)))

                self.occlusion = transforms.functional.resize(self.occlusion, (h, w))

                left = random.randint(int(-img_w + (occupied_width * img_w)), int(w - (occupied_width * img_w)))
                
                if left > -img_w + (occupied_width_edges * img_w) and left < w - (occupied_width_edges * img_w):
                    top = random.randint(int(-img_h + (visible_height_range[0] * h)), int(-img_h + (visible_height_range[1] * h)))
                else:
                    top = random.randint(int(-img_h + (visible_height_range_edges[0] * h)), int(-img_h + (visible_height_range_edges[1] * h)))

                self.occlusion = transforms.functional.crop(self.occlusion, top, left, img_h, img_w)

                self.occlusion_mask = self.occlusion[3:]
                self.occlusion_mask[self.occlusion_mask < 1] = 0

                cj = transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0)

                self.occlusion = cj(self.occlusion[0:3]) * self.occlusion_mask
                self.occlusion_mask = 1 - self.occlusion_mask

            elif obj == 3:
                # umbrellas
                width_ratio_range = (1.5, 1.7)
                angle_range = (-40, 40)
                height_offset = (-0.05, 0)

                self.occlusion = Image.open(random.choice(self.umbrellas))
                self.occlusion = transforms.functional.to_tensor(self.occlusion)

                if self.occlusion.size(0) != 4:
                    print('Images need to be RGBA')
                    return img, om

                # 0 transparent
                # 1 visible

                w = random.randint(int(img_w * width_ratio_range[0]), int(img_w * width_ratio_range[1]))
                h = int(w * (self.occlusion.size(1) / self.occlusion.size(2)))

                self.occlusion = transforms.functional.resize(self.occlusion, (h, w))

                angle = random.randint(angle_range[0], angle_range[1])
                self.occlusion = transforms.functional.rotate(self.occlusion, angle)

                top = random.randint(int(-height_offset[1] * img_h), int(-height_offset[0] * img_h))
                left = random.randint(0, w-img_w)

                self.occlusion = transforms.functional.crop(self.occlusion, top, left, img_h, img_w)

                self.occlusion_mask = self.occlusion[3:]
                self.occlusion_mask[self.occlusion_mask < 1] = 0

                cj = transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0.5)

                self.occlusion = cj(self.occlusion[0:3]) * self.occlusion_mask
                self.occlusion_mask = 1 - self.occlusion_mask

            self.set_variables = False
        
        img = (img * self.occlusion_mask) + self.occlusion

        return img, om * self.occlusion_mask


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


def get_transform_random(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='random'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)

def get_transform_random_solid(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='random_solid'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_histogram(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='histogram'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_blur(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='blur'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_random_shape(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(RandomShapeErasing())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_random_object(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(RandomObject())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_random_shape_random_object(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ToTensor())
        transform_list.append(RandomShapeErasing())
        transform_list.append(RandomObject())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_random(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='random'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_random_solid(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='random_solid'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_histogram(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='histogram'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_blur(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(CustomRandomErasing(v='blur'))
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_random_shape(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(RandomShapeErasing())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_random_object(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(RandomObject())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def get_transform_cj_random_shape_random_object(training=True, hw=(384, 128), inc=1.05):

    transform_list = []

    if training:
        nhw = (int(hw[0]*inc), int(hw[1]*inc))
        transform_list.append(Resize(nhw))
        transform_list.append(RandomHorizontalFlip())
        transform_list.append(RandomCrop(hw))
        transform_list.append(ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_list.append(ToTensor())
        transform_list.append(RandomShapeErasing())
        transform_list.append(RandomObject())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    else:
        transform_list.append(Resize(hw))
        transform_list.append(ToTensor())
        transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)

class CustomDataset(Dataset):

    def __init__(self, root, extensions, transform_fn=get_transform_histogram, training=True):
        self.root = root
        self.training = training
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
        transform = self.transform_fn(training=self.training)

        img = self.files[idx]
        path = os.path.join(self.root, img)
        pil_img = Image.open(path).convert('RGB')

        tensor_img, occlusion_mask = transform(pil_img)

        return tensor_img, self.labels[idx], self.original_labels[idx], occlusion_mask

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


def test():
    train_path = 'C:\\Users\\Leonardo Capozzi\\Documents\\Datasets\\Market-1501-v15.09.15\\bounding_box_train'
    extensions = ['.jpg']

    dataset = CustomDataset(train_path, extensions, transform_fn=get_transform_random_shape_random_object, training=True)

    tensor_img, labels, original_labels, occlusion_mask = dataset[148]

    print((tensor_img, labels, original_labels, occlusion_mask))

    pil_img = transforms.ToPILImage()(tensor_img)
    pil_img.show()

    pil_img = transforms.ToPILImage()(occlusion_mask)
    pil_img.show()

    pil_img = transforms.ToPILImage()(transforms.functional.resize(occlusion_mask, (24, 8)))
    pil_img.show()


if __name__ == '__main__':
    test()

