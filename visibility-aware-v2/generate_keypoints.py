import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import os
import json
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

COCO_PERSON_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

color_palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0,
                 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0,
                 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0,
                 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0,
                 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0,
                 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0,
                 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192,
                 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192,
                 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128,
                 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128,
                 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192,
                 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192,
                 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128,
                 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128,
                 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160,
                 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192,
                 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128,
                 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160,
                 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224,
                 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224,
                 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160,
                 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64,
                 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0,
                 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192,
                 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128,
                 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128,
                 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128,
                 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128,
                 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32,
                 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224,
                 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192,
                 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96,
                 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]

person_label = 1
person_threshold = 0.8

batch_size = 8  # can be increased if all frames have the same resolution

accepted_extensions = ['.jpg', '.jpeg', '.png']  # file formats of the frames

root_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15'  # root directory of saved frames
target_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15-pose'  # output path for the pose information (mimics file organization of root path)
draw_pose_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15-drawings'  # output for the frames with the pose drawings (mimics file organization of root path)
draw_heatmap_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15-heatmap'  # output for the frames with the heatmap drawings (mimics file organization of root path)

draw_pose = True  # output drawings or not
draw_heatmap = True  # output heatmap drawings or not


class CustomDataset(Dataset):

    def __init__(self, root, extensions, transform):
        self.root = root
        self.extensions = extensions
        self.transform = transform

        self.files = []

        for path, subdirs, files in os.walk(self.root):
            for name in files:
                file_path = os.path.join(path, name)
                file_path = os.path.relpath(file_path, self.root)

                if os.path.splitext(file_path)[1] in self.extensions:
                    self.files.append(file_path)

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.root, self.files[idx])

        frame = Image.open(frame_path).convert('RGB')
        tensor_frame = self.transform(frame)

        return tensor_frame, self.files[idx]


def draw_line(kp1, kp2, dic, draw, color):
    if dic[kp1][2] == 1 and dic[kp2][2] == 1:
        x1 = dic[kp1][0]
        y1 = dic[kp1][1]
        x2 = dic[kp2][0]
        y2 = dic[kp2][1]
        draw.line([x1, y1, x2, y2], fill=color)


def draw_keypoints(img_path, dics):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    for p in dics:

        # face
        draw_line('left_ear', 'left_eye', p, draw, (255, 0, 0))
        draw_line('left_eye', 'nose', p, draw, (255, 0, 0))
        draw_line('nose', 'right_eye', p, draw, (255, 0, 0))
        draw_line('right_eye', 'right_ear', p, draw, (255, 0, 0))

        # arms
        draw_line('left_wrist', 'left_elbow', p, draw, (0, 255, 0))
        draw_line('left_elbow', 'left_shoulder', p, draw, (0, 255, 0))
        draw_line('left_shoulder', 'right_shoulder', p, draw, (0, 255, 0))
        draw_line('right_shoulder', 'right_elbow', p, draw, (0, 255, 0))
        draw_line('right_elbow', 'right_wrist', p, draw, (0, 255, 0))

        # legs
        draw_line('left_ankle', 'left_knee', p, draw, (0, 0, 255))
        draw_line('left_knee', 'left_hip', p, draw, (0, 0, 255))
        draw_line('left_hip', 'right_hip', p, draw, (0, 0, 255))
        draw_line('right_hip', 'right_knee', p, draw, (0, 0, 255))
        draw_line('right_knee', 'right_ankle', p, draw, (0, 0, 255))

        bb = p['bounding_box']

        x1 = bb[0]
        y1 = bb[1]
        x2 = bb[0] + bb[2]
        y2 = bb[1]
        draw.line((x1, y1, x2, y2), fill=(0, 255, 0))

        x1 = bb[0] + bb[2]
        y1 = bb[1]
        x2 = bb[0] + bb[2]
        y2 = bb[1] + bb[3]
        draw.line((x1, y1, x2, y2), fill=(0, 255, 0))

        x1 = bb[0] + bb[2]
        y1 = bb[1] + bb[3]
        x2 = bb[0]
        y2 = bb[1] + bb[3]
        draw.line((x1, y1, x2, y2), fill=(0, 255, 0))

        x1 = bb[0]
        y1 = bb[1] + bb[3]
        x2 = bb[0]
        y2 = bb[1]
        draw.line((x1, y1, x2, y2), fill=(0, 255, 0))

    return img


'''def generate_keypoint_map(img_path, dics, r=5):
    img = Image.open(img_path)
    width, height = img.size

    img = Image.new('L', (width, height))
    draw = ImageDraw.Draw(img)

    for p in dics:
        for key in p.keys():
            coords = p[key]

            if coords[2] == 1:
                x = coords[0]
                y = coords[1]

                draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    return img'''


def generate_keypoint_map(img_path, dics, r=4):
    img = Image.open(img_path)
    width, height = img.size

    img = Image.new('P', (width, height))
    draw = ImageDraw.Draw(img)

    for p in dics:
        for ki in range(len(COCO_PERSON_KEYPOINT_NAMES)):
            coords = p[COCO_PERSON_KEYPOINT_NAMES[ki]]

            if coords[2] == 1:
                x = coords[0]
                y = coords[1]

                draw.ellipse([x-r, y-r, x+r, y+r], fill=ki+1)

    img.putpalette(color_palette)
    img = img.convert('RGB')

    return img


def main():
    dataset = CustomDataset(root_path, accepted_extensions, transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    keypoints_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights, progress=True)
    keypoints_model = keypoints_model.to(device)
    keypoints_model.eval()

    with torch.no_grad():
        for frames, paths in tqdm(dataloader):
            
            skip_batch = True
            
            for path in paths:
                tp = os.path.join(target_path, path)
                tp_dir, tp_name = os.path.split(tp)
                tp_name = os.path.splitext(tp_name)[0]
                
                if not os.path.exists(os.path.join(tp_dir, f'{tp_name}.json')):
                    skip_batch = False
                    break
            
            if skip_batch:
                continue
                
            frames = frames.to(device)
            predictions = keypoints_model(frames)

            for pred, path in zip(predictions, paths):
                boxes = pred['boxes'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                keypoints = pred['keypoints'].cpu().numpy()
                keypoints_scores = pred['keypoints_scores'].cpu().numpy()

                dics = []

                for p in range(labels.shape[0]):
                    if labels[p] == person_label and scores[p] > person_threshold:
                        dic = {}

                        # bounding box
                        bounding_box = np.array([boxes[p][0], boxes[p][1], boxes[p][2] - boxes[p][0], boxes[p][3] - boxes[p][1]])
                        dic['bounding_box'] = bounding_box.tolist()

                        # keypoints
                        for ki in range(len(COCO_PERSON_KEYPOINT_NAMES)):
                            person_kp = keypoints[p][ki].tolist()
                            visibility = keypoints_scores[p][ki]

                            if visibility < 0:
                                person_kp[2] = 0
                            else:
                                person_kp[2] = 1

                            dic[COCO_PERSON_KEYPOINT_NAMES[ki]] = person_kp

                        dics.append(dic)

                tp = os.path.join(target_path, path)
                tp_dir, tp_name = os.path.split(tp)
                tp_name = os.path.splitext(tp_name)[0]

                os.makedirs(tp_dir, exist_ok=True)

                with open(os.path.join(tp_dir, f'{tp_name}.json'), 'w') as json_file:
                    json.dump(dics, json_file, indent=4)

                if draw_pose:
                    dp = os.path.join(draw_pose_path, path)
                    dp_dir, dp_name = os.path.split(dp)
                    dp_name = os.path.splitext(dp_name)[0]

                    os.makedirs(dp_dir, exist_ok=True)

                    draw_keypoints(os.path.join(root_path, path), dics).save(os.path.join(dp_dir, f'{dp_name}.png'))

                if draw_heatmap:
                    hp = os.path.join(draw_heatmap_path, path)
                    hp_dir, hp_name = os.path.split(hp)
                    hp_name = os.path.splitext(hp_name)[0]

                    os.makedirs(hp_dir, exist_ok=True)

                    generate_keypoint_map(os.path.join(root_path, path), dics).save(os.path.join(hp_dir, f'{hp_name}.png'))


if __name__ == '__main__':
    main()
