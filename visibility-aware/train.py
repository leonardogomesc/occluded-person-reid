import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from utils import train
from data import VideoDataset, get_transform
from torch.utils.data import DataLoader


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_epochs = 10
    n_classes = 51

    model = models.video.r3d_18(pretrained=True, progress=True)
    model.fc = nn.Linear(512, n_classes)

    for param in model.stem.parameters():
        param.requires_grad = False

    for param in model.layer1.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = False
    
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_path = 'model.pth'

    frame_root = 'C:\\Users\\leona\\Documents\\Dataset\\hmdb51_org_frames'
    train_labels_path = 'C:\\Users\\leona\\Documents\\Dataset\\hmdb51_org_frames\\train_labels.csv'
    valid_labels_path = 'C:\\Users\\leona\\Documents\\Dataset\\hmdb51_org_frames\\valid_labels.csv'
    test_labels_path = 'C:\\Users\\leona\\Documents\\Dataset\\hmdb51_org_frames\\test_labels.csv'

    seq_len = 10
    sliding_window_size = 10

    train_dataset = VideoDataset(frame_root, train_labels_path, seq_len, sliding_window_size, get_transform, training=True)
    valid_dataset = VideoDataset(frame_root, valid_labels_path, seq_len, sliding_window_size, get_transform, training=False)
    
    batch_size = 8

    # Create Dataloader to read the data within batch sizes and put into memory. 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    train(n_epochs, model, loss_fn, optimizer, device, save_path, train_loader, valid_loader, n_classes)


if __name__ == '__main__':
    main()

