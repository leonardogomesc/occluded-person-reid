import torch
import torch.nn as nn
import torch.optim as optim
import time
from data import CustomDataset, BatchSampler
from torch.utils.data import DataLoader
from models import MyModel
from triplet import batch_hard_mine_triplet, calculate_distance_matrix
import numpy as np
import sys
from datetime import datetime


def main(dataset_name):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    n_epochs = 120
    n_persons = 16
    n_pictures = 4

    lr = 3.5e-4

    if dataset_name == 'market-occ':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        margin = 0.3
        nk = 14
        lamb_cls = 0.3
        lamb_tri = 1
        lamb_div = 1
    elif dataset_name == 'duke-occ':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_train'
        extensions = ['.jpg']
        margin = 0.3
        nk = 14
        lamb_cls = 0.3
        lamb_tri = 1
        lamb_div = 1
    elif dataset_name == 'market':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        margin = 0.3
        nk = 6
        lamb_cls = 1
        lamb_tri = 1
        lamb_div = 1
    elif dataset_name == 'duke':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\bounding_box_train'
        extensions = ['.jpg']
        margin = 0.3
        nk = 14
        lamb_cls = 1
        lamb_tri = 1
        lamb_div = 1
    

    dataset = CustomDataset(train_path, extensions, training=True)
    batch_sampler = BatchSampler(dataset, n_persons, n_pictures)
    train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    num_classes = dataset.get_num_classes()

    model = MyModel(num_classes, nk)
    model = model.to(device)

    ce = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)

    save_path = datetime.now().strftime(f'{dataset_name}-%y%m%d%H%M%S.pt')

    print('Starting Training')

    train_loss_hist = []
    train_loss_min = np.inf

    for epoch in range(n_epochs):
        t_start = time.time()

        ######################
        # training the model #
        ######################

        train_loss = 0.0
        
        model.train()

        for batch_idx, data in enumerate(train_loader):
            img, person_labels, person_labels_original = data

            # move to GPU
            img = img.to(device)
            person_labels = person_labels.to(device)

            enc_feat_gap, enc_feat_logits, dec_feat, dec_logits_list = model(img)

            # loss en
            l_cls = ce(enc_feat_logits, person_labels)

            distance_matrix = torch.cdist(enc_feat_gap, enc_feat_gap, p=2)
            l_tri = batch_hard_mine_triplet(distance_matrix, person_labels, margin=margin)

            l_en = (lamb_cls * l_cls) + (lamb_tri * l_tri)

            # loss dis

            l_cls = 0.0

            for dec_logits in dec_logits_list:
                l_cls += ce(dec_logits, person_labels)
            
            l_cls /= len(dec_logits_list)

            l_tri = 0.0

            for f in dec_feat:
                distance_matrix = torch.cdist(f, f, p=2)
                l_tri += batch_hard_mine_triplet(distance_matrix, person_labels, margin=margin)

            l_tri /= len(dec_feat)
            
            l_dis = (lamb_cls * l_cls) + (lamb_tri * l_tri)

            # loss div
            dec_feat = dec_feat.permute(1, 0, 2)

            dec_feat_inner_matrix = torch.sum(dec_feat.unsqueeze(2)*dec_feat.unsqueeze(1), dim=3)

            dec_feat_norm = torch.norm(dec_feat, p=2, dim=2)

            dec_feat_norm_matrix = dec_feat_norm.unsqueeze(2) * dec_feat_norm.unsqueeze(1)

            dec_feat_matrix = dec_feat_inner_matrix / dec_feat_norm_matrix

            eye = 1 - torch.eye(dec_feat_matrix.size(1), device=device)

            l_div = dec_feat_matrix * eye

            l_div = l_div.view(l_div.size(0), -1)

            l_div = torch.sum(l_div, dim=1)

            l_div /= dec_feat_matrix.size(1) * (dec_feat_matrix.size(1) - 1)

            l_div = torch.mean(l_div)

            loss = l_en + l_dis + (lamb_div * l_div)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            sys.stdout.write("\r" + '........ mini-batch {} loss: {:.3f}'.format(batch_idx + 1, loss.item()))
            sys.stdout.flush()
        
        scheduler.step()
        
        train_loss /= batch_idx + 1
    
        train_loss_hist.append(train_loss)

        t_end = time.time()

        # printing training/validation statistics 
        print('\n')
        print(f'Epoch: {epoch}')
        print(f'\tTraining Loss: {train_loss}')
        print(f'Total time: {t_end - t_start} s')
        
        ## saving the model if loss has decreased
        if train_loss < train_loss_min:
            print('Saving model')
            torch.save({'epoch': epoch,
                        'train_loss_hist': train_loss_hist,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, save_path)
            train_loss_min = train_loss
        
        print('\n\n')



if __name__ == '__main__':
    main('market-occ')
    # main('duke-occ')
    # main('market')
    # main('duke')

