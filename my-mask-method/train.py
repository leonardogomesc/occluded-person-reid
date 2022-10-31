from tkinter import E
import torch
import torch.nn as nn
import torch.optim as optim
import time
from data import CustomDataset, BatchSampler, get_transform_random, get_transform_random_solid, get_transform_histogram, get_transform_blur, get_transform_cj_random, get_transform_cj_random_solid, get_transform_cj_histogram, get_transform_cj_blur
from torch.utils.data import DataLoader
from models import MyModel
from triplet import batch_hard_mine_triplet, calculate_distance_matrix
import numpy as np
import sys
from datetime import datetime


def main(dataset_name):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_epochs = 160
    n_persons = 16
    n_pictures = 4

    if dataset_name == 'market_occ_random':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_cj_random
    elif dataset_name == 'market_occ_random_solid':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_cj_random_solid
    elif dataset_name == 'market_occ_histogram':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_cj_histogram
    elif dataset_name == 'market_occ_blur':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_cj_blur
    elif dataset_name == 'market_random':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_random
    elif dataset_name == 'market_random_solid':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_random_solid
    elif dataset_name == 'market_histogram':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_histogram
    elif dataset_name == 'market_blur':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_blur
    elif dataset_name == 'duke_random':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_random
    elif dataset_name == 'duke_random_solid':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_random_solid
    elif dataset_name == 'duke_histogram':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_histogram
    elif dataset_name == 'duke_blur':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_train'
        extensions = ['.jpg']
        transform_fn = get_transform_blur

    dataset = CustomDataset(train_path, extensions, transform_fn=transform_fn, training=True)
    batch_sampler = BatchSampler(dataset, n_persons, n_pictures)
    train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    num_classes = dataset.get_num_classes()

    model = MyModel(num_classes)
    model = model.to(device)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

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
            img, person_labels, person_labels_original, occlusion_mask = data

            # move to GPU
            img = img.to(device)
            person_labels = person_labels.to(device)
            occlusion_mask = occlusion_mask.to(device)

            global_feat, global_logits, fdb_feat, fdb_logits, rvd_logits, occlusion_mask = model(img, occlusion_mask)

            # global loss

            gid_loss = ce(global_logits, person_labels)
            gtri_loss = batch_hard_mine_triplet(torch.cdist(global_feat, global_feat, p=2), person_labels)

            # feature dropping branch loss

            fdbid_loss = ce(fdb_logits, person_labels)
            fdbtri_loss = batch_hard_mine_triplet(torch.cdist(fdb_feat, fdb_feat, p=2), person_labels)

            # region visibility discriminator loss

            rvd_loss = bce(rvd_logits, occlusion_mask)

            loss = gid_loss + gtri_loss + fdbid_loss + fdbtri_loss + rvd_loss

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

def main_test(dataset_name, checkpoint_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    
    if dataset_name == 'market':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_test'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\query'
        extensions = ['.jpg']
        num_classes = 751
        transform_fn = get_transform_random
    elif dataset_name == 'duke-occ':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_test'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\query'
        extensions = ['.jpg']
        num_classes = 702
        transform_fn = get_transform_random
    elif dataset_name == 'occ-reid':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\OccludedREID\\gallery'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\OccludedREID\\query'
        extensions = ['.jpg']
        num_classes = 751
        transform_fn = get_transform_cj_random
    elif dataset_name == 'part-reid':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\Partial_REID\\whole_body_images'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\Partial_REID\\partial_body_images'
        extensions = ['.jpg']
        num_classes = 751
        transform_fn = get_transform_cj_random
    elif dataset_name == 'part-ilids':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\PartialiLIDS\\gallery'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\PartialiLIDS\\query'
        extensions = ['.jpg']
        num_classes = 751
        transform_fn = get_transform_cj_random

    test_dataset = CustomDataset(test_path, extensions, transform_fn=transform_fn, training=False)
    query_dataset = CustomDataset(query_path, extensions, transform_fn=transform_fn, training=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_query = DataLoader(query_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = MyModel(num_classes)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    print(checkpoint['epoch'])
    
    print('Starting Test')
    print(dataset_name)
    print(checkpoint_path)

    test_feat_list = []
    test_labels = []

    query_feat_list = []
    query_labels = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(test_loader):
            img, person_labels, person_labels_original, occlusion_mask = data

            # forward
            img = img.to(device)

            global_feat, global_logits, fdb_feat, fdb_logits, rvd_logits, occlusion_mask = model(img)

            feat = torch.cat([global_feat, fdb_feat], dim=1)
            feat = feat.cpu()

            test_feat_list.append(feat)
            test_labels.append(person_labels_original)

        for i, data in enumerate(test_loader_query):
            img, person_labels, person_labels_original, occlusion_mask = data

            # forward
            img = img.to(device)

            global_feat, global_logits, fdb_feat, fdb_logits, rvd_logits, occlusion_mask = model(img)

            feat = torch.cat([global_feat, fdb_feat], dim=1)
            feat = feat.cpu()

            query_feat_list.append(feat)
            query_labels.append(person_labels_original)


        test_feat_list = torch.cat(test_feat_list, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        query_feat_list = torch.cat(query_feat_list, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        print(test_feat_list.size())
        print(test_labels.size())
        print(query_feat_list.size())
        print(query_labels.size())

        distance_matrix = torch.cdist(query_feat_list, test_feat_list, p=2)

        sorted_matrix = torch.argsort(distance_matrix, dim=1)

        rank1 = sorted_matrix[:, :1]
        rank1_correct = 0

        rank3 = sorted_matrix[:, :3]
        rank3_correct = 0

        rank5 = sorted_matrix[:, :5]
        rank5_correct = 0

        total = 0

        for i in range(len(query_labels)):
            q_label = query_labels[i]

            # print([q_label, test_labels[sorted_matrix[i, :10]]])

            if q_label in test_labels[rank1[i]]:
                rank1_correct += 1

            if q_label in test_labels[rank3[i]]:
                rank3_correct += 1

            if q_label in test_labels[rank5[i]]:
                rank5_correct += 1

            total += 1

        # print(rank1_correct)
        # print(total)
        # print(distance_matrix.size())
        
        print('rank1 acc: ' + str(rank1_correct / total))
        print('rank3 acc: ' + str(rank3_correct / total))
        print('rank5 acc: ' + str(rank5_correct / total))

        # map

        expanded_test_labels = test_labels.repeat(query_labels.size()[0], 1)

        sorted_labels_matrix = torch.gather(expanded_test_labels, 1, sorted_matrix)

        query_mask = torch.unsqueeze(query_labels, 1) == sorted_labels_matrix

        cum_true = torch.cumsum(query_mask, dim=1)

        num_pred_pos = torch.cumsum(torch.ones_like(cum_true), dim=1)

        p = query_mask * (cum_true / num_pred_pos)

        ap = torch.sum(p, 1)/torch.sum(query_mask, 1)

        map = torch.mean(ap)

        # map = torch.sum(p)/torch.sum(query_mask)

        print('')
        print('map: ' + str(map.item()))
   

if __name__ == '__main__':
    '''main('market_random')
    main('market_random_solid')
    main('market_histogram')
    main('market_blur')

    main('duke_random')
    main('duke_random_solid')
    main('duke_histogram')
    main('duke_blur')'''
    
    
    main_test('duke-occ', 'duke_random-221031030044.pt')
    # main_test('duke-occ', 'market_random_solid-221030104424.pt')
    # main_test('duke-occ', 'market_histogram-221030144042.pt')
    # main_test('duke-occ', 'market_blur-221030192750.pt')

    # main_test('occ-reid', 'market_occ_random-221010021816.pt')
    # main_test('part-reid', 'market_occ_random-221010021816.pt')
    # main_test('part-ilids', 'market_occ_random-221010021816.pt')
    
