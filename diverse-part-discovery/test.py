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


def main_test(dataset_name, checkpoint_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 64

    if dataset_name == 'market':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_test'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\query'
        extensions = ['.jpg']
        nk = 6
        num_classes = 751
    elif dataset_name == 'duke':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\bounding_box_test'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\query'
        extensions = ['.jpg']
        nk = 14
        num_classes = 702
    elif dataset_name == 'duke-occ':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_test'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\query'
        extensions = ['.jpg']
        nk = 14
        num_classes = 702
    elif dataset_name == 'occ-reid':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\OccludedREID\\gallery'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\OccludedREID\\query'
        extensions = ['.jpg']
        nk = 14
        num_classes = 751
    elif dataset_name == 'part-reid':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\Partial_REID\\whole_body_images'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\Partial_REID\\partial_body_images'
        extensions = ['.jpg']
        nk = 14
        num_classes = 751
    elif dataset_name == 'part-ilids':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\PartialiLIDS\\gallery'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\PartialiLIDS\\query'
        extensions = ['.jpg']
        nk = 14
        num_classes = 751


    test_dataset = CustomDataset(test_path, extensions, training=False)
    query_dataset = CustomDataset(query_path, extensions, training=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_query = DataLoader(query_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = MyModel(num_classes, nk)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    print('Starting Test')
    print(dataset_name)

    test_features = []
    test_labels = []

    query_features = []
    query_labels = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(test_loader):
            img, person_labels, person_labels_original = data

            # forward
            img = img.to(device)

            enc_feat_gap, enc_feat_logits, dec_feat, dec_logits_list = model(img)

            dec_feat = dec_feat.permute(1, 0, 2)
            dec_feat = dec_feat.reshape(dec_feat.size(0), -1)

            features = torch.cat((enc_feat_gap, dec_feat), dim=1)
            features = features.cpu()

            test_features.append(features)
            test_labels.append(person_labels_original)

        for i, data in enumerate(test_loader_query):
            img, person_labels, person_labels_original = data

            # forward
            img = img.to(device)

            enc_feat_gap, enc_feat_logits, dec_feat, dec_logits_list = model(img)

            dec_feat = dec_feat.permute(1, 0, 2)
            dec_feat = dec_feat.reshape(dec_feat.size(0), -1)

            features = torch.cat((enc_feat_gap, dec_feat), dim=1)
            features = features.cpu()

            query_features.append(features)
            query_labels.append(person_labels_original)

        test_features = torch.cat(test_features, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        query_features = torch.cat(query_features, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        distance_matrix = torch.cdist(query_features, test_features, p=2)
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

            if q_label in test_labels[rank1[i]]:
                rank1_correct += 1

            if q_label in test_labels[rank3[i]]:
                rank3_correct += 1

            if q_label in test_labels[rank5[i]]:
                rank5_correct += 1

            total += 1

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
    # main_test('market', 'market-220919043253.pt')
    # main_test('duke', 'duke-220919064246.pt')
    # main_test('duke-occ', 'duke-occ-220919015509.pt')
    # main_test('occ-reid', 'market-occ-220918234142.pt')
    main_test('part-reid', 'market-occ-220918234142.pt')
    main_test('part-ilids', 'market-occ-220918234142.pt')

    
