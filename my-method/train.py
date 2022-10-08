import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import train
from data import CustomDataset, BatchSampler, get_transform_1, get_transform_2
from torch.utils.data import DataLoader
from models import MyModel
from triplet import batch_hard_mine_triplet, calculate_distance_matrix
import numpy as np
import sys
from datetime import datetime


def main(dataset_name):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_epochs = 80
    n_persons = 16
    n_pictures = 4

    if dataset_name == 'market_occ':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        num_stripes = 6
        lr = 0.02
        alpha = 0.9
        transform_fn = get_transform_2
    elif dataset_name == 'market':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
        extensions = ['.jpg']
        num_stripes = 6
        lr = 0.02
        alpha = 0.9
        transform_fn = get_transform_1
    elif dataset_name == 'duke':
        train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_train'
        extensions = ['.jpg']
        num_stripes = 4
        lr = 0.05
        alpha = 0.8
        transform_fn = get_transform_1

    dataset = CustomDataset(train_path, extensions, num_stripes, transform_fn=transform_fn, training=True)
    batch_sampler = BatchSampler(dataset, n_persons, n_pictures)
    train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    num_classes = dataset.get_num_classes()

    model = MyModel(num_classes, num_stripes=num_stripes)
    model = model.to(device)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

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
            img, person_labels, person_labels_original, occlusion_labels = data

            # move to GPU
            img = img.to(device)
            person_labels = person_labels.to(device)
            occlusion_labels = occlusion_labels.to(device)

            global_feat, global_logits, local_feat_list, local_logits_list, rvd_logits_list = model(img)

            gid_loss = ce(global_logits, person_labels)

            distance_matrix = calculate_distance_matrix(occlusion_labels, 
                                                        occlusion_labels, 
                                                        local_feat_list, 
                                                        local_feat_list, 
                                                        global_feat, 
                                                        global_feat)


            gtri_loss = batch_hard_mine_triplet(distance_matrix, person_labels)

            pid_loss = 0
            ptri_loss = 0
            rvd_loss = 0

            for stripe in range(num_stripes):
                pid_loss += ce(local_logits_list[stripe], person_labels)
                ptri_loss += batch_hard_mine_triplet(torch.cdist(local_feat_list[stripe], local_feat_list[stripe], p=2), person_labels)

                rvd_loss += bce(rvd_logits_list[stripe], occlusion_labels[:, stripe].unsqueeze(1).float())

            pid_loss /= num_stripes
            ptri_loss /= num_stripes
            rvd_loss /= num_stripes

            loss = ((1 - alpha) * gid_loss) + (alpha * pid_loss) + rvd_loss + ptri_loss + gtri_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            sys.stdout.write("\r" + '........ mini-batch {} loss: {:.3f}'.format(batch_idx + 1, loss.item()))
            sys.stdout.flush()
        
        # scheduler.step()
        
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
        test_pose_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15-pose\\bounding_box_test'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\query'
        query_pose_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15-pose\\query'
        extensions = ['.jpg']
        num_stripes = 6
        num_classes = 751
        transform_fn = get_transform_1
    elif dataset_name == 'duke-occ':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\bounding_box_test'
        test_pose_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID-pose\\bounding_box_test'
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID\\query'
        query_pose_path = 'C:\\Users\\leona\\Documents\\Dataset\\Occluded-DukeMTMC-reID-pose\\query'
        extensions = ['.jpg']
        num_stripes = 4
        num_classes = 702
        transform_fn = get_transform_1
    elif dataset_name == 'occ-reid':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\OccludedREID\\gallery'
        test_pose_path = None
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\OccludedREID\\query'
        query_pose_path = None
        extensions = ['.jpg']
        num_stripes = 6
        num_classes = 751
        transform_fn = get_transform_2
    elif dataset_name == 'part-reid':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\Partial_REID\\whole_body_images'
        test_pose_path = None
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\Partial_REID\\partial_body_images'
        query_pose_path = None
        extensions = ['.jpg']
        num_stripes = 6
        num_classes = 751
        transform_fn = get_transform_2
    elif dataset_name == 'part-ilids':
        test_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\PartialiLIDS\\gallery'
        test_pose_path = None
        query_path = 'C:\\Users\\leona\\Documents\\Dataset\\partial_dataset\\PartialiLIDS\\query'
        query_pose_path = None
        extensions = ['.jpg']
        num_stripes = 6
        num_classes = 751
        transform_fn = get_transform_2

    test_dataset = CustomDataset(test_path, test_pose_path, extensions, num_stripes, transform_fn=transform_fn, training=False)
    query_dataset = CustomDataset(query_path, query_pose_path, extensions, num_stripes, transform_fn=transform_fn, training=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_query = DataLoader(query_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = MyModel(num_classes, num_stripes=num_stripes)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    print('Starting Test')
    print(dataset_name)

    test_local_feat_list = []
    test_global_feat = []
    test_occlusion_labels = []
    test_labels = []

    query_local_feat_list = []
    query_global_feat = []
    query_occlusion_labels = []
    query_labels = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(test_loader):
            img, person_labels, person_labels_original, occlusion_labels = data

            # forward
            img = img.to(device)

            global_feat, global_logits, local_feat_list, local_logits_list, rvd_logits_list = model(img)

            local_feat_list = torch.stack(local_feat_list, dim=0).cpu()
            global_feat = global_feat.cpu()

            rvd_logits_list = torch.stack([(torch.sigmoid(rvdl.view(-1)) > 0.5).int() for rvdl in rvd_logits_list], dim=1)
            rvd_logits_list = rvd_logits_list.cpu()

            person_labels_original = person_labels_original.cpu()

            test_local_feat_list.append(local_feat_list)
            test_global_feat.append(global_feat)
            test_occlusion_labels.append(rvd_logits_list)
            test_labels.append(person_labels_original)

        for i, data in enumerate(test_loader_query):
            img, person_labels, person_labels_original, occlusion_labels = data

            # forward
            img = img.to(device)

            global_feat, global_logits, local_feat_list, local_logits_list, rvd_logits_list = model(img)

            local_feat_list = torch.stack(local_feat_list, dim=0).cpu()
            global_feat = global_feat.cpu()

            rvd_logits_list = torch.stack([(torch.sigmoid(rvdl.view(-1)) > 0.5).int() for rvdl in rvd_logits_list], dim=1)
            rvd_logits_list = rvd_logits_list.cpu()

            person_labels_original = person_labels_original.cpu()

            query_local_feat_list.append(local_feat_list)
            query_global_feat.append(global_feat)
            query_occlusion_labels.append(rvd_logits_list)
            query_labels.append(person_labels_original)


        test_local_feat_list = torch.cat(test_local_feat_list, dim=1)
        test_global_feat = torch.cat(test_global_feat, dim=0)
        test_occlusion_labels = torch.cat(test_occlusion_labels, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        query_local_feat_list = torch.cat(query_local_feat_list, dim=1)
        query_global_feat = torch.cat(query_global_feat, dim=0)
        query_occlusion_labels = torch.cat(query_occlusion_labels, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        distance_matrix = calculate_distance_matrix(query_occlusion_labels, 
                                                    test_occlusion_labels, 
                                                    query_local_feat_list, 
                                                    test_local_feat_list, 
                                                    query_global_feat, 
                                                    test_global_feat)

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
    main('market_occ')
    main('market')
    main('duke')

    # main_test('market', 'checkpoint_adam_bck.pt')
    # main_test('duke-occ', 'checkpoint_adam_duke_bck.pt')
    # main_test('occ-reid', 'market-221006035932.pt')
    # main_test('part-reid', 'market-221006035932.pt')
    # main_test('part-ilids', 'market-221006035932.pt')

