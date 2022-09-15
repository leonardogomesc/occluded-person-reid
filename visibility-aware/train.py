import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import train
from data import CustomDataset
from torch.utils.data import DataLoader
from models import MyModel


def test(model, test_loader, test_loader_query):
    test_features = []
    test_labels = []

    query_features = []
    query_labels = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(test_loader):
            images, labels = data

            # forward
            images = images.to(device)

            features = model(images)
            features = features.cpu()

            test_features.append(features)
            test_labels.append(labels)

        for i, data in enumerate(test_loader_query):
            images, labels = data

            # forward
            images = images.to(device)

            features = model(images)
            features = features.cpu()

            query_features.append(features)
            query_labels.append(labels)

        test_features = torch.cat(test_features, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        query_features = torch.cat(query_features, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        distance_matrix = torch.cdist(query_features, test_features, p=2)
        sorted_matrix = torch.argsort(distance_matrix, dim=1)

        rank1 = sorted_matrix[:, :1]
        rank1_correct = 0

        rank5 = sorted_matrix[:, :5]
        rank5_correct = 0

        rank10 = sorted_matrix[:, :10]
        rank10_correct = 0

        total = 0

        for i in range(len(query_labels)):
            q_label = query_labels[i]

            if q_label in test_labels[rank1[i]]:
                rank1_correct += 1

            if q_label in test_labels[rank5[i]]:
                rank5_correct += 1

            if q_label in test_labels[rank10[i]]:
                rank10_correct += 1

            total += 1

        print('rank1 acc: ' + str(rank1_correct / total))
        print('rank5 acc: ' + str(rank5_correct / total))
        print('rank10 acc: ' + str(rank10_correct / total))

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


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_epochs = 10
    num_classes = 51

    model = MyModel(num_classes)
    model = model.to(device)

    ce = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    save_path = 'model.pth'
    
    batch_size = 64

    # Create Dataloader to read the data within batch sizes and put into memory. 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    print('Starting Training')

    valid_loss_min = float('inf')
    train_loss_hist = []
    valid_loss_hist = []

    for epoch in range(1, n_epochs + 1):
        t_start = time.time()

        ######################
        # training the model #
        ######################

        train_loss = 0.0

        train_predicted = []
        train_true = []

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            data = data.to(device)
            target = target.to(device)
          
            output = model(data)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            train_loss += loss.item()

            train_predicted.extend(torch.argmax(output, 1).tolist())
            train_true.extend(target.tolist())

            sys.stdout.write("\r" + '........ mini-batch {} loss: {:.3f}'.format(batch_idx + 1, loss.item()))
            sys.stdout.flush()
        
        train_loss /= batch_idx + 1

        train_acc = np.mean(calculate_accuracy(train_true, train_predicted, n_classes=n_classes)) * 100
    
        ########################    
        # validating the model #
        ########################

        valid_loss = 0.0

        valid_predicted = []
        valid_true = []

        model.eval()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(valid_loader):
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                loss = loss_fn(output, target)

                valid_loss += loss.item()

                valid_predicted.extend(torch.argmax(output, 1).tolist())
                valid_true.extend(target.tolist())
        
        valid_loss /= batch_idx + 1

        valid_acc = np.mean(calculate_accuracy(valid_true, valid_predicted, n_classes=n_classes)) * 100

        train_loss_hist.append(train_loss)
        valid_loss_hist.append(valid_loss)

        t_end = time.time()

        # printing training/validation statistics 
        print('\n')
        print(f'Epoch: {epoch}')
        print(f'\tTraining Loss: {train_loss}')
        print(f'\tTraining Accuracy: {train_acc}')
        print(f'\tValidation Loss: {valid_loss}')
        print(f'\tValidation Accuracy: {valid_acc}')
        print(f'Total time: {t_end - t_start} s')
        
        ## saving the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print('Saving model')
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
        
        print('\n\n')
            
    # return trained model
    return model, train_loss_hist, valid_loss_hist


if __name__ == '__main__':
    main()

