import sys
import time
import torch
import numpy as np


def calculate_accuracy(gt, pre, n_classes=2):

    confusion_matrix = np.zeros((n_classes, n_classes))
    
    for t, p in zip(gt, pre):
        confusion_matrix[t, p] += 1
    
    return np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=1)


def train(n_epochs, model, loss_fn, optimizer, device, save_path, train_loader, valid_loader, n_classes):
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

