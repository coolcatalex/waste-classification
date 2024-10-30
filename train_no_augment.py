"""
################ How to run this code ################'

python train_no_augment.py --model resnet18 --model_save_dir resnet18_no_aug --use_pretrained True --train_data dataset/train_resized
--test_data dataset/test_resized --max_epoch 100

python train_no_augment.py --model alexnet --model_save_dir log/alexnet_no_aug_no_pretrained --use_pretrained False
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
import time

# Define transformations for the training and validation sets
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )
        print('Model Dir Created in ', path)
    else:
        print('Model Dir Exist in ', path)

# def get_model(model_name, num_classes):
#     """Function to return the specified model."""
#     model = getattr(models, model_name)(pretrained=True)
#     if model_name.startswith('resnet') or model_name.startswith('vgg') or model_name.startswith('alexnet') or model_name.startswith('densenet'):
#         num_ftrs = model.classifier[-1].in_features if model_name.startswith('vgg') else model.fc.in_features
#         model.classifier[-1] = nn.Linear(num_ftrs, num_classes) if model_name.startswith('vgg') else nn.Linear(num_ftrs, num_classes)
#         if model_name.startswith('resnet') or model_name.startswith('densenet') or model_name.startswith('alexnet'):
#             num_ftrs = model.fc.in_features
#             model.fc = nn.Linear(num_ftrs, num_classes)
#         else:
#             num_ftrs = model.classifier[-1].in_features
#             model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
#     elif model_name.startswith('mobilenet') or model_name.startswith('shufflenet'):
#         num_ftrs = model.classifier[-1].in_features
#         model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
#     elif model_name.startswith('efficientnet'):
#         num_ftrs = model.classifier[-1].in_features
#         model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
#     elif model_name.startswith('googlenet'):
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, num_classes)
#     elif model_name.startswith('mnasnet'):
#         num_ftrs = model.classifier[-1].in_features
#         model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
#     elif model_name.startswith('squeezenet'):
#         num_classes = model.num_classes
#         model.classifier = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
#         model.num_classes = num_classes
#     else:
#         raise ValueError(f"Model {model_name} is not supported.")
#     return model

def get_model(model_name, num_classes, pretrained):
    """Function to return the specified model."""
    print('Load pretrained', pretrained)
    model = getattr(models, model_name)(pretrained=pretrained)

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'fc'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Conv2d):
        model.classifier = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError(f"Model {model_name} is not supported or needs custom handling.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TrashNet')
    parser.add_argument('--model', type=str, required=True, default='mobilenet_experiment',
                        help='Name of the model to use (e.g., mobilenet_v2, resnet18, alexnet).')
    parser.add_argument('--model_save_dir', type=str, default='mobilenet_experiment', help='Experiment save directory')
    parser.add_argument('--use_pretrained', type=bool, default=False, help='Use imagenet pretrained weights')
    parser.add_argument('--train_data', type=str, default='/home/pankaj/IC_models/trashNet/dataset/train_resized', help='Path to the training data.')
    parser.add_argument('--test_data', type=str, default='/home/pankaj/IC_models/trashNet/dataset/test_resized', help='Path to the validation data.')
    parser.add_argument('--max_epochs', type=int, default=100)



    args = parser.parse_args()
    print(args)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(root=args.train_data, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=args.test_data, transform=transform_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load a pre-trained MobileNet model and modify the classifier
    # model = models.mobilenet_v2(pretrained=args.use_pretrained)
    model = get_model(args.model, len(train_dataset.classes), args.use_pretrained)
    # num_ftrs = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_ftrs, len(train_dataset.classes))

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize variables to track the best model
    best_accuracy = 0.0
    best_model_wts = model.state_dict()

    ########################### Create Model Dir ##############################
    makedirs(args.model_save_dir)
    if os.path.exists(os.path.join(args.model_save_dir, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'model.pth')))
        print('Existing Model weights Loaded !!!!')
    else:
        print('No Existing Model weights !!!!')

    if os.path.exists(os.path.join(args.model_save_dir, 'optimizer.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'optimizer.pth')))
        print('Existing Optimizer state Loaded !!!!')
    else:
        print('No Existing Optimizer state weights !!!!')

    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(args.model_save_dir, 'tb_data'))

    if os.path.exists(os.path.join(args.model_save_dir, 'infos.json')):
        info = json.load(open(os.path.join(args.model_save_dir, 'infos.json'), 'r'))
        print('Info file Loaded!!!!')
        init_epoch = info['init_epoch']
    # Training the model
    else:
        init_epoch = 0


    start_time=time.time()
    for epoch in range(init_epoch, args.max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{args.max_epochs}, Loss: {avg_train_loss}')

        # Evaluate on the validation set
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        print('Validation in progress ... ')
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Validation Loss: {avg_val_loss}')
        print(f'Validation Accuracy: {val_accuracy}%')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)


        info={'init_epoch': epoch,
              'train_loss':avg_train_loss,
              'val_loss':avg_val_loss,
              'acc':val_accuracy,
              'prec':precision,
              'recall':recall,
              'f1':f1
              }
        # Save the model if it has the best accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(args.model_save_dir, 'model-best.pth') )
            torch.save(optimizer.state_dict(), os.path.join(args.model_save_dir, 'optimizer-best.pth'))
            json.dump(info, open(os.path.join(args.model_save_dir, 'infos-best.json'), 'w'))
            print('Best model saved with accuracy:', best_accuracy)

        torch.save(best_model_wts, os.path.join(args.model_save_dir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.model_save_dir, 'optimizer.pth'))
        json.dump(info, open(os.path.join(args.model_save_dir, 'infos.json'), 'w'))
        print('Model saved with accuracy:', best_accuracy)

    ex_time=time.time()-start_time

    print('Training complete')
    print('Total training time ', time.strftime("%H:%M:%S", time.gmtime(ex_time)))
    writer.close()
