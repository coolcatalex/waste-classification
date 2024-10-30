import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse


def get_model(model_name, num_classes):
    """Function to return the specified model."""
    model = getattr(models, model_name)(pretrained=False)

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


def evaluate(args):
    # Define transformations for the test set
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=args.test_data_path, transform=transform_test)
    print(test_dataset.classes)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the specified model
    model = get_model(args.model, len(test_dataset.classes))

    # Load the best model weights
    model.load_state_dict(torch.load(args.best_model_path))

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Evaluate on the test set
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {test_accuracy}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print("{:0.2f},{:0.2f},{:0.2f},{:0.2f}".format(test_accuracy, precision*100, recall*100, f1*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN on the test dataset.')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the model to use (e.g., mobilenet_v2, resnet18, alexnet).')
    parser.add_argument('--test_data_path', type=str, default='/home/pankaj/IC_models/trashNet/dataset/test_resized',
                        help='Path to the test data.')
    parser.add_argument('--best_model_path', type=str, required=True, help='Path to the best model weights.')

    args = parser.parse_args()
    evaluate(args)
