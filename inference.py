"""
python inference.py --model mobilenet_v3_large
--model_path log/mobilenet_v3_large_no_aug_no_pretrained/model-best.pth --image_path test3_can.jpg

"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import argparse


def get_model(model_name, num_classes):
    """Function to return the specified model."""
    model = getattr(models, model_name)(pretrained=True)

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


def load_image(image_path, transform=None):
    """Load an image and apply transformations."""
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image


def predict(image_path, model, transform, class_names, device):
    """Run inference on an image and return the predicted class."""
    model.eval()
    image = load_image(image_path, transform)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move the input to the same device as the model

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze()
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        # print(probabilities.shape, probabilities[predicted.item()], predicted.item())
    return predicted_class, probabilities[predicted.item()]


def get_class_names(data_path):
    """Get class names from the subdirectories of the given path."""
    class_names = [d.name for d in os.scandir(data_path) if d.is_dir()]
    class_names.sort()
    return class_names


def main(args):
    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get class names from the training data path
    class_names = get_class_names(args.data_path)
    # print(class_names)
    # Load the specified model
    model = get_model(args.model, len(class_names))
    model.load_state_dict(torch.load(args.model_path))

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Perform inference
    predicted_class, predicted_probability = predict(args.image_path, model, transform, class_names, device)
    print(f'Predicted class: {predicted_class}')
    print(f'Prediction probability: {predicted_probability*100:0.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on an image using a trained CNN.')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the model to use (e.g., mobilenet_v2, resnet18, alexnet).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--data_path', type=str, default='/home/pankaj/IC_models/trashNet/dataset/test_resized',
                        help='Path to the data folder containing class subfolders.')

    args = parser.parse_args()
    main(args)
