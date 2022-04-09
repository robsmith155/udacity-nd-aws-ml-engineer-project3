import io
import json
import logging
import os
import numpy as np
from PIL import Image
import requests
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def net():
    '''
    Instantiates pre-trained ResNet18 model and adds classification layer to end of model.
    '''
    model = models.resnet18(pretrained=False)
    
    num_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_features, 1))

    return model


def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def input_fn(request_body, content_type='image/jpeg'):
    logger.info('Deserializing input data')
    logger.info(f'Request content type: {type(request_body)}')
    if content_type == 'image/jpeg':
        logger.info('Loading image')
        return Image.open(io.BytesIO(request_body))
    elif content_type == 'application/json':
        img_request = requests.get(request_body['url'], stream=True)
        return Image.open(io.BytesIO(img_request.content))
    raise Exception(f'Unsupported content type ({type(request_body)}). Expected image/jpeg')


def predict_fn(input_object, model):
    logger.info('Starting predict function')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    # Transform data
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

    logger.info('Passing data through image transformations')
    input_object = test_transforms(input_object)
    logger.info(f'Input mean: {np.mean(input_object.numpy().flatten())}')
    logger.info(f'Input std: {np.std(input_object.numpy().flatten())}')

    with torch.no_grad():
        logger.info(f'Making prediction on input object')
        logger.info(f'Input object type is {type(input_object)}')
        logger.info(f'Input object size is {input_object.shape}')
        prediction = model(input_object.unsqueeze(0).to(device))
        logger.info(f'Output from PyTorch model: {prediction}')
    
    return prediction


# # Serialize the prediction result into the desired response content type
# def output_fn(prediction, accept='application/json'):        
#     logger.info('Serializing the generated output.')
#     if accept == 'application/json': return json.dumps(prediction), accept
#     raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))   