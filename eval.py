import torch
import albumentations as A
from going_modular.utils.transforms import RandomResizedCropRect, GaussianNoise
from going_modular.dataloader.multitask import create_concatv3_multitask_datafetcher
from going_modular.utils.roc_auc import compute_auc
import os
from typing import Literal

MODEL_USE: Literal['quant','non_quant'] = "quant"

quant_model_path = "checkpoint/concat3/2025-05-28_18-12-42/models/best_cosine_auc_48.pth"
non_quant_model_path = "checkpoint/concat3/2025-05-28_19-44-01/models/best_cosine_auc_21.pth"

device = torch.device("cpu")
CONFIGURATION = {
    'dataset_dir': 'dataset',
    'device': device,
    'num_workers': 4,
    'batch_size': 64,
    'image_size': 256
}

CONFIGURATION['num_classes_test'] = len(os.listdir(f'{CONFIGURATION['dataset_dir']}/Albedo/test'))

train_transform = A.Compose([
    RandomResizedCropRect(256),
    GaussianNoise(),
], additional_targets={
    'albedo': 'image',
    'depthmap': 'image'
})


test_transform = A.Compose([
    A.Resize(height=CONFIGURATION['image_size'], width=CONFIGURATION['image_size'])
],
    additional_targets={
    'albedo': 'image',
    'depthmap': 'image'
})

(
    _ , 
    test_dataloader
) = create_concatv3_multitask_datafetcher(
    CONFIGURATION, 
    train_transform, 
    test_transform
)

model = torch.load(
    quant_model_path if MODEL_USE == "quant" else non_quant_model_path, 
    map_location= "cpu", 
    weights_only= False
)

auc_scores = compute_auc(
    dataloader= test_dataloader, 
    model= model, 
    device= device, 
    num_classes= CONFIGURATION['num_classes_test']
)

