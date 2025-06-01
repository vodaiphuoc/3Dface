import torch
import albumentations as A
from going_modular.model.get_model import ConcatMTLFaceRecognitionV3
from going_modular.utils.transforms import RandomResizedCropRect, GaussianNoise
from going_modular.dataloader.multitask import create_concatv3_multitask_datafetcher
from going_modular.utils.roc_auc import compute_auc
import os
from typing import Literal
import psutil

MODEL_USE: Literal['quant','non_quant'] = "quant"

quant_ckpt_path = "checkpoint/concat3/2025-05-28_18-12-42/models/best_cosine_auc_48.pth"
non_quant_ckpt_path = "checkpoint/concat3/2025-05-28_19-44-01/models/best_cosine_auc_21.pth"

device = torch.device("cpu")


CONFIGURATION = {
    'type': 'concat3',
    'use_quant': True if MODEL_USE == "quant" else False,
    'freeze_options': 'layer4',
    # Thư mục
    'dataset_dir': 'dataset',
    'checkpoint_dir': None,
    'num_classes': 262,
    'epochs': 5,
    'num_workers': 1,
    'batch_size': 64,
    'image_size': 256,
    'base_lr': 1e-4,
    
    # Cấu hình network
    'backbone': 'miresnet18',
    'embedding_size': 512,
    'loss_id_weight': 0.01,
    'loss_gender_weight': 20,
    'loss_emotion_weight': 5,
    'loss_pose_weight': 20,
    'loss_spectacles_weight': 10,
    'loss_facial_hair_weight': 10,
}


CONFIGURATION['num_classes_test'] = len(os.listdir(f'{CONFIGURATION['dataset_dir']}/Albedo/test'))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    # in bytes, convert to MB
    return process.memory_info().rss / (1024 * 1024)

# Before model loading/inference
_before = get_memory_usage()
print(f"Memory before: {_before:.2f} MB")


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
    test_dataloader,
    gallery_dataloader
) = create_concatv3_multitask_datafetcher(
    CONFIGURATION, 
    train_transform, 
    test_transform
)


state_dict = torch.load(
    quant_ckpt_path if MODEL_USE == "quant" else non_quant_ckpt_path, 
    map_location= "cpu", 
    weights_only= True
)

if CONFIGURATION['use_quant']:
    model = ConcatMTLFaceRecognitionV3(
        config = CONFIGURATION,
        load_checkpoint = False,
        backbone_quant_mode="ptq"
    )

    torch.backends.quantized.engine = 'x86'
    model.train()
    model = torch.ao.quantization.prepare(model)
    model = torch.ao.quantization.convert(model)
else:
    model = ConcatMTLFaceRecognitionV3(
        config = CONFIGURATION,
        load_checkpoint = False,
        backbone_quant_mode="no"
    )

model.load_state_dict(state_dict)
print(type(model.mtl_normalmap.backbone))
model.eval()





auc_scores = compute_auc(
    dataloader= test_dataloader, 
    model= model, 
    device= device, 
    num_classes= CONFIGURATION['num_classes_test']
)

