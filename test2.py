from going_modular.model.get_model import ConcatMTLFaceRecognitionV3

CONFIGURATION = {
    'type': 'concat3',
    'use_quant': False,
    'freeze_options': 'layer3',
    # Thư mục
    'dataset_dir': '/kaggle/input/hoangvn-3dmultitask',
    'checkpoint_dir': None,
    'num_classes': 262,
    'epochs': 5,
    'num_workers': 4,
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


model = ConcatMTLFaceRecognitionV3(
    config =CONFIGURATION,
    load_checkpoint = False,
    backbone_quant_mode = "no"
)

