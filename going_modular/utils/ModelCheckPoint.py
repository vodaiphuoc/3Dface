import torch
from termcolor import cprint
import os
import copy
# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

from torchao.quantization import (
    quantize_,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig
)
from torchao.quantization.qat import (
    FromIntXQuantizationAwareTrainingConfig,
)

class ModelCheckpoint:
    def __init__(self, filepath, verbose=0):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.verbose = verbose

    def __call__(self, model, optimizer, epoch, use_quant:bool):
        if use_quant:
            copied_model = copy.deepcopy(model)
            quantize_(copied_model, FromIntXQuantizationAwareTrainingConfig())
            quantize_(copied_model, Int8DynamicActivationInt4WeightConfig(group_size=32))
            save_model = copied_model
        else:
            save_model = copy.deepcopy(model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        os.makedirs(os.path.join(os.path.dirname(self.filepath),str(epoch)), exist_ok=True)
        torch.save(checkpoint, self.filepath.replace('checkpoint.pth',f'{epoch}/checkpoint.pth'))
        if self.verbose > 0:
            cprint(f"\tSaving model and optimizer state to {self.filepath}", 'cyan')

    def load_checkpoint(self, model, optimizer, scheduler):
        """
        Load checkpoint vào model, optimizer và scheduler từ file.
        """
        if os.path.isfile(self.filepath):
            checkpoint = torch.load(self.filepath)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            epoch = checkpoint['epoch']

            if self.verbose > 0:
                cprint(f"\tLoaded model and optimizer state from {self.filepath}", 'cyan')
                cprint(f"\tResuming from epoch {epoch}", 'cyan')
        else:
            if self.verbose > 0:
                cprint(f"\tNo checkpoint found at {self.filepath}. Starting from scratch.", 'red')
