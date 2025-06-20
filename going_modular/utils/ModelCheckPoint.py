import torch
import os
import copy
# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

BACKEND = "x86"

class ModelCheckpoint:
    def __init__(
            self, 
            filepath,
            test_dataloader: torch.utils.data.DataLoader,
            verbose=0,
        ):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.verbose = verbose
        self.test_dataloader = test_dataloader

    def __call__(self, model, optimizer, epoch, use_quant:bool):
        if use_quant:
            copied_model = copy.deepcopy(model).cpu()
            torch.backends.quantized.engine = BACKEND
            copied_model = torch.ao.quantization.prepare(copied_model)
            copied_model.eval()
            with torch.no_grad():
                for X, _ in self.test_dataloader:
                    X = X.cpu()
                    copied_model(X)

            save_model = torch.ao.quantization.convert(copied_model)
        else:
            save_model = copy.deepcopy(model)

        os.makedirs(os.path.join(os.path.dirname(self.filepath),str(epoch)), exist_ok=True)
        torch.save(save_model.state_dict(), self.filepath.replace('checkpoint.pth',f'{epoch}/checkpoint.pth'))
        if self.verbose > 0:
            print(f"\tSaving model and optimizer state to {self.filepath}")

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
                print(f"\tLoaded model and optimizer state from {self.filepath}")
                print(f"\tResuming from epoch {epoch}")
        else:
            if self.verbose > 0:
                print(f"\tNo checkpoint found at {self.filepath}. Starting from scratch.", 'red')
