import torch
from going_modular.model.ConcatMTLFaceRecognition import ConcatMTLFaceRecognitionV3
from fvcore.nn import FlopCountAnalysis

def run_count_flops(
    dataloader: torch.utils.data.DataLoader, 
    model: ConcatMTLFaceRecognitionV3, 
    device: str
):
    mean_flops = 0
    step = 0
    with torch.no_grad():
        for batch in dataloader:
            images, y = batch
            images = images.to(device)
            
            flops = FlopCountAnalysis(model, images)
            mean_flops += flops.total()
            step += 1
    mean_flops = mean_flops/step
    print("mean flops: ", mean_flops)





def get_flops(
        dataloader: torch.utils.data.DataLoader, 
        model: ConcatMTLFaceRecognitionV3, 
        device: str
    ):
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        for batch in dataloader:
            images, y = batch
            images = images.to(device)
            model(images)
    total_flops =  flop_counter.get_total_flops()
    
    return total_flops