import gc
import torch

CPU_DEVICE = torch.device("cpu")
ACC_DEVICE = torch.device("cuda") if torch.cuda.is_available() else CPU_DEVICE
def do_gc():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        pass

def migrate_to_device( t, device ):
    if t is None:
        return t

    if not hasattr(t, 'device'):
        t.device = device
        t.manually_track_device = True
    elif t.device == device:
        return t

    if hasattr(t, 'manually_track_device') and t.manually_track_device:
        t.device = device

    t = t.to(device)
    
    do_gc()
    
    return t



