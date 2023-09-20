import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

def post_process_pipeline(cfg, val_data, val_df):
    
    # val_data['generated_ids'].shape => torch.Size([64, 33])
    generated_ids = val_data['generated_ids'].cpu()
    seq_len = val_data['seq_len'].cpu()
    conf = val_data['aux_logits'].sigmoid().cpu().numpy()[:,0]

    # [s for s in generated_id] => [60, 32, 45, 36, 0, 50, 51, 39, 36, 49, 36, 49, 36, 49, 36, 49, 36, 49, 36, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59]
    # "".join([cfg.rev_character_map.get(s, "") for s in generated_id]) => ane sthererererere

    
    phrase_preds = np.array(["".join([cfg.rev_character_map.get(s, "") for s in generated_id]) for generated_id in generated_ids.numpy()])
    
    # phrase_preds.shape => (64,)
        
    dummy_phrase = "".join([cfg.rev_character_map.get(s, "") for s in cfg.dummy_phrase_ids])
    
    phrase_preds_pp = phrase_preds.copy()
    mask = (conf<cfg.pp_min_conf) | (seq_len.numpy() < cfg.max_len_for_dummy)
    print(mask.mean())
    phrase_preds_pp[mask] = dummy_phrase
    
    return {'phrase_preds':phrase_preds,'phrase_preds_pp':phrase_preds_pp}
