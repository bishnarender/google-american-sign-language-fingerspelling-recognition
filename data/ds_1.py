import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch.nn as nn
import torch.nn.functional as F
import math

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

tr_collate_fn = None
val_collate_fn = None

class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def normalize(self,x):
        nonan = x[~torch.isnan(x)].view(-1, x.shape[-1])
        x = x - nonan.mean(0)[None, None, :]
        
        # unbiased is renamed as correction in latest versions.
        x = x / nonan.std(0, unbiased=False)[None, None, :]
        return x
    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x
        
    def forward(self, x):
        
        #seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3
        x = x.reshape(x.shape[0],3,-1).permute(0,2,1)
        # x.shape => torch.Size([216, 130, 3])
                
        x = self.normalize(x)
        # x.shape => torch.Size([216, 130, 3])
        
        x = self.fill_nans(x)
        # x.shape => torch.Size([216, 130, 3])

        return x


# augs
def flip(data, flip_array):
    # data.shape, data[:,:,0].shape => torch.Size([216, 130, 3]), torch.Size([216, 130])
    data[:,:,0] = -data[:,:,0]
    data = data[:,flip_array]
    # data.shape => torch.Size([216, 130, 3])
    return data

def interpolate_or_pad(data, max_len=100, mode="start"):
    # max_len => 
    # data.shape => torch.Size([172, 130, 3])
    diff = max_len - data.shape[0]
    
    # data[:,0,0].shape => torch.Size([172])
    if diff <= 0:  # crop
        data = F.interpolate(data.permute(1,2,0),max_len).permute(2,0,1)
        mask = torch.ones_like(data[:,0,0])
        return data, mask
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2]))
    mask = torch.ones_like(data[:,0,0])
    data = torch.cat([data, padding * coef])
    mask = torch.cat([mask, padding[:,0,0] * coef])
    # mask.shape => torch.Size([384])

    return data, mask


def outer_cutmix(data, phrase, data2, phrase2):
    cut_off = np.random.rand()
    # cut_off => 0.8819831062049998
    # len(phrase) => 13 
    # data.shape => torch.Size([156, 130, 3])
    
    cut_off_phrase = np.clip(round(len(phrase) * cut_off), 1, len(phrase)-1)
    cut_off_phrase2 = np.clip(round(len(phrase2) * cut_off), 1, len(phrase2)-1)
    
    # cut_off_phrase, cut_off_phrase2 => 11, 16
    
    cut_off_data = np.clip(round(data.shape[0] * cut_off), 1, data.shape[0]-1)
    cut_off_data2 = np.clip(round(data2.shape[0] * cut_off), 1, data2.shape[0]-1)
    
    # cut_off_data, cut_off_data2 => 83, 46
          
    if np.random.rand() < 0.5: # 0 my_
        new_phrase = phrase2[cut_off_phrase2:] + phrase[:cut_off_phrase]
        new_data = torch.cat([data2[cut_off_data2:], data[:cut_off_data]])
    else:
        new_phrase = phrase[cut_off_phrase:] + phrase2[:cut_off_phrase2]
        # data.shape => torch.Size([156, 130, 3])
        new_data = torch.cat([data[cut_off_data:], data2[:cut_off_data2]])            
        # new_data.shape => torch.Size([154, 130, 3])    

    return new_data, new_phrase



class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.aug = aug
        # cfg.min_seq_len => 15
        
        if mode =='train':
            to_drop = self.df['seq_len'] < cfg.min_seq_len
            self.df = self.df[~to_drop].copy()
            print(f'new shape {self.df.shape[0]}, dropped {to_drop.sum()} sequences shorter than min_seq_len {cfg.min_seq_len}')                
        
        with open(cfg.data_folder + 'inference_args.json', "r") as f:
            columns = json.load(f)['selected_columns']
        
        self.xyz_landmarks = np.array(columns)
        # len(self.xyz_landmarks) => 390
        # self.xyz_landmarks[:6] => ['x_face_0' 'x_face_61' 'x_face_185' 'x_face_40' 'x_face_39' 'x_face_37']
                
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[:len(self.xyz_landmarks)//3]])
        # landmarks[:5] => ['face_0' 'face_61' 'face_185' 'face_40' 'face_39']
        # landmarks[-5:] => ['pose_13' 'pose_15' 'pose_17' 'pose_19' 'pose_21']        
        
        
        # cfg.symmetry_fp => datamount/symmetry.csv
        symmetry = pd.read_csv(cfg.symmetry_fp).set_index('id')
        # symmetry.head(15) => 
        #         corresponding_id
        # id                      
        # face_0            face_0
        # face_1            face_1
        # face_2            face_2
        # face_3          face_248
        # face_4            face_4
        # face_5            face_5
        # face_6            face_6
        # face_7          face_249
        # face_8            face_8
        # face_9            face_9
        # face_10          face_10
        # face_11          face_11
        # face_12          face_12
        # face_13          face_13
        # face_14          face_14
               

        flipped_landmarks = symmetry.loc[landmarks]['corresponding_id'].values
        
        # landmarks[:,None][:11] => [['face_0'], ['face_61'], ['face_185'], ['face_40'], ['face_39'], ['face_37'], ['face_267'], ['face_269'], ['face_270'], ['face_409'], ['face_291']]
        
        # flipped_landmarks[None,:][:11] => [['face_0' 'face_291' 'face_409' 'face_270' 'face_269' 'face_267'  'face_37' 'face_39' 'face_40' 'face_185' 'face_61']]
        
        
        self.flip_array = np.where(landmarks[:,None]==flipped_landmarks[None,:])[1]
        # self.flip_array.shape => (130,)
        # self.flip_array[:13] => array([  0,  10,   9,   8,   7,   6,   5,   4,   3,   2,   1,  19,  18,])
        # self.flip_array => provides an index position where a landmark/keypoint is situated in flipped_landmarks order.
        
        self.max_len = cfg.max_len
        
        self.processor = Preprocessing()
        
        #target stuff
        self.max_phrase = cfg.max_phrase
        self.char_to_num, self.num_to_char, _ = self.cfg.tokenizer
        self.pad_token_id = self.char_to_num[self.cfg.pad_token]
        self.start_token_id = self.char_to_num[self.cfg.start_token]
        self.end_token_id = self.char_to_num[self.cfg.end_token]

        self.flip_aug = cfg.flip_aug
        self.outer_cutmix_aug = cfg.outer_cutmix_aug

        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        file_id, sequence_id, phrase = row[['file_id','sequence_id','phrase']]
        
        data = self.load_one(file_id, sequence_id)
        # data.shape => (210, 390)
        
        seq_len = data.shape[0]
        data = torch.from_numpy(data)
        data = self.processor(data)
        # data.shape => torch.Size([210, 130, 3])

        if self.mode == 'train':            
            
            # self.flip_aug => 0.5
            if np.random.rand() < self.flip_aug:
                data = flip(data, self.flip_array)  
            
            # self.outer_cutmix_aug => 0.5 
            if np.random.rand() < self.outer_cutmix_aug:#self.outer_cutmix_aug: # 1 my_
                participant_id = row['participant_id']
#                 phrase_type = row['phrase_type']

                sequence_id = row['sequence_id']                                
                
                mask = (self.df['participant_id']==participant_id) & (self.df['sequence_id']!=sequence_id)
                # mask => select other sequences of this participant.
                # mask.shape => (48764,)
                
                
                if mask.sum() > 0:
                
                    row2 = self.df[mask].sample(1).iloc[0]
                    file_id2, sequence_id2,phrase2 = row2[['file_id','sequence_id','phrase']]
                    data2 = self.load_one(file_id2, sequence_id2)
                    seq_len2 = data2.shape[0]
                    data2 = torch.from_numpy(data2)
                    data2 = self.processor(data2)                

                    data, phrase = outer_cutmix(data, phrase, data2, phrase2)
                
                
            if self.aug:
                data = self.augment(data)


        data, mask = interpolate_or_pad(data, max_len=self.max_len)
        
        token_ids, attention_mask = self.tokenize(phrase)
        
        feature_dict = {'input':data,
                        'input_mask':mask,
                        'token_ids':token_ids, 
                        'attention_mask':attention_mask,
                        'seq_len':torch.tensor(seq_len)}
        
        return feature_dict
    
    def augment(self,x):
        # type(self.aug) => <class 'albumentations.core.composition.Compose'>
        x_aug = self.aug(image=x)['image']
        # x_aug.shape => torch.Size([214, 130, 3])
        return x_aug
    
    def tokenize(self,phrase):
        phrase_ids = [self.char_to_num[char] for char in phrase]
        
        # self.max_phrase, len(phrase_ids), self.end_token_id => 33, 14, 61
        
        if len(phrase_ids) > self.max_phrase - 1:
            phrase_ids = phrase_ids[:self.max_phrase - 1]
            
        phrase_ids = phrase_ids + [self.end_token_id]
        attention_mask = [1] * len(phrase_ids)
        
        to_pad = self.max_phrase - len(phrase_ids)
        # self.pad_token_id => 59        
        phrase_ids = phrase_ids + [self.pad_token_id] * to_pad
        attention_mask = attention_mask + [0] * to_pad
        
        return torch.tensor(phrase_ids).long(), torch.tensor(attention_mask).long()
        
    
    def setup_tokenizer(self):
        with open (self.cfg.character_to_prediction_index_fn, "r") as f:
            char_to_num = json.load(f)

        n= len(char_to_num)
        char_to_num[self.cfg.pad_token] = n
        char_to_num[self.cfg.start_token] = n+1
        char_to_num[self.cfg.end_token] = n+2
        num_to_char = {j:i for i,j in char_to_num.items()}
        return char_to_num, num_to_char

    def __len__(self):
        return len(self.df)

    def load_one(self, file_id, sequence_id):
        path = self.data_folder + f'{file_id}/{sequence_id}.npy'
        data = np.load(path) # seq_len, 3* nlandmarks
        return data
