import os
import random
from string import ascii_letters
from unidecode import unidecode

import torch
from torch import nn
from sklearn.model_selection import train_test_split

_ = torch.manual_seed(42)

#data_dir = "./data/names"
# lang2label = {
#     file_name.split(".")[0]: torch.tensor([i], dtype=torch.long)
#     for i, file_name in enumerate(os.listdir(data_dir))
# }

def get_lang2label (data_dir) :
    lang2label = {
        file_name.split(".")[0]: torch.tensor([i], dtype=torch.long)
        for i, file_name in enumerate(os.listdir(data_dir))
    }
    return lang2label

def get_char2idx () :

    char2idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}
    return char2idx

def get_tensor(data_dir) :
    tensor_names = []
    target_langs = []
    
    lang2label = get_lang2label(data_dir)
    char2idx = get_char2idx()
    
    num_letters = len(char2idx)
    num_langs = len(lang2label)
    print('num_lang, {} num_letters {}'.format(num_langs, num_letters))
    
    def name2tensor(name):
        tensor = torch.zeros(len(name), 1, num_letters)
        for i, char in enumerate(name):
            tensor[i][0][char2idx[char]] = 1
        return tensor    
    
    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file)) as f:
            lang = file.split(".")[0]
            names = [unidecode(line.rstrip()) for line in f]
            for name in names:
                try:
                    tensor_names.append(name2tensor(name))
                    target_langs.append(lang2label[lang])
                except KeyError:
                    pass
                
    return tensor_names, target_langs, num_langs, num_letters

def get_train_test_data(tensor_names, target_langs) :
    train_idx, test_idx = train_test_split(
        range(len(target_langs)), 
        test_size=0.1, 
        shuffle=True, 
        stratify=target_langs,
        random_state=1
    )

    train_dataset = [
        (tensor_names[i], target_langs[i])
        for i in train_idx
    ]

    test_dataset = [
        (tensor_names[i], target_langs[i])
        for i in test_idx
    ]    
    
    print(f"Train: {len(train_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset
