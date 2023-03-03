#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import random
import os.path
from pathlib import Path

import torch
import torchvision
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset

from extra_models.alexnet import AlexNet
from extra_models.resnet import ResNet
from extra_models.densenet import DenseNet
from extra_models.wrn import WideResNet
from extra_models.mobilenetv2 import cifar100_mobilenetv2_x1_4
from extra_models.vgg import cifar100_vgg16_bn
from extra_models.shufflenetv2 import cifar100_shufflenetv2_x2_0


os.environ["CUDA_VISIBLE_DEVICES"]=str(0); 
device = torch.device("cuda:0")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic  = True
torch.set_num_threads(1)

BATCH_SIZE = 64


def get_model(model_name): 
    class_name = {
        "AlexNet": AlexNet,
        "ResNet": ResNet,
        "WideResNet": WideResNet,
        "DenseNet": DenseNet,
        "VGG16": cifar100_vgg16_bn,
        "MobileNetV2": cifar100_mobilenetv2_x1_4,
        "ShuffleNetV2": cifar100_shufflenetv2_x2_0        
    }
    
    class_parms = {
        "AlexNet": {"num_classes": 100},
        "ResNet": {"num_classes": 100, "depth": 164, "block_name": "bottleNeck"},
        "WideResNet": {"num_classes": 100, "depth": 28, "widen_factor": 10, "dropRate": 0.3},
        "DenseNet": {"num_classes": 10, "depth": 190, "growthRate": 40},
        "VGG16": {},
        "MobileNetV2": {},
        "ShuffleNetV2": {}        
    }
    
    path_parms = {
        "AlexNet": "./extra_models/cifar100_AlexNet/__model.ckpt",
        "ResNet": "./extra_models/cifar100_ResNet/__model.ckpt",
        "WideResNet": "./extra_models/cifar100_WideResNet/__model.ckpt",
        "VGG16": "./extra_models/cifar100_VGG16/__model.ckpt",
        "MobileNetV2": "./extra_models/cifar100_MobileNetV2/__model.ckpt",
        "ShuffleNetV2": "./extra_models/cifar100_ShuffleNetV2/__model.ckpt"
    }
    
    model = class_name[model_name](**class_parms[model_name])

    state_dict = torch.load(path_parms[model_name])["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '')] = state_dict.pop(key)
            
    model.load_state_dict(state_dict)
    model.to(device)   
    model.eval()
    
    return model

def get_features(model, x, _type):
    if _type == "gap":    
        return model.obtain_features_gap(x)
    
    if _type == "gap_all":    
        return model.obtain_features_gap_all(x)
    
    if _type == "gmp":    
        return model.obtain_features_gmp(x)
    
    if _type == "scda":    
        return model.obtain_features_scda(x)
    
    if _type == "crow":    
        return model.obtain_features_crow(x)    
    
def get_df(model, loader, _type):
    all_out = []
    features = None
    with torch.no_grad():
        for i_batch, (data, targets) in enumerate(loader):
            sys.stdout.write("{}/{} -- {}\r".format(i_batch, len(loader), features.shape if features is not None else ""))
            sys.stdout.flush()

            data = data.to(device)

            outputs = model(data)
            features = get_features(model, data, _type)
            for i in range(len(data)):
                out = {}
                out["id"] = i_batch * batch_size + i
                out["original_label"] = targets[i].item()
                out["features"] = np.array(features[i].detach().cpu())
                out["classifier"] = np.array(outputs[i].detach().cpu())
                all_out.append(out)
                        
    df = pd.DataFrame(all_out)
    df["predicted_label"] = df["classifier"].apply(lambda x: np.argmax(x))
    return df

def save_df(path, df):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)

    np.set_printoptions(suppress=True, threshold=np.inf, precision=8, floatmode="maxprec_equal")

    df = df.rename(index={0: "id", 1: "original_label", 2: "features", 3: "classifier"})
    df.to_pickle(path+".pickle")

class LimitedDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return min(len(self.dataset), self.n)

    def __getitem__(self, i):
        return self.dataset[i]


batch_size = BATCH_SIZE
dataset_path = "./datasets/"

cifar_normalize_mean = (0.5071, 0.4867, 0.4408)
cifar_normalize_std = (0.2675, 0.2565, 0.2761)

cifar_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(cifar_normalize_mean, cifar_normalize_std)
])

cifar100_train_set = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=cifar_transform)
cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train_set, batch_size=batch_size, shuffle=False)

cifar100_test_set = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=cifar_transform)
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test_set, batch_size=batch_size, shuffle=False)

cifar10_set = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=cifar_transform)
cifar10_set = LimitedDataset(cifar10_set, len(cifar100_test_set))
cifar10_loader = torch.utils.data.DataLoader(cifar10_set, batch_size=batch_size, shuffle=False)

svhn_set = torchvision.datasets.SVHN(root=dataset_path, split="train", download=True, transform=cifar_transform)
svhn_set = LimitedDataset(svhn_set, len(cifar100_test_set))
svhn_loader = torch.utils.data.DataLoader(svhn_set, batch_size=batch_size, shuffle=False)

noise_set = torchvision.datasets.FakeData(size=len(cifar100_test_set), image_size=(3, 32, 32), num_classes=100, transform=cifar_transform)
noise_loader = torch.utils.data.DataLoader(noise_set, batch_size=batch_size, shuffle=False)

loaders = {
    "train": cifar100_train_loader,
    "test": cifar100_test_loader,
    "cifar10": cifar10_loader,
    "svhn": svhn_loader,
    "noise": noise_loader,
}    

models = ["AlexNet", "ResNet",  "VGG16", "MobileNetV2", "ShuffleNetV2", "WideResNet"]
extract_methods = ["gap", "gmp", "scda", "crow", "gap_all"]

for i_model, model_name in enumerate(models):
    print("\n", model_name)
    model = get_model(model_name)
    for extract_method in extract_methods:
        print("--- {}".format(extract_method))
        for loader_name, loader in loaders.items():           
            path = "./features/{}/{}/{}".format(extract_method, model_name, loader_name)
            if os.path.exists(path+".pickle"):
                print(">>>>>> {} >>> already exist: {}".format(loader_name, path))
                continue
                
            df = get_df(model, loader, extract_method)
            save_df(path, df)
            
            print(">>>>>> {} >>> saved: {}".format(loader_name, path))
        torch.cuda.empty_cache()
