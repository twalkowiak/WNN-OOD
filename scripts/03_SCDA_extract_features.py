#!/usr/bin/env python
# coding: utf-8

from models.wrn import WideResNet
from models.densenet import DenseNet
from models.resnet import ResNet
from pathlib import Path
from torch.utils.data import Dataset
import queue
import torch.nn.functional as F
import torchvision
import torch
import random
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, '../')


# https://github.com/bearpaw/pytorch-classification

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
device = torch.device("cuda:0")

dataset_path = "../datasets/"
batch_size = 128

cifar_normalize_mean = (0.4914, 0.4822, 0.4465)
cifar_normalize_std = (0.2023, 0.1994, 0.2010)
cifar_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(cifar_normalize_mean, cifar_normalize_std)
])

# based on https://github.com/PyRetri/PyRetri


class SCDA(torch.nn.Module):
    def __init__(self):
        super(SCDA, self).__init__()

    def bfs(self, x: int, y: int, mask: torch.tensor,
            cc_map: torch.tensor, cc_id: int) -> int:
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        q = queue.LifoQueue()
        q.put((x, y))

        ret = 1
        cc_map[x][y] = cc_id

        while not q.empty():
            x, y = q.get()

            for (dx, dy) in dirs:
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                    if mask[new_x][new_y] == 1 and cc_map[new_x][new_y] == 0:
                        q.put((new_x, new_y))
                        ret += 1
                        cc_map[new_x][new_y] = cc_id
        return ret

    def find_max_cc(self, mask: torch.tensor) -> torch.tensor:
        """
        Find the largest connected component of the mask。
        Args:
            mask (torch.tensor): the original mask.
        Returns:
            mask (torch.tensor): the mask only containing the maximum connected component.
        """
        assert mask.ndim == 4
        assert mask.shape[1] == 1
        mask = mask[:, 0, :, :]
        for i in range(mask.shape[0]):
            m = mask[i]
            cc_map = torch.zeros(m.shape)
            cc_num = list()

            for x in range(m.shape[0]):
                for y in range(m.shape[1]):
                    if m[x][y] == 1 and cc_map[x][y] == 0:
                        cc_id = len(cc_num) + 1
                        cc_num.append(self.bfs(x, y, m, cc_map, cc_id))

            max_cc_id = cc_num.index(max(cc_num)) + 1
            m[cc_map != max_cc_id] = 0
        mask = mask[:, None, :, :]
        return mask

    def forward(self, x):
        fea = x
        if fea.ndimension() == 4:
            mask = fea.sum(dim=1, keepdims=True)
            thres = mask.mean(dim=(2, 3), keepdims=True)
            mask[mask <= thres] = 0
            mask[mask > thres] = 1
            mask = self.find_max_cc(mask)
            fea = fea * mask

            gap = fea.mean(dim=(2, 3))
            gmp, _ = fea.max(dim=3)
            gmp, _ = gmp.max(dim=2)

            return torch.cat([gap, gmp], dim=1)
        else:
            raise Exception("SCDA: Wrong dimension")


scda = SCDA()


def obtain_features(model, x):
    if isinstance(model, ResNet) or isinstance(model, DenseNet):
        for name, module in model._modules.items():
            if name == 'avgpool':
                return scda(x)
            else:
                x = module(x)

    if isinstance(model, WideResNet):
        for name, module in model._modules.items():
            x = module(x)
            if name == 'relu':
                return scda(x)


def get_df(model, loader):
    all_out = []
    with torch.no_grad():
        for i_batch, (data, targets) in enumerate(loader):
            sys.stdout.write("{}/{}\r".format(i_batch, len(loader)))
            sys.stdout.flush()

            data = data.to(device)

            outputs = model(data)
            features = obtain_features(model, data)
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

    np.set_printoptions(
        suppress=True,
        threshold=np.inf,
        precision=8,
        floatmode="maxprec_equal")

    df = df.rename(
        index={
            0: "id",
            1: "original_label",
            2: "features",
            3: "classifier"})
    df.to_pickle(path + ".pickle")


def load_model(model_class, parms, path):
    state_dict = torch.load(path)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)

    model = model_class(**parms)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class LimitedDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return min(len(self.dataset), self.n)

    def __getitem__(self, i):
        return self.dataset[i]


cifar10_train_set = torchvision.datasets.CIFAR10(
    root=dataset_path, train=True, download=True, transform=cifar_transform)
cifar10_train_loader = torch.utils.data.DataLoader(
    cifar10_train_set, batch_size=batch_size, shuffle=False)

cifar10_test_set = torchvision.datasets.CIFAR10(
    root=dataset_path,
    train=False,
    download=True,
    transform=cifar_transform)
cifar10_test_loader = torch.utils.data.DataLoader(
    cifar10_test_set, batch_size=batch_size, shuffle=False)

cifar100_set = torchvision.datasets.CIFAR100(
    root=dataset_path,
    train=True,
    download=True,
    transform=cifar_transform)
cifar100_set = LimitedDataset(cifar100_set, len(cifar10_test_set))
cifar100_loader = torch.utils.data.DataLoader(
    cifar100_set, batch_size=batch_size, shuffle=False)

svhn_set = torchvision.datasets.SVHN(
    root=dataset_path,
    split="train",
    download=True,
    transform=cifar_transform)
svhn_set = LimitedDataset(svhn_set, len(cifar10_test_set))
svhn_loader = torch.utils.data.DataLoader(
    svhn_set, batch_size=batch_size, shuffle=False)

noise_set = torchvision.datasets.FakeData(
    size=len(cifar10_test_set), image_size=(
        3, 32, 32), num_classes=10, transform=cifar_transform)
noise_loader = torch.utils.data.DataLoader(
    noise_set, batch_size=batch_size, shuffle=False)

loaders = {
    "train": cifar10_train_loader,
    "test": cifar10_test_loader,
    "cifar100": cifar100_loader,
    "svhn": svhn_loader,
    "noise": noise_loader
}

models = {
    "cifar10_resnet": {
        "model_class": ResNet,
        "parms": {"num_classes": 10, "depth": 164, "block_name": "bottleNeck"},
        "path": "../models/cifar10_resnet.pth.tar"
    },

    "cifar10_wrn": {
        "model_class": WideResNet,
        "parms": {"num_classes": 10, "depth": 28, "widen_factor": 10, "dropRate": 0.3},
        "path": "../models/cifar10_wrn.pth.tar"
    },

    "cifar10_densenet": {
        "model_class": DenseNet,
        "parms": {"num_classes": 10, "depth": 190, "growthRate": 40},
        "path": "../models/cifar10_densenet.pth.tar"
    }
}


def run():
    for model_name, model_info in models.items():
        print("\n", model_name)
        model = load_model(**model_info)
        for loader_name, loader in loaders.items():
            print(">>>", loader_name)
            path = "../features/scda/{}/{}".format(model_name, loader_name)
            df = get_df(model, loader)
            save_df(path, df)
            print(">>>>>> Saved:", path)
    torch.cuda.empty_cache()


def run_attacks():
    _path = "../attacked_images"
    for model_name, model_info in models.items():
        print("\n", model_name)
        model = load_model(**model_info)

        attack_model_path = "{}/{}/".format(_path, model_name)
        method_folders = [
            o for o in os.listdir(attack_model_path) if os.path.isdir(
                os.path.join(
                    attack_model_path, o))]
        for method_name in method_folders:
            attack_path = "{}/{}".format(attack_model_path, method_name)
            _set = torchvision.datasets.ImageFolder(
                root=attack_path, transform=cifar_transform)
            _loader = torch.utils.data.DataLoader(
                _set, batch_size=batch_size, shuffle=False)

            print(">>>", method_name)
            path = "../features/scda/{}/{}".format(model_name, method_name)
            df = get_df(model, _loader)
            save_df(path, df)
            print(">>>>>> Saved:", path)
        torch.cuda.empty_cache()


run()
run_attacks()
