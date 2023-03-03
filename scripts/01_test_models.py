#!/usr/bin/env python
# coding: utf-8

from models.wrn import WideResNet
from models.densenet import DenseNet
from models.resnet import ResNet
import torchvision
import torch
import random
import numpy as np
import os
import sys

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
device = torch.device("cuda:0")

dataset_path = "./datasets/"
batch_size = 64

cifar_normalize_mean = (0.4914, 0.4822, 0.4465)
cifar_normalize_std = (0.2023, 0.1994, 0.2010)
cifar_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(cifar_normalize_mean, cifar_normalize_std)
])


def test_model(name, model, test_loader):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("{} - acc: {:.2%}".format(name, correct / total))


def load_model(model_class, parms, path):
    state_dict = torch.load(path)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)

    model = model_class(**parms)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


cifar10_test_set = torchvision.datasets.CIFAR10(
    root=dataset_path,
    train=False,
    download=True,
    transform=cifar_transform)
cifar10_test_loader = torch.utils.data.DataLoader(
    cifar10_test_set, batch_size=batch_size, shuffle=False)

cifar10_resnet = load_model(
    ResNet, {"num_classes": 10, "depth": 164,
             "block_name": "bottleNeck"}, "./models/cifar10_resnet.pth.tar"
)

cifar10_wrn = load_model(
    WideResNet, {"num_classes": 10, "depth": 28, "widen_factor": 10,
                 "dropRate": 0.3}, "./models/cifar10_wrn.pth.tar"
)

cifar10_densenet = load_model(
    DenseNet, {"num_classes": 10, "depth": 190,
               "growthRate": 40}, "./models/cifar10_densenet.pth.tar"
)

print("---CIFAR-10---")
test_model("ResNet", cifar10_resnet, cifar10_test_loader)
test_model("WideResNet", cifar10_wrn, cifar10_test_loader)
test_model("DenseNet", cifar10_densenet, cifar10_test_loader)
