#!/usr/bin/env python
# coding: utf-8

from models.wrn import WideResNet
from models.densenet import DenseNet
from models.resnet import ResNet
from pathlib import Path
from PIL import Image
import shutil
import torch.nn as nn
import torchattacks
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

# os.environ["CUDA_VISIBLE_DEVICES"]=str(1);
device = torch.device("cuda:0")

dataset_path = "../datasets/"
batch_size = 32

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
    transform=torchvision.transforms.ToTensor())
cifar10_test_loader = torch.utils.data.DataLoader(
    cifar10_test_set, batch_size=batch_size, shuffle=False)

cifar10_resnet = load_model(
    ResNet, {"num_classes": 10, "depth": 164,
             "block_name": "bottleNeck"}, "../models/cifar10_resnet.pth.tar"
)

cifar10_wrn = load_model(
    WideResNet, {"num_classes": 10, "depth": 28, "widen_factor": 10,
                 "dropRate": 0.3}, "../models/cifar10_wrn.pth.tar"
)

cifar10_densenet = load_model(
    DenseNet, {"num_classes": 10, "depth": 190,
               "growthRate": 40}, "../models/cifar10_densenet.pth.tar"
)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, _input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (_input - mean) / std


def modify_model(model):
    norm_layer = Normalize(cifar_normalize_mean, cifar_normalize_std)
    return nn.Sequential(
        norm_layer,
        model
    ).to(device)


def save_image(path, image):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)

    torchvision.utils.save_image(image, path)


def get_predicted_classes(model, loader):
    guess_all = []
    predicted_all = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = torch.nn.functional.softmax(model(inputs), dim=1)

            values, predicted = outputs.max(1)
            predicted_all.extend(predicted)

            guess = predicted.eq(targets)
            for i, value in enumerate(values):
                if not guess[i] and value < 0.95:
                    o = [int(v * 100)
                         for v in outputs[i].cpu().detach().numpy()]
                    guess[i] = True
            guess_all.extend(guess)

    ok_indexes = [i for i, g in enumerate(guess_all) if g]
    wrong_indexes = [i for i, g in enumerate(guess_all) if not g]
    return ok_indexes, wrong_indexes, predicted_all


def get_model(name):
    if name == "cifar10_resnet":
        return cifar10_resnet

    if name == "cifar10_wrn":
        return cifar10_wrn

    if name == "cifar10_densenet":
        return cifar10_densenet


def get_attack(method, model, recursive):
    if method == "CW":
        return torchattacks.CW(model, steps=(1000 * (recursive + 1)), lr=0.01)

    if method == "DeepFool":
        return torchattacks.DeepFool(model, steps=(
            50 * (recursive + 1)), overshoot=(0.02 * (recursive + 1)))

    if method == "FGSM":
        return torchattacks.FGSM(model, eps=((8 + (2 * recursive)) / 255))

    if method == "OnePixel":
        return torchattacks.OnePixel(
            model, pixels=recursive + 1, steps=75, popsize=400)

    if method == "PGD":
        return torchattacks.PGD(model, eps=(
            (8 + (2 * recursive)) / 255), alpha=2 / 255, steps=7)

    if method == "Square":
        return torchattacks.Square(model, eps=((8 + (2 * recursive)) / 255))

    if method == "SparseFool":
        return torchattacks.SparseFool(model, steps=(
            20 * (recursive + 1)), overshoot=(0.02 * (recursive + 1)), lam=3)


def run(models, attack_methods, temp_name):
    for model_name in models:
        df = pd.DataFrame()
        print(">>>", model_name)
        model = get_model(model_name)
        _model = modify_model(model)
        for method in attack_methods:
            print("\n>>>>>>", method)
            number_of_adversarial_images = 1000
            recursive = 0
            stop = False

            while not stop and number_of_adversarial_images >= 0:
                attack = get_attack(method, _model, recursive)
                for i_batch, (inputs, targets) in enumerate(
                        cifar10_test_loader):
                    print("> {}:{} -- left: {}".format(recursive,
                          i_batch, number_of_adversarial_images))
                    if stop:
                        break

                    # attack
                    x = inputs.to(device)
                    y = targets.to(device)

                    torch.set_grad_enabled(True)
                    attacked_images = attack(x, y)
                    torch.set_grad_enabled(False)

                    # verify successful attacks
                    # a) save images
                    shutil.rmtree('{}/'.format(temp_name), ignore_errors=True)
                    for i in range(len(attacked_images)):
                        _cifar10_id = i_batch * batch_size + i
                        attack_path = "./{}/{}/{}".format(
                            temp_name, model_name, method)
                        name = "{}.{}.png".format(_cifar10_id, recursive)

                        save_image("{}/{}/{}".format(attack_path,
                                   y[i], name), attacked_images[i])

                    # b) test them
                    attack_path = "{}/{}/{}".format(temp_name,
                                                    model_name, method)
                    _set = torchvision.datasets.ImageFolder(
                        root=attack_path, transform=cifar_transform)
                    _loader = torch.utils.data.DataLoader(
                        _set, batch_size=batch_size, shuffle=False)
                    ok_indexes, wrong_indexes, predicted = get_predicted_classes(
                        model, _loader)

                    # c) copy successful ones
                    for i in wrong_indexes:
                        src = _set.imgs[i][0]
                        dst = src.replace(
                            "{}/".format(temp_name), "../attacked_images/")

                        directory = os.path.dirname(dst)
                        Path(directory).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dst)

                        number_of_adversarial_images -= 1
                        if number_of_adversarial_images == 0:
                            stop = True
                            break

                recursive += 1
                if recursive >= 10:
                    stop = True
                    break

            shutil.rmtree('{}/'.format(temp_name), ignore_errors=True)
            attack_path = "../attacked_images/{}/{}".format(model_name, method)
            _set = torchvision.datasets.ImageFolder(
                root=attack_path, transform=cifar_transform)
            _loader = torch.utils.data.DataLoader(
                _set, batch_size=batch_size, shuffle=False)
            test_model(attack_path, model, _loader)


models = ["cifar10_resnet", "cifar10_wrn", "cifar10_densenet"]
attack_methods = [
    "FGSM",
    "CW",
    "DeepFool",
    "FGSM",
    "OnePixel",
    "PGD",
    "Square",
    "SparseFool"]
run(models, attack_methods, "temp")
