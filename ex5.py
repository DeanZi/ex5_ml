import ntpath

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import os
import time
from gcommand_dataset import GCommandLoader
import torch

from typing import List, cast


class VGG(nn.Module):
    def __init__(
            self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


cfg = [64, "M", 128, "M", 256, 256, "M", 512, "M"]


def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def validate(model, e):
    validation_loss = 0.0
    correct_predictions = 0
    model.eval()
    for data, labels in validation_loader:
        target = model(data.to(device)).to(device)
        loss = F.nll_loss(target, labels.to(device))
        validation_loss += loss.item()
        _, prediction = target.max(1)
        correct_predictions += (prediction.to(device) == labels.to(device)).sum()
    print(
        '\nValidation set: Accuracy: {:.0f}%\n'.format(100. * correct_predictions / len(validation_loader.dataset)))
    print(f'Validation Epoch {e + 1} \t\t Validation Loss: {validation_loss / len(validation_loader)}')
    return validation_loss / len(validation_loader)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def test(model):
    predictions = []
    model.eval()
    for index, (data, label) in enumerate(test_loader):
        target = model(data.to(device))
        _, prediction = target.max(1)
        test_file = path_leaf(dataset_test.spects[index][0])
        predictions.append(test_file + ',' + classes[prediction])
    output_file = open('test_y', 'w')
    for y in predictions:
        output_file.write(y + '\n')
    output_file.close()


def train(model, optimizer, epoch, with_validation, with_test=False):
    if with_validation:
        best_validation_loss = np.inf
    for e in range(epoch):
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device)).to(device)
            loss = F.nll_loss(output, target.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            e + 1, train_loss / len(train_loader)))

        if with_validation:
            validation_loss = validate(model, e)
            if validation_loss <= best_validation_loss:
                print('Saving state of model...')
                best_validation_loss = validation_loss
                if not os.path.isdir('current_state'):
                    os.mkdir('current_state')
                torch.save(model.state_dict(), 'current_state/model_state.pt')

    if with_test:
        test(model)


if __name__ == '__main__':
    start_time = time.time()
    dataset = GCommandLoader('data/train')
    classes = dataset.classes
    new_ds = []
    index = 0
    for element in dataset:
        new_ds.append(element)
        index += 1
        if index == 50:
            break
    train_loader = torch.utils.data.DataLoader(
        new_ds, batch_size=256, shuffle=True,
        pin_memory=True)

    dataset = GCommandLoader('data/valid')
    print(dataset.class_to_idx)
    new_ds = []
    index = 0
    for element in dataset:
        new_ds.append(element)
        index += 1
        if index == 50:
            break
    validation_loader = torch.utils.data.DataLoader(
        new_ds, pin_memory=True)

    dataset_test = GCommandLoader('data/test')
    new_ds = []
    index = 0
    for i, element in enumerate(dataset_test):
        new_ds.append(element)
        index += 1
        if index == 50:
            break
    test_loader = torch.utils.data.DataLoader(
        new_ds, pin_memory=True)
    ent_time = time.time()
    print(f"Data was loaded from {abs(start_time - ent_time)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = VGG(make_layers(cfg, batch_norm=True), num_classes=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, optimizer, 2, True, with_test=True)
