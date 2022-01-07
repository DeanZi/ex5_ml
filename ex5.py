import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import os

from gcommand_dataset import GCommandLoader
import torch

from typing import List, cast


class VGG(nn.Module):
    def __init__(
            self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 30)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


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


def test(model):
    predictions = []
    model.eval()
    for data in test_loader:
        target = model(data)
        _, prediction = target.max(1)
        predictions.append(prediction)
    output_file = open('test_y', 'w')
    for y in predictions:
        output_file.write(str(y.item()) + '\n')
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
    dataset = GCommandLoader('data/train')
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True,
        pin_memory=True)

    dataset = GCommandLoader('data/valid')
    validation_loader = torch.utils.data.DataLoader(
        dataset, pin_memory=True)

    # dataset = GCommandLoader('data/test')
    # test_loader = torch.utils.data.DataLoader(
        # dataset, pin_memory=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = VGG(make_layers(cfg, batch_norm=True), num_classes=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, optimizer, 17, True)
