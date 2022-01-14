import ntpath
import torch.nn as nn
from torch import optim
import numpy as np
import os
from gcommand_dataset import GCommandLoader
import torch


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


def validate(model, e):
    loss_func = nn.CrossEntropyLoss()
    validation_loss = 0.0
    correct_predictions = 0
    model.eval()
    for data, labels in validation_loader:
        target = model(data.to(device)).to(device)
        loss = loss_func(target, labels.to(device))
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
    model.load_state_dict(torch.load('current_state/model_state_new.pt'))
    model.eval()
    for index, (data, label) in enumerate(test_loader):
        target = model(data.to(device)).to(device)
        _, prediction = target.max(1)
        test_file = path_leaf(dataset_test.spects[index][0])
        predictions.append(test_file + ',' + classes[prediction])
    output_file = open('test_y', 'w')
    for y in predictions:
        output_file.write(y + '\n')
    output_file.close()


def train(model, optimizer, epoch, with_validation, with_test=False):
    loss_func = nn.CrossEntropyLoss()
    if with_validation:
        best_validation_loss = np.inf
    for e in range(epoch):
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device)).to(device)
            loss = loss_func(output, target.to(device))
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
                torch.save(model.state_dict(), 'current_state/model_state_new.pt')

    if with_test:
        test(model)


if __name__ == '__main__':
    dataset_train = GCommandLoader('gcommands/train')
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=100, shuffle=True,
        pin_memory=True)
    classes = dataset_train.classes

    dataset_validation = GCommandLoader('gcommands/valid')
    validation_loader = torch.utils.data.DataLoader(
        dataset_validation, pin_memory=True)

    dataset_test = GCommandLoader('gcommands/test')
    test_loader = torch.utils.data.DataLoader(
        dataset_test, pin_memory=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = VGG(make_layers(cfg, batch_norm=True), num_classes=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, 12, True, True)
