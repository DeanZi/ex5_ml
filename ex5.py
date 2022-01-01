import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from gcommand_dataset import GCommandLoader
import torch


def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg)
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 30)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


def validate(model, e):
    validation_loss = 0.0
    correct_predictions = 0
    model.eval()
    for data, labels in validation_loader:
        target = model(data)
        loss = F.nll_loss(target, labels)
        validation_loss += loss.item()
        _, prediction = target.max(1)
        correct_predictions += (prediction == labels).sum()
        print(
            '\nValidation set: Accuracy: {:.0f}%\n'.format(100. * correct_predictions / len(validation_loader.dataset)))
        print(f'Validation Epoch {e + 1} \t\t Validation Loss: {validation_loss / len(validation_loader)}')


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
            validate(model, e)

    if with_test:
        test(model)


if __name__ == '__main__':
    dataset = GCommandLoader('train')
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True,
        pin_memory=True)

    dataset = GCommandLoader('validation')
    validation_loader = torch.utils.data.DataLoader(
        dataset, pin_memory=True)

    dataset = GCommandLoader('test')
    test_loader = torch.utils.data.DataLoader(
        dataset, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = VGG().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(model, optimizer, 10, False)