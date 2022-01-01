from gcommand_dataset import GCommandLoader
import torch

dataset = GCommandLoader('train')


dataset = GCommandLoader('train')

train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True,
         pin_memory=True)

for input,label in train_loader:
    print(f"input shape : {input.shape}, label shape : {len(label)}")

# test_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=100, shuffle=None,
#         num_workers=20, pin_memory=True, sampler=None)
#
# for input,label in test_loader:
#     print(input.size(), len(label))
