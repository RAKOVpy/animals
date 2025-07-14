import torch

# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10, input_size=224):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         # Рассчитываем размер после свёрточных слоёв
#         size_after_conv = input_size // 4  # Два пулинга с stride=2 уменьшают размер в 4 раза
#         self.flatten_size = 64 * size_after_conv * size_after_conv
#
#         self.fc1 = nn.Linear(self.flatten_size, 256)
#         self.fc2 = nn.Linear(256, num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # [batch, channels*height*width]
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
