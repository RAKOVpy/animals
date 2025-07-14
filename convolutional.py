import torch
from datasets import get_loaders
# from model import SimpleCNN
from model import model
from trainer import train_model
from utils import count_parameters

device = torch.device('cpu')

train_loader, test_loader = get_loaders(batch_size=32)

# simple_cnn = SimpleCNN(num_classes=10).to(device)

model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=512, out_features=128),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(in_features=128, out_features=10),
    torch.nn.Softmax(dim=1),
)
model.requires_grad_(False)
model = model.to(device)

model.layer4.requires_grad_(True)
model.layer3.requires_grad_(True)
model.fc.requires_grad_(True)

print(f"Simple CNN parameters: {count_parameters(model)}")

print("Training Simple CNN...")
simple_history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))

'''
early_stopping = EarlyStopping(
    monitor="val_loss",   # Мониторим validation loss
    patience=5,           # Сколько эпох ждать улучшения
    mode="min",           # Минимизируем loss (для accuracy — "max")
    verbose=True,
)

trainer = Trainer(
    callbacks=[early_stopping, checkpoint],
    max_epochs=100,
)
trainer.fit(model, datamodule)'''
