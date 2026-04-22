import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import UNet
from dataset import CityscapesDataset
from metrics import iou_score, dice_score

from torch.utils.data import DataLoader
import torch.nn as nn

# paths
img_dir = "data/CameraRGB"
mask_dir = "data/CameraMask"

images = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
masks = sorted([os.path.join(mask_dir, i) for i in os.listdir(mask_dir)])

X_train, X_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

train_ds = CityscapesDataset(X_train, y_train)
test_ds = CityscapesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(num_classes=23).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

losses, ious, dices = [], [], []

for epoch in range(15):
    model.train()
    epoch_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        pred = model(img)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss)

    # evaluation
    model.eval()
    miou, mdice = 0, 0

    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)

            miou += iou_score(pred, mask)
            mdice += dice_score(pred, mask)

    ious.append(miou / len(test_loader))
    dices.append(mdice / len(test_loader))

    print(epoch, epoch_loss, ious[-1], dices[-1])

# save model
torch.save(model.state_dict(), "unet_city.pth")
# after training + validation loop finishes

# plots
plt.plot(losses)
plt.savefig("loss.png")

plt.plot(ious)
plt.savefig("iou.png")

plt.plot(dices)
plt.savefig("dice.png")
