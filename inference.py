import os
import glob
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm


class MyResNeSt(nn.Module):
    def __init__(self, model_name="resnest50d",
                 num_classes=100, pretrained=True):

        super().__init__()
        self.backbone = timm.create_model(model_name,
                                          pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


class TestImageDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name


test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model_configs = [
    ("resnest200e", "best_epoch_32_resnest.pth"),
    ("resnest200e", "best_epoch_39_resnest.pth"),
    ("resnest200e", "best_epoch_28_resnest.pth")
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 100

models_ensemble = []
for mname, pth in model_configs:
    model = MyResNeSt(model_name=mname,
                      num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(pth, map_location=device))
    model.to(device)
    model.eval()
    models_ensemble.append(model)

test_dir = "/path/to/data/test"
test_dataset = TestImageDataset(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, num_workers=4)

predictions = []
image_names = []
a = 0.2
weights = [1.0, 1.0, 1.0]

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)

        ensemble_logits = None
        for idx, model in enumerate(models_ensemble):
            outputs = model(images)
            if ensemble_logits is None:
                ensemble_logits = weights[idx] * outputs
            else:
                ensemble_logits += weights[idx] * outputs

        _, preds = torch.max(ensemble_logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        image_names.extend(names)

df = pd.DataFrame({"image_name": image_names, "pred_label": predictions})
df.to_csv("prediction.csv", index=False)
