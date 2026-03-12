from pathlib import Path
import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).parent / "models" / "best_mlp_pose_model.pth"

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

ckpt = torch.load(MODEL_PATH, map_location="cpu")

class_map = ckpt.get("class_map", {
    "rock": 0,
    "scissor": 1,
    "paper": 2,
    "other": 3,
})
idx_to_class = {v: k for k, v in class_map.items()}

input_dim = ckpt.get("input_dim", 29)
num_classes = len(class_map)

model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def predict_pose(features):
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    label = idx_to_class[pred_idx]
    return label, confidence