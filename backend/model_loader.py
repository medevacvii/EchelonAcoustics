import torch
import json
from backend.vgg_model import VGGSmall   # must match exact filename

def load_vgg_model(model_path, label_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    num_classes = len(label_map)

    model = VGGSmall(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, label_map, device