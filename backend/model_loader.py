import torch
import json
from backend.vgg_model import VGGSmall  # import your model class

def load_vgg_model(model_path: str, label_map: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VGGSmall().to(device)

    # Load only state dict
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.eval()

    with open(label_map, "r") as f:
        labels = json.load(f)

    return model, labels, device