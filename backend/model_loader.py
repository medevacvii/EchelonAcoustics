import torch
import json
from backend.vgg_model import VGGSmall

def load_vgg_model(model_path, label_map_path):
    # Load label map
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    num_classes = len(label_map)

    # Build the VGG model with the correct number of output neurons
    model = VGGSmall(num_classes=num_classes)

    # Load model weights
    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    # Some checkpoints may have state_dict wrapped
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Load weights (will fail if architecture doesn't match the checkpoint)
    model.load_state_dict(state)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, label_map, device