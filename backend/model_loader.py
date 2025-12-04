import torch
import json
from backend.vgg_model import VGGSmall

def load_vgg_model(model_path, label_map_path):
    # Load label map
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    num_classes = len(label_map)

    # Build model with the correct number of output classes
    model = VGGSmall(num_classes=num_classes)

    # Load checkpoint (state dict only)
    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False  # ensure older checkpoints still load
    )

    # Some checkpoints might be wrapped like {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # This will still fail if the .pth is from the wrong architecture (e.g., 2-class)
    model.load_state_dict(state)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, label_map, device