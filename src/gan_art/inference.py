from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from gan_art.models import Generator


def load_generators(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generators = {}
    for name in ("photo_to_monet", "monet_to_photo"):
        generator = Generator().to(device)
        generator.load_state_dict(checkpoint[name])
        generator.eval()
        generators[name] = generator
    del checkpoint
    return generators, device


@torch.inference_mode()
def translate_image(image: Image.Image, generator: Generator, device: torch.device) -> Image.Image:
    preprocess = transforms.Compose([
        transforms.Resize(286, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    result = generator(tensor).squeeze(0).cpu().clamp(-1, 1)
    pixels = ((result + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)
