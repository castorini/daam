from collections import defaultdict
from pathlib import Path
from typing import List
from dataclasses import dataclass

import PIL.Image
import torch


__all__ = ['GenerationExperiment']


@dataclass
class GenerationExperiment:
    """Class to hold experiment parameters. Pickleable."""
    id: str
    image: PIL.Image.Image
    global_heat_map: torch.Tensor
    seed: int
    prompt: str

    def save(self, path: str):
        path = Path(path) / self.id
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self, path / 'generation.pt')
        self.image.save(path / 'output.png')

        with (path / 'prompt.txt').open('w') as f:
            f.write(self.prompt)

        with (path / 'seed.txt').open('w') as f:
            f.write(str(self.seed))
