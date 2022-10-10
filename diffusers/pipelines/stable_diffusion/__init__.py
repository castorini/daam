from dataclasses import dataclass, field
from typing import List, Union, Any

import numpy as np

import PIL
from PIL import Image

from ...utils import BaseOutput, is_onnx_available, is_transformers_available


@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content.
    """
    pil_images: Union[List[PIL.Image.Image], np.ndarray]
    images: Union[List[PIL.Image.Image], np.ndarray]
    intermediates: List[Any] = field(default_factory=list)
    nsfw_content_detected: List[bool] = field(default_factory=list)
    text_embeddings: Any = None
    words: List[str] = field(default_factory=list)


if is_transformers_available():
    from .pipeline_stable_diffusion import StableDiffusionPipeline
    from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
    from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from .safety_checker import StableDiffusionSafetyChecker

if is_transformers_available() and is_onnx_available():
    from .pipeline_stable_diffusion_onnx import StableDiffusionOnnxPipeline
