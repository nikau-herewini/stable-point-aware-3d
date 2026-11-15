"""SPAR3D inference adapter. Heavy dependencies load only during startup."""

import logging
import os
from contextlib import nullcontext

from api_server import InvalidImage

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self):
        import torch
        from transparent_background import Remover

        from spar3d.device import select_device

        self.torch = torch
        self.device = select_device()
        # Helpers inside SPAR3D must use the same device, without repeated probes.
        os.environ["SPAR3D_DEVICE"] = self.device
        from spar3d.system import SPAR3D

        self.model = SPAR3D.from_pretrained(
            os.environ.get("SPAR3D_MODEL", "stabilityai/stable-point-aware-3d"),
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=os.environ.get("SPAR3D_LOW_VRAM", "0") == "1"
            and self.device == "cuda",
        ).to(self.device).eval()
        # Keep foreground removal independent of the accelerator vendor.
        self.bg_remover = Remover(device="cpu")
        logger.info("SPAR3D ready on %s", self.device)

    def generate(self, image):
        from spar3d.utils import foreground_crop, remove_background

        foreground = remove_background(image, self.bg_remover)
        try:
            with foreground.getchannel("A") as alpha:
                bounds = alpha.getbbox()
            if bounds is None or bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
                raise InvalidImage("No usable foreground was detected.")
            prepared = foreground_crop(foreground, 1.3)
        finally:
            if foreground is not image:
                foreground.close()
        try:
            with self.torch.inference_mode():
                with (
                    self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                    if self.device == "cuda" and self.torch.cuda.is_bf16_supported()
                    else nullcontext()
                ):
                    mesh, _ = self.model.run_image(
                        prepared,
                        bake_resolution=1024,
                        remesh="none",
                        vertex_count=-1,
                    )
        finally:
            prepared.close()
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise InvalidImage("No mesh could be reconstructed from the image.")
        return mesh.export(file_type="glb", include_normals=True)
