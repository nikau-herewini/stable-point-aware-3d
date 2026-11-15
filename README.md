# SPAR3D API

An API-only fork of [Stable Point-Aware 3D](https://github.com/Stability-AI/stable-point-aware-3d)
that reconstructs a textured GLB from one image. SPAR3D generates a conditioning
point cloud internally to improve reconstruction of unseen surfaces.

## API

- `POST /generate`: upload one PNG/JPEG in multipart field `image`; download GLB.
- `GET /healthz`: HTTP health.
- `GET /readyz`: model readiness and selected device.

The API has **no authentication**. Keep it on a trusted private network.
Generation settings are fixed; only one request runs at a time. Busy requests
return `503`. Images are limited to 10 MiB and 16 million pixels.

## Docker deployment

```sh
docker build -t spar3d-api:cpu .
```

CPU, NVIDIA CUDA, and AMD ROCm configurations target Linux amd64. CI builds and
smoke-tests all three configurations on pull requests and publishes images to
`ghcr.io/nikau-herewini/stable-point-aware-3d` on successful main-branch jobs.
Backend tags are `cpu`, `cuda`, and `rocm`; there is no `latest` alias.

Model weights and Hugging Face tokens stay outside images. Downloads use a
persistent cache. See the [deployment guide](docs/api.md) for gated-model access,
runtime commands, GPU requirements, low-VRAM mode, and the HTTP contract.
Image smoke checks do not verify full-model inference or GPU hardware execution.

## Development checks

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
ruff check api_server.py api_backend.py tests
```

Tests exercise the API and backend control flow without downloading model weights.
Docker builds also check native CPU geometry and texture operations.

## Model and license

Model access is gated at
[Hugging Face](https://huggingface.co/stabilityai/stable-point-aware-3d).
See [LICENSE.md](LICENSE.md) for the model and code license.

[Paper](https://arxiv.org/abs/2501.04689):

```BibTeX
@article{spar3d2025,
  title={{SPAR3D}: Stable Point-Aware Reconstruction of {3D} Objects from Single Images},
  author={Huang, Zixuan and Boss, Mark and Vasishta, Aaryaman and Rehg, James Matthew and Jampani, Varun},
  journal={arXiv preprint},
  year={2025}
}
```
