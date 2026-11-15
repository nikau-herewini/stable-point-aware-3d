# SPAR3D API deployment

This API converts 1 image into 1 textured GLB. It matches the image-only contract
of the Stable Fast 3D API fork. It does not expose point-cloud editing, generation
options, batching, or the Gradio UI. The existing CLI and ComfyUI files remain.

## Build

Requires Docker with BuildKit on Linux amd64. The host needs sufficient disk space
and RAM for PyTorch, compilation, and model weights. No GPU is needed at build time.

```sh
docker build -t spar3d-api:cpu .
docker build --target api --build-arg BACKEND=cuda -t spar3d-api:cuda .
docker build --target api --build-arg BACKEND=rocm \
  --build-arg BASE_IMAGE=rocm/dev-ubuntu-22.04:6.2.4@sha256:030304ab0a88778b627b2eb2354c419146e105f5ac39ffd9c2438c82ef54b584 \
  -t spar3d-api:rocm .
```

The stages install PyTorch 2.5.1 and torchvision 0.20.1 from the CPU, CUDA 12.4,
or ROCm 6.2 wheel index. ROCm uses the PyTorch device name `cuda`. GPU support
depends on the host GPU, drivers, and installed PyTorch operators. These recipes
are not a guarantee of support for every GPU. MPS is not an API deployment target.

Native extensions compile as portable CPU operators for every backend. The
texture baker transfers tensors to CPU and returns results to the original
device. This adds transfers and CPU work to GPU inference. Build-time checks
import SPAR3D and exercise CPU grid sampling, UV unwrapping, and texture baking.
They do not download weights or verify full-model inference or GPU execution.

The Dockerfile installs torch first, constrains its ABI and NumPy, installs
`requirements-model.txt` and `requirements-api.txt`, then builds native extensions
with `--no-build-isolation`. Runtime source dependencies also use the pinned
setuptools without isolation for AlphaCLIP's legacy `pkg_resources` import.
It does not use the demo or ComfyUI dependencies.
Git dependencies have commit pins. Transitive Python packages, apt packages, and
model revisions are not fully locked. Never pass tokens as build arguments.

## Start on CPU

1. Accept the license and request model access at
   <https://huggingface.co/stabilityai/stable-point-aware-3d>.
2. Create a Hugging Face read token with access to that repository.
3. Supply it at runtime:

```bash
read -rs -p 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
docker run -d --name spar3d-api --init --restart unless-stopped \
  --stop-timeout 120 -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN -e SPAR3D_DEVICE=cpu \
  -v spar3d-model-cache:/cache spar3d-api:cpu
docker logs -f spar3d-api
```

The API has **no authentication**. Keep it on a trusted private network. Examples
bind to loopback only. Use an authenticated proxy and access controls before
wider deployment. Docker administrators can inspect container environment values.
Protect the host, token, and gated model cache.

Startup downloads SPAR3D, AlphaCLIP, and foreground-removal weights. It can take
several minutes and needs internet access on the first run. Missing access,
incompatible operators, download failures, and insufficient memory abort startup.
CPU inference can be very slow. Upstream reports approximately 10.5 GB VRAM,
or 7 GB with low-VRAM mode; actual memory use depends on hardware and inputs.

```sh
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/readyz
curl --fail-with-body -F 'image=@object.png' \
  http://127.0.0.1:8000/generate --output model.glb
```

Check curl's exit status: `--fail-with-body` can write an error body to `model.glb`.

## GPU startup

For NVIDIA, install a compatible host driver and NVIDIA Container Toolkit:

```sh
docker run --rm --init --gpus all -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN -e SPAR3D_DEVICE=cuda -e SPAR3D_LOW_VRAM=1 \
  -v spar3d-model-cache:/cache spar3d-api:cuda
```

For a ROCm-supported AMD GPU and host, provide device access and the device groups:

```sh
docker run --rm --init -p 127.0.0.1:8000:8000 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -e HF_TOKEN -e SPAR3D_DEVICE=cuda -e SPAR3D_LOW_VRAM=1 \
  -v spar3d-model-cache:/cache spar3d-api:rocm
```

Adjust the render-node path for the host. Do not use privileged mode to bypass
permissions. Background removal always uses CPU. GPU autocast uses bfloat16 only
when PyTorch reports support; otherwise inference uses its normal precision.

## Runtime configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPAR3D_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` (also ROCm) |
| `SPAR3D_LOW_VRAM` | `0` | Set `1` to dynamically load GPU model modules; ignored on CPU |
| `SPAR3D_MODEL` | `stabilityai/stable-point-aware-3d` | Hugging Face model ID or mounted model directory |
| `HF_TOKEN` | unset | Runtime token for gated model access |

Auto selection probes native operations before model loading and can select CPU
if the GPU probe fails. Explicit devices fail startup if unavailable. Model-load
or inference failures do not trigger an automatic retry. The legacy
`SPAR3D_USE_CPU=1` wins over `auto`, but not an explicit `SPAR3D_DEVICE=cuda`.

Run exactly **1 worker** without reload. Additional workers each load another
model and have independent admission locks. Scale separate containers only with
explicit memory and GPU allocation.

The container runs as UID/GID `10001:10001`. A named volume at `/cache` persists
Hugging Face, AlphaCLIP, Torch, and transparent-background downloads. Bind mounts
must be writable by UID 10001. Model files are not baked into the image.
Deleting the cache volume discards the downloaded models.

Docker checks `/readyz` every 30 seconds, with a 10-minute startup allowance.
Use `--health-start-period=20m` for slower downloads. Unhealthy status alone does
not make Docker restart a container. Inspect startup logs.

## HTTP contract for clients

- `POST /generate`: multipart form with exactly 1 file field named `image`.
  No additional files, text fields, URLs, or paths are accepted.
- Single-frame PNG/JPEG only, at least 2 pixels per dimension, at most 10 MiB and
  16,000,000 pixels. EXIF orientation is applied and transparency is preserved.
- The whole request is limited to 10 MiB plus 64 KiB, including chunked uploads.
  Uploads have a 30-second deadline.
- Success: `200`, `Content-Type: model/gltf-binary`, attachment `model.glb`,
  `Cache-Control: no-store`. Results are not stored on the server.
- Foreground images are downscaled to a longest side of 2048 before square
  cropping, to bound padded-image allocations. Degenerate foregrounds are rejected.
- Fixed generation: foreground crop ratio `1.3`, texture resolution `1024`,
  no remeshing. SPAR3D generates its own conditioning point cloud internally.
- Only 1 request is admitted at a time. Busy requests return `503` and
  `Retry-After: 5`, before reading their bodies. There is no queue. Cancellation
  does not release admission while native inference is still running.
- Errors: malformed multipart `400`, upload deadline `408`, oversized upload
  `413`, wrong content type `415`, invalid image/foreground/mesh `422`, generation
  failure `500`, not ready or busy `503`. Never interpret error bytes as a GLB.
- `GET /healthz`: `{"status":"ok"}`.
- `GET /readyz`: `{"status":"ready","device":"cpu"}` or selected device;
  otherwise `503`. Readiness does not mean idle. Uvicorn starts serving only
  after model loading completes.
- Clients must allow long inference timeouts, especially on CPU. Do not
  automatically resubmit a request with an unknown outcome.

## CI and publication

The API workflow runs weight-free unit tests and lint, then builds Linux amd64
CPU, CUDA, and ROCm images with native CPU smoke checks. Pull requests do not
publish. Successful main-branch jobs publish backend tags to
`ghcr.io/nikau-herewini/stable-point-aware-3d`: `cpu`, `cuda`, and `rocm`, plus
`sha-<full40commit>-<backend>`. There is no `latest` tag. Publication requires
GitHub Actions and GHCR permissions. Use these references only after the
corresponding publication job succeeds; initial package visibility may be private.
For immutable deployment, use an image digest, not a moving tag.

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
ruff check api_server.py api_backend.py tests
```

Unit tests use fake model/native dependencies. Neither these tests nor image
smoke checks prove full-model output quality, gated-model access, GPU execution,
or sufficient memory. Verify a real PNG/JPEG-to-GLB request on the deployment
hardware before integration.
