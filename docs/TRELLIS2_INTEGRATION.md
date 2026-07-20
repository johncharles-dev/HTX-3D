# Adding TRELLIS.2 as an HTX-3D engine

TRELLIS.2 can't run inside the `htx-3d` container (it needs the host's sm120 torch 2.10+cu128 /
from-source FlashAttention-2 stack). So the 4B model runs as a **host-side microservice** and the
backend gets a thin **`trellis2` proxy engine** that forwards each image to it. To the UI and API it
behaves like any other engine (it even gets auto-scale for free).

```
Browser ─▶ htx-3d container ─▶ Trellis2Engine (proxy)
                                   │  HTTP POST /generate  (http://host.docker.internal:8710)
                                   ▼
                         Host TRELLIS.2 microservice  ── trellis2 conda env, RTX 5090
                         (/home/cj/TRELLIS.2/service/)
```

## 1. Start the host microservice (must be running before using trellis2)

Manual:
```bash
/home/cj/TRELLIS.2/service/run_service.sh        # serves on :8710, model lazy-loads on first request
curl http://127.0.0.1:8710/health                # {"status":"ok","model_loaded":...}
```

Autostart on boot (systemd):
```bash
sudo cp /home/cj/TRELLIS.2/service/trellis2-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trellis2-service
systemctl status trellis2-service
journalctl -u trellis2-service -f              # logs
```

## 2. Apply the backend/frontend changes (requires an image rebuild)

The backend code and frontend are **baked into the image** (not bind-mounted), so code changes only
take effect after a rebuild + recreate:

```bash
cd /home/cj/HTX-3D/docker
docker compose build          # rebuilds frontend + backend into the image
docker compose up -d --force-recreate
docker logs -f htx-3d         # watch for "TRELLIS.2 engine registered"
```

Verify:
```bash
curl -s http://localhost:8000/api/health | grep -o 'trellis2'      # should appear in engines_registered
```
Then pick **TRELLIS.2** in the web UI engine selector and generate as usual.

## What changed (for review / removal)

Host (new, TRELLIS.2 repo):
- `service/trellis2_service.py`   — FastAPI microservice (model + /generate)
- `service/run_service.sh`        — launch script (sm120 / FA2 env)
- `service/trellis2-service.service` — systemd unit

Backend:
- `backend/app/services/trellis2.py`  — **new** `Trellis2Engine` proxy
- `backend/app/main.py`               — import + `tm.register_engine(trellis2)` + shutdown unload
- `backend/app/config.py`             — `TRELLIS2_SERVICE_URL`
- `backend/app/routers/generate.py`   — `_model_id_for_engine` branch; text-endpoint guard; Form doc
- `backend/app/models/schemas.py`     — `ModelType.TRELLIS2_IMAGE`
- `docker/docker-compose.yml`         — `TRELLIS2_SERVICE_URL` env + `extra_hosts: host.docker.internal:host-gateway`

Frontend:
- `frontend/src/types/index.ts`       — ModelType/EngineName + MODELS entry (id `trellis2`)
- `frontend/src/App.tsx`              — engine filter includes `trellis2`
- `frontend/src/api/client.ts`        — trellis2 target_face_count passthrough

## To remove TRELLIS.2 later (all changes are additive)

1. Delete `backend/app/services/trellis2.py`.
2. Revert the 3 `main.py` edits (import, register block, unload guard).
3. Remove the `trellis2` lines from `config.py`, `generate.py` (2), `schemas.py`, and the
   `docker-compose.yml` env + extra_hosts.
4. Remove the `trellis2` entries from `types/index.ts`, `App.tsx`, `client.ts`.
5. `docker compose build && docker compose up -d --force-recreate`.
6. `sudo systemctl disable --now trellis2-service` and stop the host service.

## Rebuild gotcha: UniDepth (auto-scale)

`auto_scale` (real-world sizing) needs `unidepth`, which was historically hand-installed into the
container and **NOT in the Dockerfile** — so the first rebuild silently disabled auto-scale for every
engine (`reason: "No module named 'unidepth'"`). Fixed: Dockerfile step 12 now installs it
(`--no-deps` to protect the pinned torch; `wandb` is imported by unidepth at load), commit-pinned.
If you ever see auto-scale return `auto_scaled: false, reason: No module named 'unidepth'`, restore it
in a running container with:
```bash
docker exec htx-3d pip install wandb
docker exec htx-3d pip install --no-deps --no-build-isolation \
  "git+https://github.com/lpiccinelli-eth/UniDepth.git@8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
```

## Notes

- **VRAM:** the host service keeps the ~7 GB model resident between requests. With the container's
  one-engine-at-a-time policy and 32 GB total, this coexists fine; restart the service to reclaim it.
- **Params / quality:** TRELLIS.2 honors `texture_size` and `target_face_count` (→ mesh decimation,
  default 200k). Because the tool's shared default texture is 1024 (looks soft/pixelated on TRELLIS.2's
  high-detail meshes), the proxy floors a ≤1024 request to **4096** (`services/trellis2.py` export_mesh);
  explicit 2048/4096 from the UI are honored. It does not use the TRELLIS ss/slat sliders; image-only.
- **Segmentation:** if a SAM3-segmented image is provided it is used directly (RGBA); otherwise the
  service removes the background itself (RMBG-2.0).
