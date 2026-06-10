# AML Server (Docker) — Setup & Run

The AML explainability server is the **keystone dependency** for Tune-Crafter's
System 2 (custom gestures via AML). Tune-Crafter's client calls it at
`http://localhost:5001` — see
[`AML_TRAIN_URL` / `AML_PREDICT_URL`](../src/components/GestureComponent.tsx#L14-L15).

The server is **not** in this repo. It lives in a sibling project:
`/Users/shubhankar/Documents/GitSpace/Docker/`

## ⚠️ Which variant to use: `discreteDocker2`

There are three near-identical folders. Only one is wired for port 5001 (what
Tune-Crafter expects). The folder that has the README (`discreteDocker`) is
**mis-configured** (Flask on 5000, mapping on 5001).

| Variant | Flask `app.run(port=)` | `interactive.sh` `-p` | Works for Tune-Crafter (5001)? |
|---|---|---|---|
| `discreteDocker` (has README) | 5000 | 5001:5001 | ❌ mismatch |
| **`discreteDocker2`** | **5001** | **5001:5001** | ✅ **USE THIS** |
| `discreteDocker 2` | 5001 | 5000:5000 | ❌ mismatch |

## Run sequence

```bash
cd "/Users/shubhankar/Documents/GitSpace/Docker/discreteDocker2"

# 1. Build the image (once). Requires Docker Desktop running.
docker build --tag alma_inria .

# 2. Start the container (maps 5001, bind-mounts cwd -> /home).
./interactive.sh        # docker run -p 5001:5001 -v $PWD:/home -it alma_inria /bin/bash

# 3. Inside the container shell:
cd home
python3 server.py       # Flask serves on 0.0.0.0:5001

# 4. Verify from the host (separate terminal):
curl -X POST http://localhost:5001/train/5 \
  -H "Content-Type: application/json" \
  -d '{"x":[[0,0,0],[1,1,1]],"y":["a","b"]}'
```

Then `npm run dev` Tune-Crafter and click the **train** button (System 2 / Custom
Test Mode).

## API contract

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/train/<nIter>` | POST | `{ x: number[][], y: string[] }` | `{ edges, classes: { <label>: [feat_conv, atoms] } }` |
| `/predict/` | POST | `{ x: [features] }` | **NOT IMPLEMENTED** — server.py has no predict route |

### Known gaps
- **No `/predict/` endpoint.** `predictGestureAML` in Tune-Crafter will 404.
  Training + the explainability payload (`edges`/`classes`/`atoms`) work; live
  AML prediction needs a `/predict` route added to `server.py`.
- **Feature confound** (see `docs/research-notes.md`): client NN now uses 46-dim
  stroke features; for a fair AML-vs-NN study both must eat the same features.

### Image build notes
- Base `ubuntu:22.04`; installs python3 + pip + tqdm, cffi, numpy, scikit-learn,
  astropy, pandas, flask, flask-cors, msgpack.
- Server depends on bundled native libs `libaml_core.so`, `libcrypto.so.3` and
  pre-trained `sensors_*.aml` files — these are mounted via `-v $PWD:/home`, so
  always run `server.py` from `/home` inside the container.
