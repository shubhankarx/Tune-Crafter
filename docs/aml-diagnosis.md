# AML Prediction — Diagnosis (why it isn't working)

> Investigation summary, June 2026. Captures why the **AML** custom-gesture path
> shows a prediction that never reacts to the user's gesture. Short version: the
> AML engine isn't learning — it returns chance-level models. This is upstream of
> the app, in the licensed DFKI engine, not in Tune-Crafter's client code.

## Symptom
The orange **Train Model (AML)** / **Predict (AML)** path shows a prediction, but
it never changes with the gesture you draw. (The blue TensorFlow.js path —
**Train Model** / **Predict a Gesture** — works fine and does learn gestures.)

## What we tested
Trained the AML Docker server (`discreteDocker2`, `localhost:5001`) directly with
data where the correct answers are known, then measured accuracy. Compared:

| Variable | Result |
|---|---|
| Fresh model vs known-good saved `sensors_30` | Both have only **trivial** atoms below each class constant |
| Iterations 50 → 400 | No change |
| Samples 20 → 200 per class | No change — **train accuracy stuck at 0.50** (2-class) |
| Matching lens: atoms-below-O vs full reserve | Both = chance (0.33 for 3-class, 0.50 for 2-class) |
| Engine's own `bestAverage` = (FPR+FNR)/2 | **Pinned at 0.5 the entire run — never learns** |

## Root cause
The trained atomization **partitions cleanly**:
- **Principal atoms** carry the class constant (`O[k]`) but **no feature constants**.
- **Feature-region atoms** carry feature constants but **no class tag**.
- **Zero atoms link a feature region to a class.**

The only classifier in the codebase (`get_misses` in `build_model.py` /
`evaluate_express.py`) scores an input by how many atoms-below-`O[k]` fail to
intersect the input's feature constants. With only the trivial principal atom
below each class, every class gets the same miss count → a tie → a meaningless,
unchanging prediction. And underneath, each per-class binary model
(`train_class`) never drops below chance error (`bestAverage = 0.5`), so it
cannot fit even its own training data.

**Conclusion:** AML prediction is blocked at the **engine/training** step, not in
the client. The engine is a closed-source, licensed `.so` (DFKI/ALMA) — making it
converge likely needs DFKI / supervisor (Janin Koch) input. See
[[publication-direction]] and [[aml-server-docker]].

## What this means for the app
- The two genuine client bugs found (atom matching should be **intersection** not
  subset; embedding should be the **LE+G thermometer** with
  `np.digitize(..., right=True)`, `nb = len(edges)+1`) are worth fixing so the
  client is *correct* for a working model — but fixing them does not turn
  chance into signal.
- The payload also can't fully reconstruct the embedding: `feat_conv` omits the
  `G` constants for middle bins.

## Options discussed (decision pending)
- **A (recommended):** Drive the "explanation-as-music" demo from the working
  TF.js NN now (NN per-class probabilities as the atoms/misses proxy); make the
  AML panel honestly report non-convergence; switch to AML later once it learns.
- **B:** Keep AML-only, make the panel honest about non-convergence, escalate the
  engine issue to DFKI / supervisor with this evidence.
- **C:** Keep investigating *why* `train_class` stays at chance (preprocessing,
  iteration/generation semantics, enforce steps, engine/license behaviour).

## Reproduce
Server must be running (see `docs/aml-server-setup.md`). Then, inside the
container, training labeled data and scoring held-out samples reproduces the
0.50 accuracy / `bestAverage 0.5` result at any sample size or iteration count.
