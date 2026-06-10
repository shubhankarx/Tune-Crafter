# Tune Crafter — Research & Publication Notes

> Working notes for turning the Master's thesis ("Understanding Digital Music Production
> in the Times of Machine Learning", Shubhankar, supervised by Janin Koch, Ex-Situ/LISN,
> ALMA project) into a publishable contribution. Captures the analysis from the
> planning sessions so nothing is lost. Living document — update as decisions are made.

---

## 1. Where the code stands (after the recent work session)

- **Fixed gestures (System 1):** MediaPipe gesture recognizer → drums, play/pause, volume,
  speed, cut/loop. Original, working.
- **Custom gestures (System 2):** record → train local **TensorFlow.js NN** → predict.
  - Rewrote features to **stroke features**: arc-length resample (16 pts) + centre/scale
    normalization + per-point **turning angles** (corner detector). 46-dim vector.
  - Confidence threshold (0.8) + N-frame stability (5) gating; in-flight predict guard;
    dispose-on-retrain; unified `runPredict`.
  - **Custom Test Mode toggle** isolates System 2 (turns off System 1 interference).
  - Live cyan **trail** + on-canvas **label + confidence overlay**.
  - **Predict button = deliberate capture→classify** (2.5s draw window; avoids classifying
    the reach-for-the-mouse motion).
  - Visible **training status** (epoch/accuracy → "✓ Trained on: …"), per-label sample counts.
- **AML server hooks:** `trainModelAML` / `predictGestureAML` POST to `http://localhost:5001`.
  NOT running / untested. **Keystone dependency** for the whole contribution.
- All client-side; in-browser ML needs no server. Type-checks clean.

### ⚠️ Two known constraints
- **AML server is the keystone.** Without it (or a stub), the AML-vs-NN comparison and any
  AML explanation can't run — that's the entire contribution.
- **Feature confound.** Thesis AML path uses 50-dim raw window (10 frames × 5 values);
  our NN now uses 46-dim stroke features. For a fair AML-vs-NN study, **both models must eat
  the SAME features**, else you compare features+model, not model.

---

## 2. Gap analysis — does the thesis fill a research gap?

**Verdict: it FRAMES a real gap but does not yet FILL it.**

- NOT novel: "gesture-controlled music" (saturated) or "user-trainable gestures"
  (= Wekinator / Fiebrink-Caramiaux / Marcelle).
- Genuinely novel angle: **AML (parameter-free, atom-based, explainable-in-principle) as the
  substrate for END-USER interactive ML in a creative tool, vs a NN baseline.** AML is barely
  present in HCI.
- Why not yet filled (thesis admits both):
  1. **No summative study** — only a formative interview study (N=7, all male) + a usage scenario.
  2. **The explanation surface is the LEAST-built part** — system shows confidence numbers,
     not *why*. AML's interpretable objects (atoms / misses / discretization bins) are never
     surfaced to the user. (Thesis: "still exploring how to provide better user feedback…")
  3. Weak feature representation (now partly addressed).

**What would convert "frames" → "fills":**
1. Build the **AML explanation interface** (atoms/misses → human-legible "why").
2. Run the **AML-vs-NN comparison study** measuring understanding / trust / ML-literacy
   (not just accuracy).

---

## 3. Five candidate contributions (creative directions)

1. **Explainability as music** — sonify AML's internal state (atoms/misses/confidence) as the
   output: missing atoms mute notes of a chord; confidence drives timbre. Distinct from generic
   data sonification. Uniquely enabled by AML's *discrete* structure (a softmax can't do it).
2. **End-user explanation UI for AML** — visualize atoms / discretization bins / misses to
   non-experts. (Has anyone built human-facing XAI for AML at all? — under research.)
3. **OctoPocus for LEARNED gestures** — real-time counterfactual coaching ("round this corner,
   you're 1 atom short"). Original OctoPocus (Bau & Mackay 2008, our ref [9]) guided *predefined*
   gestures; here the vocabulary is *ML-learned*.
4. **Mental-model probe as an evaluation** — measure whether users can *predict* the model's
   output (simulatability); test if explainability improves it. Novel dependent measure for a
   creative tool.
5. **Incremental-learning demo** — AML "recalls what it knows when taught something new";
   show/hear AML keeping old atoms while the NN must retrain and wobbles. Thesis claims this,
   never demonstrates it.

**Killer demo = 1+3+4+5 combined:** draw → chord swells as atoms match / thins on misses →
sweep toward boundary morphs timbre → wrong gesture summons coaching ghost → add a 3rd gesture
and *hear* AML keep its knowledge while NN stumbles. Explanation + music + ML lesson as one.

(ASCII mockups of all ten feature ideas are in the chat transcript; reproduce here when a
direction is chosen.)

---

## 4. Venue analysis

| Venue | Fit | Note |
|---|---|---|
| **DIS** | **Strongest near-term** | Rewards the design contribution you already have (study→implications→artifact); forgives the missing summative study; **Pictorial** option fits the visual material |
| **IUI** | High (for XAI) | Explainable + interactive ML is its core |
| **ACM C&C** | High | Creativity support + CSI instrument |
| **NIME** | High | Gesture music instrument; loves prototypes + demos |
| **UIST** | Possible | Needs a sharpened *technical* novelty (the AML explanation technique); **Demo track** is a strong low-risk fit now |
| **CHI** | Reach | Full-paper bar high on novelty; **LBW** very achievable |

**Throughline across all venues:** the **AML explanation interface** is the single piece that
upgrades "another IML music tool" → a real contribution.

---

## 5. Reading list (verify citations)

**Closest prior art (must out-position):**
- Fiebrink, Cook & Trueman — "Human Model Evaluation in Interactive Supervised Learning" (CHI 2011) ← read first
- Fiebrink & Caramiaux — "The ML Algorithm as Creative Musical Tool" (2016) [thesis ref 34]
- Amershi et al. — "Power to the People: Humans in Interactive ML" (AI Mag 2014) [ref 42]
- Françoise & Bevilacqua — "Motion-Sound Mapping through Interaction" (ACM TiiS)
- Marcelle — Françoise, Caramiaux, Sanchez (UIST 2021) [ref 15]

**Human-centered XAI (weakest flank — fix first):**
- Miller — "Explanation in AI: Insights from the Social Sciences" (Artificial Intelligence 2019)
- Liao, Gruen & Miller — "Questioning the AI" (CHI 2020)
- Kulesza et al. — "Explanatory Debugging" (IUI 2015) ← blueprint for the explanation interface
- Wang, Yang, Abdul & Lim — "Designing Theory-Driven User-Centric XAI" (CHI 2019)
- Abdul et al. — XAI HCI agenda (CHI 2018)
- HCXAI workshop series (CHI 2021–2024)
- Bryan-Kinns et al. — "Exploring XAI for the Arts" (2023) [ref 33] + 2024 follow-ups

**Creativity-support evaluation:**
- Cherry & Latulipe — Creativity Support Index (ACM ToCHI 2014)
- Frich et al. — "Mapping the Landscape of Creativity Support Tools" (CHI 2019)

**Design-research rigor (for DIS):**
- Zimmerman, Forlizzi & Evenson — "Research through Design…" (CHI 2007)
- Dourish — "Implications for Design" (CHI 2006)
- Sas et al. — "Generating Implications for Design through Design Research" (CHI 2014)

**Gesture grounding + baseline:**
- Wobbrock, Wilson & Li — "$1 Unistroke Recognizer" (UIST 2007) ← reviewers will ask why not compared
- Caramiaux et al. — "Adaptive Gesture Recognition…" (TiiS 2014) [ref 39]

**Method:** forward-citation ("Cited by", 2022–2025) on Fiebrink 2011, AML 2018, OctoPocus,
Bryan-Kinns 2023.

---

## 6. Development plan (3 buckets)

**Bucket 1 — Study apparatus (buildable now, no AML server):** per-trial logging → CSV/JSON
export; guided counterbalanced study mode; in-app CSI/NASA-TLX/SUS + understanding/trust/literacy
questionnaires; **confusion matrix** per model (thesis planned, never built); **model/session
persistence** (today wiped on refresh — fatal for a study).

**Bucket 2 — Train/test models:** side-by-side AML-vs-NN prediction (gated on server); expose NN
config (epochs + hidden size); per-model confusion matrix + per-class accuracy; NN train/val
split; **feature alignment** (shared representation).

**Bucket 3 — Explainable-AI interface (the prize, gated on AML server/stub):** "why this gesture"
atoms/misses panel; per-feature discretization view; confidence provenance; AML-vs-NN contrast;
counterfactual ("what would make this a square"); explanatory-debugging loop.

**Sequencing:** Bucket 1 + NN parts of Bucket 2 now → unblock AML server (or stub) + align
features → Bucket 3 (the contribution).

**Recommended first slice:** sonified misses + live atom ladder (directions 1+2) against a local
AML stub — novel core, no live server needed.

---

## 7. Status / open items

- [ ] **Deep-research report** (run `wf_a0f66d9b-a81`) — ranks the 5 directions by novelty vs
      current literature; **to be saved here** when complete (Section 8).
- [ ] Decide: AML server live vs local stub (+ DFKI licensing check for public hosting).
- [ ] Decide shared feature representation (resolve the AML/NN confound).
- [ ] Pick the first build slice + target venue.

## 8. Deep-research findings (workflow wf_a0f66d9b-a81 — 105 agents, 23 sources, 25 claims verified, 22 confirmed)

### Ranking of the 5 directions
| # | Direction | Novelty | Verdict |
|---|---|---|---|
| **1** | **Explainability as music** (sonify AML atoms/misses/confidence as the output) | **HIGH** | ⭐ **Pursue as core contribution** |
| 2 | End-user explanation UI for AML | **LOW** | ❌ **AVOID as headline — occupied by the supervisor** |
| 3 | OctoPocus for ML-LEARNED gestures | MED-HIGH (under-researched) | Plausible secondary; "supervisor anticipates it" claim was **refuted (0-3)** → NOT occupied, but closest prior work not pinned down |
| 4 | Simulatability / mental-model accuracy as eval | MEDIUM | ✅ **Use to EVALUATE Direction 1** (it's a method, not a contribution) |
| 5 | Incremental / no-forgetting demo vs NN | MEDIUM (thin evidence) | Weakest-supported; defer |

### 🚨 The critical finding (changes strategy)
**Direction 2 is already occupied — by the thesis's own supervisor.**
**Koch & Fortes Rey, arXiv:2507.06751 (July 2025)** built an AML gesture-typing app that streams data to the AML model, gets back *"the values of the features used to predict the recognized class, as well as the prediction itself and its confidence level… maps these features to expressive forms."* This (a) **rules out Direction 2** as a standalone novelty, and (b) **partially overlaps Direction 1** — so Direction 1 must be carefully differentiated from the supervisor's own follow-up (and ideally framed *with* her, since it's her active research line). Their differentiator: it maps *"features used"* → expression, **not the miss-structure**, and has **no user study**.

### Recommendation
**Pursue Direction 1 (explainability-as-music) as the core, evaluate it with Direction 4 (simulatability), target NIME or ACM C&C / the XAIxArts track (secondary: IUI).**

**One-sentence claim:** *"We make a parameter-free model's internal interpretable state (AML atoms, misses, and per-class confidence) directly audible as the musical output it controls — turning the explanation itself into the sound — and show users can better predict (simulate) the model's behaviour as a result."*

### Why Direction 1 is genuinely open (closest prior work + how to differentiate)
- **Schuller et al., "Towards sonification in… XAI" (ICMI 2021)** — a *vision/position paper, no prototype, no study*; sonifies **data-factor saliency / input attribution**, not internal state. → differentiate on *built + studied + internal symbolic state*.
- **audioLIME / CoughLIME** — explain models whose **input is audio** (which audio segments mattered); input attribution, not internal state. (The "CoughLIME = explanation-is-sound" counter-claim was **killed 1-2** in verification.)
- **AIive (IEEE VIS 2021)** — sonifies **training metrics** (loss/accuracy) for *artistic representation*, explicitly not explanation/per-class state.
- **ICAD / Sonification Handbook** — sonification is an established field → novelty is the **WHAT** (a model's intrinsic interpretable state as the controlled musical output), not sonification itself.
- **Bryan-Kinns XAIxArts (C&C 2023)** + **Tecks et al. (C&C 2024)** — confirm creative-XAI is *young* and frames explanation as data-curation/"define rather than explain," not surfacing internal state.

### Why AML is the right substrate
AML is parameter-free, fully discrete, grows from data via cardinal minimization; ALMA (CORDIS 952091) states its discrete-set/graph representations are "a good starting point for generating human understandable descriptions." Atoms/misses/confidence are a *genuine* internal interpretable state — categorically distinct from saliency.

### ⚠️ The central risk (and why Direction 4 neutralises it)
**AML's interpretability is self-asserted** (ALMA/CORDIS is aspirational, not peer-reviewed); **there is no evidence atoms are comprehensible to non-expert users.** This is exactly why pairing with **Direction 4 (simulatability)** is the smart move: you don't *assume* the sonified atoms are understandable — you **empirically test** whether they improve users' ability to predict the model. The risk becomes the research question. (Hase & Bansal 2020: subjective ratings ≠ how helpful explanations are; Fiebrink CHI 2011: in a music tool, accuracy negatively correlated with user satisfaction r≈−0.4 to −0.7 — "the explanation is in the behaviour.")

### Caveats from the researcher
- Forward-citation search (Fiebrink 2011, AML 2018, OctoPocus, Bryan-Kinns 2023) yielded **no confirmed surviving claims** → 2024–2026 competitors may exist that this pass missed. Do a manual "Cited by" sweep before committing.
- Directions 3 & 5 are **low-confidence** (thin/refuted evidence) — re-research if you want them.
- Supervisor's arXiv:2507.06751 is the #1 thing to read next and to discuss with Janin.

### Key sources
- Koch & Fortes Rey 2025 — arxiv.org/abs/2507.06751 ← **read first**
- Schuller et al. ICMI 2021 — opus.bibliothek.uni-augsburg.de/…/91495.pdf
- Hase & Bansal 2020 (simulatability) — arxiv.org/pdf/2005.01831
- Fiebrink, Cook & Trueman CHI 2011 — doc.gold.ac.uk/~mas01rf/publications/FiebrinkCookTrueman_CHI2011.pdf
- Bryan-Kinns XAIxArts C&C 2023 — dl.acm.org/doi/10.1145/3591196.3593517
- Tecks et al. XAIxArts 2024 — arxiv.org/pdf/2407.15216
- AML — arxiv.org/abs/1803.05252 ; ALMA — cordis.europa.eu/project/id/952091
- OctoPocus (Bau & Mackay 2008) — lri.fr/~mbl/Stanford/CS477/papers/Octopocus-UIST2008.pdf

### 8a. Digest of the supervisor's paper (Koch & Fortes Rey 2025) — and how to differentiate

**"Combining Human-centred Explainability and Explainable AI"** — Janin Koch & Vitor Fortes Rey
(DFKI). A **6-page HCXAI position paper**, 2025, funded by ALMA (ERC 952091). **No user study**
("this project is currently in progress, and will be published alongside an in-depth assessment
of the AML-model approach's explainability potential").

What it actually does:
- Frames **HCx** (contextualize a system's contribution in the user's *ongoing task*) vs **xAI**
  (explain how the system *works* internally), and argues to **combine both**.
- Lists AML's interpretability advantages: simple **3-layer structure** (inputs–atoms–outputs);
  human-defined constraints preserved as atoms grow; **AML-DL rules trace back to human policies**.
  Notes the **final atom space can be converted to a height-limited CART decision tree** for
  readability (a concrete interpretability route worth reusing).
- **Application = gesture *TYPING***: raw position → AML returns "the values of the features used
  to predict the recognized class, the prediction, and its confidence level"; the app maps these
  to **typographic expression (font, colour, style)** — see their Fig 1 (swipe keyboard). They can
  also "visualize the xAI features (e.g. gesture speed/acceleration)" and "display alternative
  gesture spaces for the user's intended goals."
- **No sonification, no sound, no music.** No simulatability / understanding measure.

**→ Positioning Direction 1 vs. this paper (the differentiation paragraph):**
Their work maps AML's *features used* to *typographic* expression in gesture **typing**, with no
study. Tune Crafter's Direction 1 differs on three axes — **(a) modality:** we sonify to
*sound/music*, not type (their work has no sonification); **(b) explanatory target:** we render the
**miss-structure** ("why-not": which atoms are *absent*) plus confidence, not only the features
used; **(c) empirical claim:** we run a **simulatability** study, which neither their work nor the
AML originators have done. Crucially it also **instantiates their own thesis**: they argue HCx
(task context) should be fused with xAI (internals) — and sonifying AML's internal state *as the
musical output the user is creating* is exactly that fusion (xAI internals expressed inside the
ongoing creative HCx task). So: differentiate on modality + misses + study; align on the HCx+xAI
frame; and **coordinate with Janin** so your study doesn't collide with her planned "in-depth
assessment."

### 8b. Deeper analysis of Janin's paper — what it means for the contribution

1. **"Explanation as music" is the PUREST instance of her central thesis.** Her real argument:
   fuse **xAI** (how the system *works* — atoms/features/confidence) with **HCx** (contextualize
   in the user's *ongoing task*), designed into the model's inner workings. Sonifying the
   explanation AS the music delivers the xAI internal state *entirely inside the HCx task* (the
   sound being made) with **no separate explanation surface, zero context-switch** — the limit
   case of the combination her paper only argues for.

2. **She hands you a warning that sharpens the design:** *"expressing parameters of how the
   system 'works'… is often less relevant to the end user's intent, [leading to] information
   fatigue and a decrease in trustworthiness."* → Don't sonify atoms as a data dump (that's the
   xAI failure she warns against). Sonify the **task-relevant** thing: "will this gesture
   reliably do what I want, and what do I change?" Chord-fullness = reliability; dissonant
   "why-not" note = actionable. That turns xAI into HCx — answering her objection in the design.

3. **Her listed AML mechanisms = sonification primitives:** (a) 3-layer "atom present if ≥1 input
   present" → chord notes (done); (b) atom space → height-limited **CART decision tree** → sonify
   the *path through the tree* as a melodic phrase (hear the reasoning chain, not just the
   verdict); (c) "AML-DL rules trace to human-defined policies" → explanation in the user's own
   terms ("you taught me a square has corners…").

4. **MayAI thread (her CHI 2019, cited in the paper):** its explainable engine used *"semantic
   and visually perceivable information… instead of explaining learned relations,"* "very valuable
   in a realistic ideation study." Sonification is the *audible* analog. Pitch: "MayAI made
   explanation visually perceivable for design; Tune Crafter makes it audibly perceivable for
   music — same philosophy, new modality."

5. **Study redefinition:** measure **task-relevant simulatability** — can the musician predict
   whether a gesture will be recognised and adjust to make it reliable? — not generic "predict
   the label." Sharper and aligned with her HCx framing.

6. **Strategic:** her paper *names the open problem you'd solve* (defers "an in-depth assessment
   of AML's explainability potential") → points to **co-authorship, not competition**. Add the
   **HCXAI workshop (CHI)** as an early venue (her community) alongside NIME / C&C-XAIxArts.

**Sharpened claim:** "We fuse xAI and human-centred explainability to their limit — a model's
internal interpretable state becomes the musical output the user creates, so the explanation needs
no separate surface — and show this audible, in-the-flow explanation lets musicians predict and
control the model in their task."

---

## 9. XAI methods landscape & where AML fits (workflow wf_92483904-b77 — 105 agents, 23 sources, 25 claims verified, 23 confirmed)

> **Scope note:** §8 is a *novelty ranking* of the 5 directions. This section is the *methods/SOTA
> landscape* — "what most people actually do in XAI, and where AML sits in it" — to ground the paper's
> Related-Work and to pick credible baselines + evaluation. Does not re-litigate §8's strategy.
> AML placement here is anchored on the thesis's **Appendix D** (the real AML mechanism), not guesswork.

### 9.1 The taxonomy everyone uses (3 axes)
Two independent 2024–25 surveys (Mersha et al. arXiv:2409.00265; Paterakis et al. arXiv:2508.11529)
converge on the same three axes:
- **Scope:** local (one prediction) vs global (whole model).
- **Training stage:** **intrinsic / ante-hoc** (interpretable by construction) vs **post-hoc** (explain after).
- **Access:** model-specific vs model-agnostic (this split nests *under* post-hoc).

### 9.2 What practitioners actually do (the "default")
- **The default is post-hoc, model-agnostic feature attribution — LIME and SHAP** — plus model-specific
  gradient/CAM methods (Integrated Gradients, Grad-CAM, LRP/DeepLIFT) for deep nets. The literature
  concentrates here *because black boxes usually out-accuracy interpretable models, so people explain
  them after the fact* (Paterakis 2025).
- **SHAP is the one to beat:** Lundberg & Lee (NeurIPS 2017) unify six prior attribution methods (incl.
  LIME, DeepLIFT) as "additive feature attribution," with a unique axiomatic solution (Shapley values).
- **Intrinsic family** the surveys enumerate: decision trees, GLMs, GAMs, KNN, Bayesian models, **RuleFit /
  decision lists**. ← this is AML's neighbourhood.
- **Concept-based** (frontier-ish): **TCAV** (Kim et al. ICML 2018) interprets internal state via
  *human-defined concepts*, not raw features. ← AML atoms are concept-like.

### 9.3 The critique current — your strategic high ground (Rudin)
- **Rudin 2019 (Nature Machine Intelligence)** — the most-cited XAI position paper: *"stop explaining
  black box models for high-stakes decisions; design models that are inherently interpretable instead."*
- **"Explanations must be wrong":** a perfectly faithful post-hoc explanation would *equal* the model, so a
  90%-faithful explanation is wrong 10% of the time. → an intrinsic explanation has **no fidelity gap**.
- **Accuracy–interpretability tradeoff is "a myth"** — *on structured data with meaningful features.*
  ⚠️ **Explicitly contested for raw audio/image/text.** Directly relevant to you (see 9.6 risk).
- **The disagreement problem** (Krishna et al. TMLR 2022): SHAP/LIME/gradients routinely *disagree* on the
  same prediction; 84% of practitioners hit this, 86% resolve it with arbitrary heuristics. → post-hoc is
  shaky ground; intrinsic side-steps it entirely.

### 9.4 How the field evaluates explanations (no agreed metric — use the trichotomy)
- **Doshi-Velez & Kim 2017** canonical trichotomy: **functionally-grounded** (proxy metrics, no humans) /
  **human-grounded** (lay users, simplified task) / **application-grounded** (real users, real task).
  Their working definition of interpretability — *"explain or present in understandable terms to a human"* —
  is the one to cite.
- **No universally accepted metric exists** (both surveys + DFKI INES 2023). Consensus = a **hybrid**:
  functionally-grounded **fidelity/faithfulness** + **human-grounded simulatability / trust / understanding**.
- Standard fidelity formulas are **perturbation-based and assume a post-hoc explainer separate from the
  model** → they don't cleanly apply to an intrinsic atom/miss explanation (open question, see 9.6).

### 9.5 Where AML actually sits (anchored on Appendix D)
Appendix D pins the mechanism: each feature is **discretized into `nb` bins** (ordered intensity) encoded as
`2(nb−1)` `≤`/`>` constants; a class is a constant `O_k`; an example is a term `T`; membership is the
**algebraic relation `O_k ≤ T`** (positive) with `O_{k'≠k} ⊄ T` (negative). An **atom** = a conjunction of
discretized-feature-intensity predicates a class requires; a **miss** = a required atom *absent* from `T`.

→ **AML = intrinsic / ante-hoc, concept- *and* rule-like interpretability.** Atoms ≈ TCAV concepts
(human-meaningful constituents) **and** ≈ RuleFit/decision-list rules (the surveys' intrinsic family); Janin's
paper even notes the atom space converts to a **height-limited CART tree**. The explanation *is the model
state*, so **Rudin's fidelity gap and the disagreement problem structurally do not apply.** That is the
paper's high ground: you are on the Rudin side of the field's central debate, not patching a black box.
*(Confidence: medium — this is a synthesis applying verified XAI claims to AML; no source verifies AML itself.)*

### 9.6 ⚠️ The risk the report surfaces (must address, don't assume away)
Rudin's "no tradeoff" holds for **structured/meaningful features**, NOT raw signal. Your domain is
gesture/audio. **If AML eats hand-crafted symbolic stroke features it is favoured; if the NN eats raw
signal it may legitimately win on accuracy.** → This is *exactly* why §1's **feature-alignment confound**
matters, and why the contribution must be **explanation quality, not accuracy** (don't claim AML is more
accurate; claim it is more *understandable* at comparable accuracy). Pre-empt the reviewer here.

### 9.7 Baselines the comparison NEEDS to be credible
1. **SHAP (and/or LIME) on the NN** — the canonical post-hoc benchmark; the explanation style AML is
   measured against (intrinsic atoms/misses vs post-hoc attributions, same input/prediction).
2. **One inherently-interpretable classical baseline** — decision tree / RuleFit / GAM / logistic
   regression — to test Rudin's no-tradeoff claim *on this task* and answer "why not just a decision tree?"
3. (From §1) **shared feature representation** so any difference is model, not features.

### 9.8 What's genuinely novel vs already-done (sonification)
- **Already established (cannot claim as novel):** intrinsic concept/rule interpretability; the need for
  human-grounded evaluation; sonification as a field (§8: ICAD/Sonification Handbook).
- **Defensible novelty = the *modality + setting*:** delivering an **intrinsic** model's **internal
  symbolic state (atoms/misses/confidence) as the real-time musical output** in an end-user creative tool.
  No source in either corpus (§8 or §9) shows auditory delivery of *internal-state* explanations.
  ⚠️ Absence-of-evidence, not proof — still owe a targeted ICAD/CHI/IUI/NIME prior-art sweep (§8 caveat).

### 9.9 Prioritized to-do for a credible, publishable contribution
1. **Resolve the feature confound + reframe the claim** (9.6): shared representation; contribution =
   *understanding at comparable accuracy*, not accuracy. ← cheapest, highest-leverage, kills the #1 reviewer attack.
2. **Stand up AML (server or stub)** so atoms/misses are real — keystone for everything (§1, §7).
3. **Build the three-arm comparison** (9.7): AML-sonified vs **NN+SHAP** vs no-explanation, shared features.
4. **Adopt the evaluation hybrid** (9.4): human-grounded **simulatability/forward-prediction** (the §8
   Direction-4 measure) + trust/understanding scales (CSI/SUS) + a defensible **faithfulness argument for
   intrinsic AML** (open question 9.6 — likely a "trivial fidelity" argument: the explanation is the model).
5. **Run the targeted sonification prior-art sweep** (9.8) to lock the novelty claim before submission.
6. **Position Related Work on the Rudin axis** (9.3): AML as intrinsic-by-design that escapes the
   fidelity gap + disagreement problem — the SOTA-relevant framing, citing Rudin 2019, Krishna 2022,
   Doshi-Velez & Kim 2017, Lundberg & Lee 2017, Kim 2018.

### Key new sources (XAI landscape — complement §8's HCI/sonification list)
- Rudin 2019 — nature.com/articles/s42256-019-0048-x ← **Related-Work anchor**
- Krishna et al. 2022 (disagreement problem) — arxiv.org/abs/2202.01602
- Doshi-Velez & Kim 2017 (eval trichotomy) — arxiv.org/pdf/1702.08608
- Lundberg & Lee 2017 (SHAP) — dl.acm.org/doi/10.5555/3295222.3295230
- Kim et al. 2018 (TCAV / concepts) — arxiv.org/abs/1711.11279
- Mersha et al. 2024 survey — arxiv.org/pdf/2409.00265 ; Paterakis et al. 2025 survey — arxiv.org/pdf/2508.11529
- Hase & Bansal 2020 (simulatability) — arxiv.org/pdf/2005.01831 *(also in §8)*

---

## 10. Thesis critique (Mackay-rigor lens) + how to finish the project

### 10.1 The real weaknesses (validating the supervisor's pushback)
- **Meta-issue (likely why she reacted):** her frameworks (co-adaptation, instrumental interaction,
  reification, OctoPocus, MayAI) are cited **decoratively, never applied**. Fix: make ONE load-bearing
  (analyze the gesture-training loop as co-adaptation, or design via instrumental interaction) or cut it.
- **Design implications D1–D5 (weakest part):** (a) don't follow from the data — the *gesture premise is
  imposed, not discovered* (themes are about workflow, never about wanting gestures); (b) generic enough to
  fit any music tool; (c) **3 of 5 orphaned** — Design Concept addresses only D1+D2, ignoring layering(D3)/
  channels(D4) which the study most emphasized. Fix: re-derive with traceable evidence+quotes, OR reframe as
  *requirements* (not study findings), OR drop the unused ones.
- **Study↔system mismatch:** study is about *composition* (layering, channels); prototype is *playback/DJ
  control*. Different problems. Fix: tighten scope so study matches artifact, or admit the prototype is a
  narrow D1/D2 probe.
- **Conclusion overclaims:** asserts AML "explainability… was demonstrated" with **no evaluation, no
  explainability study, only confidence numbers**. Title says "Explainable AI"; work never tested it. Fix:
  conclusion must match what was done (formative study + design concept + prototype); explainability eval =
  future work. Drop the "transcend the boundaries" rhetoric.
- **RQ1/2/3 are project phases, not research questions** (nothing falsifiable). **Methodology thin** (N=7 all
  male; ½-page analysis; cites Braun&Clarke but doesn't execute reflexive TA — no codebook/saturation).

### 10.2 What rigorous RESULTS look like at each stage
- **Stage 1 (formative):** codebook (codes→themes) with **prevalence** (x/7) + **2–3 verbatim quotes** per
  theme + reflexivity + saturation. Themes specific/non-obvious, ideally ones that motivate explainability.
- **Stage 2 (implications):** a **traceability matrix** `Theme → Implication → Design feature → Eval measure`
  — the auditable "golden thread."
- **Stage 3 (prototype) — OBTAINABLE NOW, currently zero numbers:** per-gesture **confusion matrix**,
  **accuracy vs #training-samples** curve, **AML-vs-NN** table (accuracy, training time, #atoms), **latency**.
- **Stage 4 (eval, the missing stage):** task completion/error, **SUS·NASA-TLX·CSI**, **simulatability**
  (predict the model), **trust calibration**; stats (Wilcoxon) + effect sizes + 95% CIs + honest negatives;
  qualitative think-aloud themes. *(A usage scenario is not a result.)*
- **Honest framing:** the problem isn't that Stage 4 is missing — it's that the conclusion claims Stage-4
  results on a Stage-3 scenario. Fix = produce the Stage-3 numbers now + reframe claims to the real stage.

### 10.3 Idea bank (24) + the SPINE to actually build
**A. Explanation interface (explainable angle):** 1 sonified reliability (chord fullness="will it fire");
2 **why-not voicing** (dissonant note=missing competing-class atom); 3 reasoning-path arpeggio (CART tree);
4 on-demand "explain this" atom replay; 5 counterfactual coaching ghost (OctoPocus-for-learned-gestures);
6 **verbosity slider** (operationalizes Janin's info-fatigue concern as a variable); 7 atom→training-example
provenance; 8 editable/explanatory-debugging loop; 9 decision-boundary sweep (theremin); 10 dual-channel
audible+visual.
**B. AML-vs-NN comparison / systems:** 11 in-browser real-time AML inference (latency/atoms — UIST result);
12 **side-by-side dual prediction**; 13 incremental-learning demo (AML keeps atoms, NN forgets);
14 **same-features harness** (fixes the confound).
**C. Measures:** 15 **simulatability** (headline); 16 counterfactual simulation; 17 **trust calibration**
(confidence tracks actual accuracy); 18 repair efficiency; 19 CSI+NASA-TLX; 20 think-aloud→mental-model themes.
**D. Protocol:** 21 confusable-pair task (circle/square); 22 teach-under-time-pressure; 23 within-subjects
counterbalanced explanation ON/OFF; 24 seeded vs self-trained.

**THE SPINE (build these 6, not 24):** #2 why-not + #6 verbosity slider + #12 side-by-side AML-vs-NN +
#14 same-features + #15 simulatability + #17 trust calibration.
**Venue routing:** run the study → **CHI**; lead with the technique (#2,3,5,11,13 + light study) → **UIST**;
lead with design tensions (#6,7,10,20 + reflection) → **DIS**.
**Spine's one-sentence claim:** "A sonified, in-the-flow explanation of a parameter-free model's internal
state — including why-not — lets musicians predict and appropriately trust a gesture model they train, which
a confidence number does not."
**Zero-AML-server starting points (highest leverage):** #14 same-features harness + #15 simulatability probe.
