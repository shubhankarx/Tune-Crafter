// AmlModel — pure, client-side Algebraic Machine Learning inference + explanation.
//
// The AML Docker server (discreteDocker2/server.py) only implements
// `POST /train/<nIter>`; there is NO `/predict` route. But the train response
// already contains everything needed to predict AND explain a new input entirely
// in the browser, so this module does exactly that — no server round-trip per
// prediction, and AML eats the SAME stroke features as the local NN.
//
// How AML represents the problem (thesis Appendix D + server `ModelConverter`):
//   - Each feature f is discretized into `nb` bins (the `edges`).
//   - Each (feature, bin) maps to one integer "constant" id via `feat_conv`
//     (a thermometer-style `LE`/`G` constant — see build_const_map in the server).
//   - A class is the constant O_k; the model is the list of "atoms" BELOW O_k.
//     Each atom is a SET of constant ids it sits below (`atom.ucs`).
//   - An input embeds to its set of constants C(x) = { feat_conv[f][bin_f] }.
//     Positive relation O_k ≤ T means every atom of class k must be ≤ the input,
//     and an atom a is ≤ the input  ⟺  a.ucs ⊆ C(x).
//
// So: an atom is MATCHED when its constants are all present in the input, and a
// MISS otherwise. The class with the fewest misses wins; confidence is the
// matched fraction. Misses are the human-legible "why-not" the explanation shows.
//
// NOTE: each trained class carries its OWN `feat_conv` (constant ids are minted
// per-class by the server's constant manager), so embedding is done per-class
// against that class's own feat_conv — never cross the id spaces.

import { Explanation } from "./ExplanationSonifier";

// ---- Raw server payload --------------------------------------------------------
// classes[label] = [feat_conv, atoms]
//   feat_conv: number[feature][bin] -> constant id
//   atoms:     number[atom][...]     -> the constant ids the atom is below (ucs)
export interface AmlPayload {
    edges: number[][];
    classes: { [label: string]: [number[][], number[][]] };
}

// ---- Parsed, ready-to-predict model -------------------------------------------
interface AmlClassModel {
    label: string;
    featConst: number[][]; // [feature][bin] -> constant id (this class's id space)
    atoms: Set<number>[];  // each atom as a Set for fast subset checks
}

export interface AmlModel {
    edges: number[][];
    featureCount: number;
    classes: AmlClassModel[];
}

// ---- Per-prediction results (what the UI renders) -----------------------------
export interface AmlAtomResult {
    ucs: number[];   // the atom's constant ids
    matched: boolean;
}

export interface AmlClassResult {
    label: string;
    atomsTotal: number;
    atomsMatched: number;
    matchedFraction: number;   // 0..1
    atoms: AmlAtomResult[];    // per-atom matched flag, for the atom ladder
}

export interface AmlPrediction {
    predictedClass: string;
    confidence: number;        // matched fraction of the predicted class
    perClass: AmlClassResult[];
}

// Validate + index the raw payload. Returns null if the shape is unusable.
export function parseAmlPayload(payload: AmlPayload | null | undefined): AmlModel | null {
    if (!payload || !payload.classes || !payload.edges) return null;
    const labels = Object.keys(payload.classes);
    if (labels.length === 0) return null;

    const classes: AmlClassModel[] = [];
    for (const label of labels) {
        const entry = payload.classes[label];
        if (!Array.isArray(entry) || entry.length < 2) continue;
        const [featConst, atoms] = entry;
        if (!Array.isArray(featConst) || !Array.isArray(atoms)) continue;
        classes.push({
            label,
            featConst,
            atoms: atoms.map(a => new Set(a)),
        });
    }
    if (classes.length === 0) return null;

    return {
        edges: payload.edges,
        featureCount: payload.edges.length,
        classes,
    };
}

// Map a continuous feature value to its discrete bin index [0, nb-1].
//
// `edges[f]` are the bin boundaries learned by the server's Bayes binning. We use
// the INTERIOR cut points (nb-1 of them) and count how many the value exceeds.
// This is the single place to tune if the server's edge convention differs — it
// is intentionally isolated and side-effect free.
function binIndex(value: number, edges: number[], nb: number): number {
    if (nb <= 1) return 0;
    let cuts = edges;
    if (cuts.length === nb + 1) {
        cuts = cuts.slice(1, nb);          // boundaries incl. min & max -> drop both ends
    } else if (cuts.length > nb - 1) {
        cuts = cuts.slice(0, nb - 1);      // be defensive about extra trailing edges
    }
    let bin = 0;
    while (bin < cuts.length && value > cuts[bin]) bin++;
    return Math.max(0, Math.min(nb - 1, bin));
}

// Build the input's constant set C(x) under a given class's feat_conv.
function embed(features: number[], edges: number[][], featConst: number[][]): Set<number> {
    const constants = new Set<number>();
    const nFeatures = Math.min(features.length, featConst.length, edges.length);
    for (let f = 0; f < nFeatures; f++) {
        const bins = featConst[f];
        if (!bins || bins.length === 0) continue;
        const bin = binIndex(features[f], edges[f] ?? [], bins.length);
        const constId = bins[bin];
        if (constId !== undefined && constId !== null) constants.add(constId);
    }
    return constants;
}

// True when every constant of the atom is present in the input (atom ≤ input).
function atomMatched(atom: Set<number>, inputConstants: Set<number>): boolean {
    for (const c of atom) {
        if (!inputConstants.has(c)) return false;
    }
    return true;
}

// Run AML inference on one feature vector. Returns per-class matched/missing atom
// breakdowns and picks the class with the highest matched fraction (fewest misses).
export function predictAml(model: AmlModel, features: number[]): AmlPrediction | null {
    if (!model || model.classes.length === 0) return null;

    const perClass: AmlClassResult[] = model.classes.map(cls => {
        const inputConstants = embed(features, model.edges, cls.featConst);
        let matchedCount = 0;
        const atoms: AmlAtomResult[] = cls.atoms.map(atom => {
            const matched = atomMatched(atom, inputConstants);
            if (matched) matchedCount++;
            return { ucs: Array.from(atom), matched };
        });
        const total = cls.atoms.length;
        return {
            label: cls.label,
            atomsTotal: total,
            atomsMatched: matchedCount,
            matchedFraction: total > 0 ? matchedCount / total : 0,
            atoms,
        };
    });

    // Winner = highest matched fraction; ties resolved by raw matched count.
    let best = perClass[0];
    for (const c of perClass) {
        if (c.matchedFraction > best.matchedFraction ||
            (c.matchedFraction === best.matchedFraction && c.atomsMatched > best.atomsMatched)) {
            best = c;
        }
    }

    return {
        predictedClass: best.label,
        confidence: best.matchedFraction,
        perClass,
    };
}

// Adapt a prediction into the `Explanation` the ExplanationSonifier consumes, so
// the chord is now driven by REAL AML atoms/misses instead of a confidence stub.
export function toExplanation(prediction: AmlPrediction): Explanation {
    const cls = prediction.perClass.find(c => c.label === prediction.predictedClass);
    const atomsTotal = cls?.atomsTotal ?? 0;
    const atomsMatched = cls?.atomsMatched ?? 0;
    return {
        predictedClass: prediction.predictedClass,
        confidence: prediction.confidence,
        atomsTotal,
        atomsMissing: atomsTotal - atomsMatched,
    };
}
