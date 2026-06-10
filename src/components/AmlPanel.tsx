// AmlPanel — all UI for the Algebraic Machine Learning (AML) path:
//   1. Connection badge   — is the AML Docker server (:5001) reachable?
//   2. Training status    — idle / training / ✓ trained / ❌ error
//   3. Prediction result  — which gesture AML thinks you drew + confidence
//   4. Atom ladder        — the EXPLANATION: which atoms matched vs missed,
//                           plus a "fullness" bar and a button to HEAR it
//                           (the explanation sonified as a chord).
//
// AML prediction + explanation are computed entirely client-side from the cached
// `/train` payload (see AmlModel.tsx) — no `/predict` endpoint is required.

import React, { useEffect, useState } from "react";
import { pingAmlServer, trainAml } from "../utils/AmlClient";
import {
    AmlModel,
    AmlPrediction,
    parseAmlPayload,
    predictAml,
    toExplanation,
} from "../utils/AmlModel";
import { ExplanationSonifier } from "../utils/ExplanationSonifier";

type ServerStatus = "unknown" | "checking" | "connected" | "unreachable";

interface AmlPanelProps {
    // Snapshot of the recorded samples as the AML train payload.
    getTrainingData: () => { x: number[][]; y: string[] };
    // Open the shared "draw now" capture window and resolve the feature vector
    // (or null if the user didn't draw enough). Reused from GestureComponent.
    captureStroke: () => Promise<number[] | null>;
    // Lazily-created shared sonifier, so the chord shares one AudioContext.
    getSonifier: () => ExplanationSonifier;
    // Whether explanation sonification is currently enabled.
    sonifyOn: boolean;
    // Play whatever sound the user mapped to a recognised label (optional).
    onPredictedLabel?: (label: string) => void;
}

const STATUS_BADGE: Record<ServerStatus, { dot: string; text: string; color: string }> = {
    unknown: { dot: "⚪", text: "AML server: unknown", color: "#777" },
    checking: { dot: "🟡", text: "AML server: checking…", color: "#b8860b" },
    connected: { dot: "🟢", text: "AML server: connected (:5001)", color: "#2e7d32" },
    unreachable: { dot: "🔴", text: "AML server: unreachable (:5001)", color: "#c62828" },
};

const AmlPanel = (props: AmlPanelProps) => {
    const { getTrainingData, captureStroke, getSonifier, sonifyOn, onPredictedLabel } = props;

    const [serverStatus, setServerStatus] = useState<ServerStatus>("unknown");
    const [nIter, setNIter] = useState<number>(5);
    const [isTraining, setIsTraining] = useState<boolean>(false);
    const [trainStatus, setTrainStatus] = useState<string>(
        "No AML model yet — record at least 2 labels, then Train (AML).",
    );
    const [model, setModel] = useState<AmlModel | null>(null);
    const [isCapturing, setIsCapturing] = useState<boolean>(false);
    const [prediction, setPrediction] = useState<AmlPrediction | null>(null);

    // Probe the server once on mount so the badge is meaningful before any action.
    useEffect(() => {
        let cancelled = false;
        setServerStatus("checking");
        pingAmlServer().then(ok => {
            if (!cancelled) setServerStatus(ok ? "connected" : "unreachable");
        });
        return () => { cancelled = true; };
    }, []);

    const refreshStatus = async () => {
        setServerStatus("checking");
        const ok = await pingAmlServer();
        setServerStatus(ok ? "connected" : "unreachable");
    };

    const handleTrain = async () => {
        const { x, y } = getTrainingData();
        if (x.length === 0) {
            setTrainStatus("⚠️ No recorded gestures yet — record some samples first.");
            return;
        }
        const labels = Array.from(new Set(y));
        if (labels.length < 2) {
            setTrainStatus(`⚠️ Need at least 2 different labels to train (you have: ${labels.join(", ") || "none"}).`);
            return;
        }

        setIsTraining(true);
        setModel(null);
        setPrediction(null);
        setTrainStatus(`Training (AML) on ${x.length} samples for ${nIter} iterations… this can take a while.`);

        const result = await trainAml(x, y, nIter);
        if (!result.ok) {
            if (result.kind === "unreachable") setServerStatus("unreachable");
            setTrainStatus(`❌ AML training failed (${result.kind}): ${result.message}`);
            setIsTraining(false);
            return;
        }

        setServerStatus("connected");
        const parsed = parseAmlPayload(result.payload);
        if (!parsed) {
            setTrainStatus("❌ AML returned a model we couldn't read (no usable atoms).");
            setIsTraining(false);
            return;
        }

        setModel(parsed);
        const summary = parsed.classes
            .map(c => `${c.label} (${c.atoms.length} atoms)`)
            .join(", ");
        setTrainStatus(`✓ AML trained on: ${summary}. Now Predict to see which atoms match.`);
        setIsTraining(false);
    };

    const handlePredict = async () => {
        if (!model) {
            setTrainStatus("⚠️ Train an AML model first.");
            return;
        }
        setIsCapturing(true);
        setPrediction(null);
        const features = await captureStroke();
        setIsCapturing(false);
        if (!features) {
            setTrainStatus("Didn't see enough movement — draw a clear shape and try again.");
            return;
        }

        const pred = predictAml(model, features);
        if (!pred) return;
        setPrediction(pred);

        if (sonifyOn) getSonifier().play(toExplanation(pred));
        if (onPredictedLabel) onPredictedLabel(pred.predictedClass);
    };

    const badge = STATUS_BADGE[serverStatus];
    const predictedClass = prediction
        ? prediction.perClass.find(c => c.label === prediction.predictedClass)
        : undefined;

    return (
        <div style={{ marginTop: "16px", padding: "12px", border: "2px solid #ff8c00", borderRadius: "8px", position: "relative", zIndex: 2 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
                <h3 style={{ margin: 0 }}>Algebraic Machine Learning (AML)</h3>
                <button
                    onClick={refreshStatus}
                    title="Re-check the AML server connection"
                    style={{
                        display: "flex", alignItems: "center", gap: "6px",
                        background: "transparent", border: `1px solid ${badge.color}`,
                        color: badge.color, borderRadius: "999px", padding: "4px 10px",
                        fontWeight: "bold", cursor: "pointer",
                    }}
                >
                    {badge.dot} {badge.text} ⟳
                </button>
            </div>

            {/* Training controls */}
            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    Iterations:
                    <input
                        type="number"
                        min={1}
                        max={100}
                        value={nIter}
                        onChange={e => setNIter(Math.max(1, Number(e.target.value) || 1))}
                        style={{ width: "70px", padding: "6px", borderRadius: "4px" }}
                    />
                </label>
                <button
                    onClick={handleTrain}
                    disabled={isTraining}
                    style={{
                        backgroundColor: isTraining ? "#999" : "darkorange",
                        color: "white", padding: "10px 14px", borderRadius: "5px",
                        border: "none", fontWeight: "bold",
                    }}
                >
                    {isTraining ? "Training (AML)…" : "Train Model (AML)"}
                </button>
                <button
                    onClick={handlePredict}
                    disabled={!model || isCapturing || isTraining}
                    style={{
                        backgroundColor: isCapturing ? "#00C8FF" : (model ? "#b8860b" : "#999"),
                        color: "white", padding: "10px 14px", borderRadius: "5px",
                        border: "none", fontWeight: "bold",
                    }}
                >
                    {isCapturing ? "✏️ Draw your shape!" : "▶ Predict (AML)"}
                </button>
            </div>

            <p style={{
                fontWeight: "bold", marginTop: "10px",
                color: isTraining ? "#b8860b" : (model ? "#2e7d32" : "#555"),
            }}>
                {trainStatus}
            </p>

            {/* Prediction result + atom-ladder explanation */}
            {prediction && predictedClass && (
                <div style={{ marginTop: "8px", padding: "10px", background: "#fff7ec", borderRadius: "6px" }}>
                    <div style={{ fontSize: "1.2em", fontWeight: "bold" }}>
                        Predicted (AML): {prediction.predictedClass} — {(prediction.confidence * 100).toFixed(0)}% atoms matched
                    </div>
                    <div style={{ color: "#555", margin: "4px 0 10px" }}>
                        {predictedClass.atomsMatched}/{predictedClass.atomsTotal} atoms of “{prediction.predictedClass}” are present.
                        {predictedClass.atomsTotal - predictedClass.atomsMatched > 0 && (
                            <> {predictedClass.atomsTotal - predictedClass.atomsMatched} are missing (the “why-not”).</>
                        )}
                    </div>

                    <AtomLadder
                        matched={predictedClass.atomsMatched}
                        total={predictedClass.atomsTotal}
                    />

                    <button
                        onClick={() => getSonifier().play(toExplanation(prediction))}
                        style={{
                            marginTop: "10px", backgroundColor: "#3DDC84", color: "white",
                            padding: "8px 12px", borderRadius: "5px", border: "none", fontWeight: "bold",
                        }}
                        title="Hear the explanation: matched atoms fill the chord; misses mute notes; confidence sets brightness."
                    >
                        🔊 Hear this explanation
                    </button>

                    {/* Per-class breakdown — how close every class was. */}
                    <div style={{ marginTop: "12px" }}>
                        <h4 style={{ margin: "0 0 6px" }}>How every class scored</h4>
                        {prediction.perClass
                            .slice()
                            .sort((a, b) => b.matchedFraction - a.matchedFraction)
                            .map(c => (
                                <div key={c.label} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                                    <span style={{ width: "90px", fontWeight: c.label === prediction.predictedClass ? "bold" : "normal" }}>
                                        {c.label}
                                    </span>
                                    <div style={{ flex: 1, height: "12px", background: "#e0e0e0", borderRadius: "6px", overflow: "hidden" }}>
                                        <div style={{
                                            width: `${Math.round(c.matchedFraction * 100)}%`, height: "100%",
                                            background: c.label === prediction.predictedClass ? "#2e7d32" : "#b8860b",
                                        }} />
                                    </div>
                                    <span style={{ width: "70px", textAlign: "right", color: "#555" }}>
                                        {c.atomsMatched}/{c.atomsTotal}
                                    </span>
                                </div>
                            ))}
                    </div>
                </div>
            )}
        </div>
    );
};

// A compact "atom ladder": one cell per atom of the predicted class. Lit cells are
// matched atoms; greyed cells are misses. The chord you hear is exactly this — the
// thinner the lit portion, the hollower the sound. Capped so a large model doesn't
// flood the DOM (the count above is always exact).
const ATOM_LADDER_CAP = 80;

const AtomLadder = (props: { matched: number; total: number }) => {
    const { matched, total } = props;
    const shown = Math.min(total, ATOM_LADDER_CAP);
    // Keep matched cells visually first so the "fullness" reads left-to-right.
    const matchedShown = Math.round((matched / Math.max(1, total)) * shown);
    const cells = Array.from({ length: shown }, (_, i) => i < matchedShown);

    return (
        <div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "3px" }}>
                {cells.map((lit, i) => (
                    <span
                        key={i}
                        title={lit ? "atom matched" : "atom missing (why-not)"}
                        style={{
                            width: "12px", height: "12px", borderRadius: "2px",
                            background: lit ? "#2e7d32" : "#d6d6d6",
                            border: "1px solid rgba(0,0,0,0.1)",
                        }}
                    />
                ))}
            </div>
            {total > ATOM_LADDER_CAP && (
                <div style={{ fontSize: "0.8em", color: "#888", marginTop: "4px" }}>
                    Showing {shown} of {total} atoms (scaled).
                </div>
            )}
        </div>
    );
};

export default AmlPanel;
