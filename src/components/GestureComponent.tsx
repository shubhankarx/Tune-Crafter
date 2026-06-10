import React, { useState, useEffect, useRef } from "react";
import ReactGA from 'react-ga4';
import RegionsPlugin from 'wavesurfer.js/src/plugin/regions';
import 'bootstrap/dist/css/bootstrap.css';

import { GestureRecognizer, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import WaveSurfer from "wavesurfer.js";
import * as tf from '@tensorflow/tfjs';
import { GestureModel } from "../models/GestureModel";
import { AudioManager } from "../AudioManager";
import VolumeProgressBar from "./VolumeProgressBar";
import { ExplanationSonifier, stubExplanationFromConfidence } from "../utils/ExplanationSonifier";
import AmlPanel from "./AmlPanel";

// Stroke feature layout. We track the index-finger tip's 2D path, then:
//   1. resample it to RESAMPLE_N points  -> speed / stroke-length invariant
//   2. centre + scale it                 -> position / size invariant
//   3. append per-point turning angles   -> corners (square) vs smooth (circle)
// Raw coordinates alone can't tell a square from a circle drawn in the same spot;
// the turning angles are what make those shapes separable.
const RESAMPLE_N = 16;                 // points the stroke is resampled to
const LIVE_WINDOW_FRAMES = 48;         // raw frames kept for live recognition (~1.6s of motion)
const FEATURE_LENGTH = RESAMPLE_N * 2 + (RESAMPLE_N - 2); // x,y per point + turning angles

// Live-recognition gating: only emit a label once the model is confident AND the
// same label has held for a few consecutive frames. Prevents per-frame argmax
// flicker from spamming mapped sounds.
const CONFIDENCE_THRESHOLD = 0.8;
const STABILITY_FRAMES = 5;

export interface Coordinates {
    x: number;
    y: number;
}

interface RecordedGesture {
    x: number[];
    y: string;
    duration: number;
}

interface GestureComponentProps {
    video: HTMLVideoElement | null,
    waveform: WaveSurfer | null,
    soundManager: AudioManager
}

const GestureComponent = (props: GestureComponentProps) => {
    const { video, waveform, soundManager } = props;

    // ---- Per-frame scratch state (only ever used inside the single rAF loop) ----
    let canvasElement: any | null = null;
    let canvasCtx: any | null = null;
    let results: any = undefined;
    const videoHeight = "100vh";
    const videoWidth = "auto";
    let volumeTimer: any = null;

    // GestureModel must be stable across renders (it holds FSM + region state),
    // so it lives in a ref instead of being recreated on every render.
    const modelRef = useRef<GestureModel | null>(null);
    if (!modelRef.current) {
        modelRef.current = new GestureModel(soundManager);
    }
    const model = modelRef.current;

    // ---- React state (drives the UI) ----
    const [volume, setVolume] = useState<number>(50);
    const [isVolumeVisible, setIsVolumeVisible] = useState<boolean>(false);
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [predictedConfidence, setPredictedConfidence] = useState<number>(0); // 0..1 for the manual prediction (colours the result text)
    const [isCapturingPredict, setIsCapturingPredict] = useState<boolean>(false); // 2.5s "draw now" window for the Predict button
    const [predictStatus, setPredictStatus] = useState<string>("");            // result/hint text for the Predict button
    const [recordedGestures, setRecordedGestures] = useState<RecordedGesture[]>([]);
    const [isModelTrained, setIsModelTrained] = useState<boolean>(false);
    const [isTraining, setIsTraining] = useState<boolean>(false);              // disables buttons while a fit() runs
    const [trainingStatus, setTrainingStatus] = useState<string>("No model yet — record at least 2 labels, then Train.");
    const [isCustomMode, setIsCustomMode] = useState<boolean>(false);          // ON = test custom model only, automatic detection off
    const [sonifyOn, setSonifyOn] = useState<boolean>(false);                  // Direction 1: sonify the model's explanation on each recognised gesture
    const [gestureLabel, setGestureLabel] = useState<string>("");
    const [gestureSoundMap, setGestureSoundMap] = useState<{ [key: string]: string }>({});
    const [mapGesture, setMapGesture] = useState<string>("");
    const [mapSound, setMapSound] = useState<string>("");

    // ---- Refs (read by the long-lived requestAnimationFrame loop / async handlers,
    //      which capture their first-render closure and would otherwise see stale state) ----
    const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);
    const rafStartedRef = useRef<boolean>(false);
    const isRecordingRef = useRef<boolean>(false);
    const recordingStartTimeRef = useRef<number | null>(null);
    const liveBufferRef = useRef<number[][]>([]);          // rolling window of recent frames, always updated
    const recordingBufferRef = useRef<number[][]>([]);     // unbounded buffer, captures the full gesture while recording
    const recordedGesturesRef = useRef<RecordedGesture[]>([]);
    const classifierRef = useRef<tf.LayersModel | null>(null);   // trained local model (single source of truth)
    const labelOrderRef = useRef<string[]>([]);            // label order used at train time => reused at predict time
    const gestureLabelRef = useRef<string>("");
    const gestureSoundMapRef = useRef<{ [key: string]: string }>({});
    const lastRecognizedRef = useRef<string>("");          // edge-trigger guard so live recognition doesn't spam audio
    const isPredictingRef = useRef<boolean>(false);        // skip a frame if the previous predict hasn't resolved
    const candidateLabelRef = useRef<string>("");          // label currently building up stability
    const candidateCountRef = useRef<number>(0);           // consecutive frames the candidate label has held
    const rafIdRef = useRef<number | null>(null);          // handle to the pending animation frame, for cancellation
    const loopStoppedRef = useRef<boolean>(false);         // set on unmount so the rAF loop stops rescheduling itself
    const currentGestureElRef = useRef<HTMLOutputElement | null>(null); // cached #current_gesture node (avoids per-frame DOM lookups)
    const isCustomModeRef = useRef<boolean>(false);        // mirror of isCustomMode for the rAF loop
    const trailRef = useRef<{ x: number; y: number }[]>([]); // raw index-tip image coords for drawing the on-screen trail
    const displayLabelRef = useRef<string>("");            // latest predicted label, for the on-canvas overlay
    const displayConfRef = useRef<number>(0);              // latest confidence 0..1, for the on-canvas overlay
    const predictCapturingRef = useRef<boolean>(false);    // true during the Predict button's "draw now" window
    const predictBufferRef = useRef<number[][]>([]);       // clean stroke captured for one-shot prediction
    const sonifyOnRef = useRef<boolean>(false);            // mirror of sonifyOn for the rAF loop
    const sonifierRef = useRef<ExplanationSonifier | null>(null); // turns an explanation into a chord

    // Resolve (and cache) the on-screen gesture label element. The rAF loop and
    // several handlers write to it every frame, so we look it up once.
    const getCurrentGestureEl = (): HTMLOutputElement | null => {
        if (!currentGestureElRef.current) {
            currentGestureElRef.current = document.getElementById('current_gesture') as HTMLOutputElement | null;
        }
        return currentGestureElRef.current;
    };

    const setCurrentGestureText = (text: string) => {
        const el = getCurrentGestureEl();
        if (el) el.innerText = text;
    };

    const MAX_RECORDING_DURATION = 2000; // 2 seconds

    // Keep refs in sync with state so the rAF loop always reads fresh values.
    useEffect(() => { recordedGesturesRef.current = recordedGestures; }, [recordedGestures]);
    useEffect(() => { gestureLabelRef.current = gestureLabel; }, [gestureLabel]);
    useEffect(() => { gestureSoundMapRef.current = gestureSoundMap; }, [gestureSoundMap]);
    useEffect(() => { isRecordingRef.current = isRecording; }, [isRecording]);
    useEffect(() => { isCustomModeRef.current = isCustomMode; }, [isCustomMode]);
    useEffect(() => { sonifyOnRef.current = sonifyOn; }, [sonifyOn]);

    // ---------------------------------------------------------------------------
    // Feature extraction
    // ---------------------------------------------------------------------------

    // Make all landmark coordinates relative to the wrist (landmarks[0]) so the
    // gesture is position-invariant.
    const normalizeCoordinates = (landmarks: any) => {
        const basePoint = landmarks[0];
        return landmarks.map((point: any) => ({
            x: (point.x - basePoint.x),
            y: (point.y - basePoint.y),
            z: point.z ? (point.z - basePoint.z) : 0.0
        }));
    };

    // Reduce one frame of (normalized) landmarks to the keypoint we track: the index-finger tip.
    const extractGestureFeatures = (landmarks: any): number[] => {
        if (!landmarks || landmarks.length === 0) return [0.0, 0.0, 0.0];
        const keypoint = landmarks[8]; // index-finger tip
        return [keypoint.x, keypoint.y, keypoint.z || 0.0];
    };

    // ---- Stroke feature engineering -----------------------------------------

    // Total arc length of a 2D path.
    const pathLength2d = (pts: number[][]): number => {
        let len = 0;
        for (let i = 1; i < pts.length; i++) {
            len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
        }
        return len;
    };

    // Resample the path to n points spaced equally along its arc length ($1-style),
    // so two strokes of the same shape but different speed/length line up.
    const resampleStroke = (pts: number[][], n: number): number[][] => {
        const total = pathLength2d(pts);
        if (total === 0) return Array.from({ length: n }, () => [pts[0][0], pts[0][1]]);
        const interval = total / (n - 1);
        const out: number[][] = [[pts[0][0], pts[0][1]]];
        let acc = 0;
        let prev = [pts[0][0], pts[0][1]];
        for (let i = 1; i < pts.length;) {
            const seg = Math.hypot(pts[i][0] - prev[0], pts[i][1] - prev[1]);
            if (acc + seg >= interval && seg > 0) {
                const t = (interval - acc) / seg;
                const nx = prev[0] + t * (pts[i][0] - prev[0]);
                const ny = prev[1] + t * (pts[i][1] - prev[1]);
                out.push([nx, ny]);
                prev = [nx, ny];
                acc = 0;
            } else {
                acc += seg;
                prev = [pts[i][0], pts[i][1]];
                i++;
            }
        }
        while (out.length < n) out.push([pts[pts.length - 1][0], pts[pts.length - 1][1]]);
        return out.slice(0, n);
    };

    // Centre the stroke on its centroid and scale so its largest extent is 1.
    const normalizeStroke = (pts: number[][]): number[][] => {
        let cx = 0, cy = 0;
        for (const p of pts) { cx += p[0]; cy += p[1]; }
        cx /= pts.length; cy /= pts.length;
        let maxExtent = 0;
        for (const p of pts) {
            maxExtent = Math.max(maxExtent, Math.abs(p[0] - cx), Math.abs(p[1] - cy));
        }
        const scale = maxExtent > 1e-6 ? maxExtent : 1;
        return pts.map(p => [(p[0] - cx) / scale, (p[1] - cy) / scale]);
    };

    // Signed turning angle (radians) at each interior point: ~0 along a smooth
    // curve, large at a sharp corner. The square-vs-circle discriminator.
    const turningAngles = (pts: number[][]): number[] => {
        const angles: number[] = [];
        for (let i = 1; i < pts.length - 1; i++) {
            const ax = pts[i][0] - pts[i - 1][0];
            const ay = pts[i][1] - pts[i - 1][1];
            const bx = pts[i + 1][0] - pts[i][0];
            const by = pts[i + 1][1] - pts[i][1];
            const cross = ax * by - ay * bx;
            const dot = ax * bx + ay * by;
            angles.push(Math.atan2(cross, dot));
        }
        return angles;
    };

    // Turn a raw frame buffer (each frame = [x, y, z] of the index tip) into the
    // fixed-length stroke feature vector. Always returns FEATURE_LENGTH numbers so
    // training and prediction tensors stay the same shape.
    const extractFeatures = (buffer: number[][]): number[] => {
        const out = new Array(FEATURE_LENGTH).fill(0.0);
        const raw = buffer.filter(p => p && p.length >= 2).map(p => [p[0], p[1]]);
        if (raw.length < 2 || pathLength2d(raw) < 1e-4) return out; // no real stroke yet
        const pts = normalizeStroke(resampleStroke(raw, RESAMPLE_N));
        for (let i = 0; i < RESAMPLE_N; i++) {
            out[2 * i] = pts[i][0];
            out[2 * i + 1] = pts[i][1];
        }
        const angles = turningAngles(pts);
        for (let i = 0; i < angles.length; i++) {
            out[2 * RESAMPLE_N + i] = angles[i];
        }
        return out;
    };

    // Push the latest frame onto the rolling live buffer (kept at SLIDING_WINDOW_SIZE).
    // Runs every frame a hand is visible, independent of recording, so prediction
    // always has fresh data to work with.
    const pushLiveFrame = (landmarks: any) => {
        const normalized = normalizeCoordinates(landmarks);
        const point = extractGestureFeatures(normalized);
        const buf = liveBufferRef.current;
        buf.push(point);
        if (buf.length > LIVE_WINDOW_FRAMES) {
            buf.shift();
        }
        // During recording, accumulate every frame (unbounded) so the whole gesture
        // stroke is captured, not just the last LIVE_WINDOW_FRAMES frames.
        if (isRecordingRef.current) {
            recordingBufferRef.current.push(point);
        }
        // During a Predict-button capture window, collect the clean stroke separately.
        if (predictCapturingRef.current) {
            predictBufferRef.current.push(point);
        }
    };

    // ---------------------------------------------------------------------------
    // Recording
    // ---------------------------------------------------------------------------

    const handleLabelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setGestureLabel(event.target.value);
    };

    const handleRecordButtonClick = () => {
        if (!isRecordingRef.current) {
            startRecording();
        } else {
            stopRecording();
        }
    };

    const startRecording = () => {
        if (!gestureLabelRef.current.trim()) {
            console.warn("Tip: enter a gesture label before recording so the sample is labelled.");
        }
        setIsRecording(true);
        isRecordingRef.current = true;
        recordingStartTimeRef.current = Date.now();
        liveBufferRef.current = []; // start a fresh window for this sample
        recordingBufferRef.current = []; // start a fresh full-duration capture
        console.log("Recording started for label:", gestureLabelRef.current || "(unlabeled)");

        // Automatically stop after the max duration.
        setTimeout(() => {
            if (isRecordingRef.current) {
                console.log(`Auto-stop after ${MAX_RECORDING_DURATION}ms`);
                stopRecording();
            }
        }, MAX_RECORDING_DURATION);
    };

    const stopRecording = () => {
        if (!isRecordingRef.current) return;
        setIsRecording(false);
        isRecordingRef.current = false;
        const duration = Date.now() - (recordingStartTimeRef.current || Date.now());

        const buffer = recordingBufferRef.current;
        if (buffer.length > 0) {
            // Convert the full captured stroke into the fixed-length feature vector
            // (extractFeatures resamples internally, so no manual downsampling needed).
            const features = extractFeatures(buffer);
            const gesture: RecordedGesture = {
                x: features,
                y: gestureLabelRef.current.trim() || "unlabeled",
                duration
            };
            setRecordedGestures(prev => [...prev, gesture]);
            console.log(`Recorded gesture (${buffer.length} frames -> window):`, gesture);
        } else {
            console.log("No frames captured during recording.");
        }
        recordingBufferRef.current = [];
        recordingStartTimeRef.current = null;
    };

    // ---------------------------------------------------------------------------
    // Local TensorFlow.js training & prediction
    // ---------------------------------------------------------------------------

    // Count how many samples we have per label — surfaced in the UI so the user
    // can see whether the data is balanced enough to train well.
    const sampleCounts = (gestures: RecordedGesture[]): Record<string, number> => {
        const counts: Record<string, number> = {};
        for (const g of gestures) counts[g.y] = (counts[g.y] || 0) + 1;
        return counts;
    };

    const trainModel = async () => {
        const gestures = recordedGesturesRef.current;
        if (gestures.length === 0) {
            setTrainingStatus("⚠️ No recorded gestures yet — record some samples first.");
            return;
        }

        // Dynamic labels: derive a stable ordered list of the unique labels and
        // one-hot encode against it. The same ordering is reused at predict time.
        const labelOrder = Array.from(new Set(gestures.map(g => g.y)));
        if (labelOrder.length < 2) {
            setTrainingStatus(`⚠️ Need at least 2 different labels to train (you have: ${labelOrder.join(", ") || "none"}).`);
            return;
        }

        const EPOCHS = 50;
        setIsTraining(true);
        setTrainingStatus("Training… starting");

        const inputs = gestures.map(g => g.x); // each row already FEATURE_LENGTH long
        const labels = gestures.map(g => {
            const vec = new Array(labelOrder.length).fill(0);
            vec[labelOrder.indexOf(g.y)] = 1;
            return vec;
        });

        const xs = tf.tensor2d(inputs);
        const ys = tf.tensor2d(labels);
        console.log("Training tensor shapes:", xs.shape, ys.shape, "labels:", labelOrder);

        const net = tf.sequential();
        net.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [inputs[0].length] }));
        net.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        net.add(tf.layers.dense({ units: labelOrder.length, activation: 'softmax' }));
        net.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        try {
            await net.fit(xs, ys, {
                epochs: EPOCHS,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const acc = logs?.acc !== undefined ? (logs.acc * 100).toFixed(0) : "?";
                        setTrainingStatus(`Training… epoch ${epoch + 1}/${EPOCHS} — accuracy ${acc}%`);
                        console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss?.toFixed(4)}, accuracy = ${logs?.acc?.toFixed(4)}`);
                    }
                }
            });
            classifierRef.current?.dispose(); // free the previous model before replacing it
            classifierRef.current = net;
            labelOrderRef.current = labelOrder;
            setIsModelTrained(true);
            const counts = sampleCounts(gestures);
            const summary = labelOrder.map(l => `${l} (${counts[l]})`).join(", ");
            setTrainingStatus(`✓ Trained on: ${summary}. Make a gesture and hold it to test.`);
            console.log("Model trained successfully. Labels:", labelOrder);
        } catch (error) {
            setTrainingStatus("❌ Training failed — see console for details.");
            console.error("Error during training:", error);
        } finally {
            setIsTraining(false);
            xs.dispose();
            ys.dispose();
        }
    };

    // Maps the model's output vector back to a label using the training-time order.
    const labelFromPrediction = (probs: number[]): string => {
        const idx = probs.indexOf(Math.max(...probs));
        return labelOrderRef.current[idx] ?? "unknown";
    };

    // Single source of truth for running the local model on one feature window.
    // Returns the probability vector, or null if no model is trained. Always
    // disposes the intermediate tensors.
    const runPredict = async (features: number[]): Promise<number[] | null> => {
        const net = classifierRef.current;
        if (!net) return null;
        const input = tf.tensor2d([features]);
        const out = net.predict(input) as tf.Tensor;
        try {
            const arr = (await out.array()) as number[][];
            return arr[0];
        } finally {
            input.dispose();
            out.dispose();
        }
    };

    const PREDICT_CAPTURE_MS = 2500; // how long the "draw now" window lasts

    // Lazily create (and share) the one explanation sonifier so every code path
    // reuses a single AudioContext.
    const ensureSonifier = (): ExplanationSonifier => {
        if (!sonifierRef.current) sonifierRef.current = new ExplanationSonifier();
        return sonifierRef.current;
    };

    // Open a short "draw now" window so the user can draw ONE clean shape, then
    // resolve its feature vector — avoids classifying the stray motion of reaching
    // for the button. Shared by the local-NN and AML predict flows. Resolves null
    // if a capture is already running or the user didn't draw enough.
    const captureStroke = (): Promise<number[] | null> => {
        return new Promise(resolve => {
            if (predictCapturingRef.current) {
                resolve(null);
                return;
            }
            predictBufferRef.current = [];
            predictCapturingRef.current = true;
            setIsCapturingPredict(true);
            setTimeout(() => {
                predictCapturingRef.current = false;
                setIsCapturingPredict(false);
                const buf = predictBufferRef.current;
                resolve(buf.length < RESAMPLE_N ? null : extractFeatures(buf));
            }, PREDICT_CAPTURE_MS);
        });
    };

    // Snapshot the recorded samples as the AML train payload.
    const getAmlTrainingData = (): { x: number[][]; y: string[] } => {
        const gestures = recordedGesturesRef.current;
        return { x: gestures.map(g => g.x), y: gestures.map(g => g.y) };
    };

    // Deliberate one-shot prediction with the local TensorFlow.js model.
    const predictGesture = async () => {
        if (!classifierRef.current) {
            setPredictStatus("⚠️ Train a model first.");
            return;
        }
        setPredictStatus("✏️ Draw your shape now…");
        const features = await captureStroke();
        if (!features) {
            setPredictStatus("Didn't see enough movement — draw a clear shape and try again.");
            return;
        }
        const probs = await runPredict(features);
        if (!probs) return;
        const label = labelFromPrediction(probs);
        const conf = Math.max(...probs);
        console.log("Predict (captured stroke):", probs, "=>", label);
        setPredictedConfidence(conf);
        setPredictStatus(`Predicted: ${label} — ${(conf * 100).toFixed(0)}%`);
        // Sonify the explanation for THIS one-shot prediction, ungated — so a
        // deliberately sloppy gesture audibly produces a thin, muffled chord
        // (the full uncertainty range, which the live >=0.8 gate hides).
        if (sonifyOnRef.current) {
            ensureSonifier().play(stubExplanationFromConfidence(label, conf));
        }
    };

    // Continuous recognition called from the rAF loop. Emits (and plays a mapped
    // sound) only when the model is confident and the same label has held steady
    // for a few frames, then once per change — avoiding per-frame audio spam.
    const recognizeGesture = async () => {
        if (!classifierRef.current || isPredictingRef.current) return;
        // Wait for enough points to form a meaningful stroke before recognizing.
        if (liveBufferRef.current.length < RESAMPLE_N) return;
        isPredictingRef.current = true; // block overlapping per-frame predicts
        try {
            const probs = await runPredict(extractFeatures(liveBufferRef.current));
            if (!probs) return;

            const confidence = Math.max(...probs);
            const label = labelFromPrediction(probs);

            // Live read-out is drawn on the canvas overlay (via refs) rather than React
            // state, so we don't trigger a re-render on every animation frame.
            displayLabelRef.current = label;
            displayConfRef.current = confidence;

            // Below the confidence floor: show it, but don't build stability or fire sound.
            if (confidence < CONFIDENCE_THRESHOLD) {
                candidateLabelRef.current = "";
                candidateCountRef.current = 0;
                return;
            }

            // Require the same label for STABILITY_FRAMES consecutive frames.
            if (label === candidateLabelRef.current) {
                candidateCountRef.current += 1;
            } else {
                candidateLabelRef.current = label;
                candidateCountRef.current = 1;
            }

            // Edge-trigger: only PLAY a sound when a newly-stable label differs from
            // the last one we played — keeps the display live but the audio calm.
            if (candidateCountRef.current >= STABILITY_FRAMES && label !== lastRecognizedRef.current) {
                lastRecognizedRef.current = label;
                playMappedSound(label);
                // Direction 1: sonify the model's explanation as a chord (missing
                // atoms mute notes; confidence sets brightness). Stubbed from
                // confidence today; swap in real AML atoms/misses when available.
                if (sonifyOnRef.current) {
                    ensureSonifier().play(stubExplanationFromConfidence(label, confidence));
                }
            }
        } finally {
            isPredictingRef.current = false;
        }
    };

    // AML training, prediction and explanation now live in <AmlPanel>, computed
    // client-side from the cached /train payload (see src/utils/AmlModel.tsx).

    // ---------------------------------------------------------------------------
    // Gesture -> sound mapping
    // ---------------------------------------------------------------------------

    const playMappedSound = (detectedGesture: string) => {
        const soundName = gestureSoundMapRef.current[detectedGesture];
        if (soundName && soundManager) {
            soundManager.playSound(soundName);
            setCurrentGestureText(`${detectedGesture} -> ${soundName}`);
        }
    };

    // ---------------------------------------------------------------------------
    // MediaPipe gesture recognizer + render loop
    // ---------------------------------------------------------------------------

    // Start the recognizer + render loop once, when video and waveform are both
    // ready. This effect must NOT have a cleanup: video/waveform arrive
    // asynchronously (null -> set), so a cleanup here would fire on every dependency
    // change and tear down the loop before it ever runs. Teardown lives in the
    // unmount-only effect below.
    useEffect(() => {
        if (video && waveform && !rafStartedRef.current) {
            loopStoppedRef.current = false;
            createGestureRecognizer().then(() => {
                if (!rafStartedRef.current) {
                    rafStartedRef.current = true;
                    rafIdRef.current = requestAnimationFrame(predictWebcam);
                }
            });
            setAudioObjects();
        }
    }, [video, waveform]);

    // Release the heavier resources only when the component actually unmounts.
    useEffect(() => {
        return () => {
            loopStoppedRef.current = true; // stop the rAF loop rescheduling itself
            if (rafIdRef.current != null) {
                cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
            }
            gestureRecognizerRef.current?.close();
            gestureRecognizerRef.current = null;
            classifierRef.current?.dispose();
            classifierRef.current = null;
        };
    }, []);

    const createGestureRecognizer = async () => {
        const recognizer = await loadModelWithRetry();
        if (recognizer) {
            gestureRecognizerRef.current = recognizer;
        } else {
            console.error("Gesture recognizer creation failed.");
        }

        if (!model.haveRegions()) {
            const regions = waveform?.addPlugin(RegionsPlugin.create({}));
            regions?.on('region-created', (region: any) => {
                if (region.loop) {
                    region.playLoop();
                }
            });
            regions?.on('region-out', (region: any) => {
                if (region.loop) {
                    region.play();
                }
            });
            regions?.on('region-removed', (_: any) => {
                waveform?.play();
            });
            model.setRegions(regions);
        }
    };

    async function loadModelWithRetry() {
        const maxRetries = 3;
        let currentRetry = 0;
        let recognizer;

        while (currentRetry < maxRetries) {
            try {
                const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm");
                recognizer = await GestureRecognizer.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
                    },
                    numHands: 2,
                    runningMode: "VIDEO"
                });
                break;
            } catch (error) {
                console.error(`Error on attempt #${currentRetry + 1}:`, error);
                currentRetry++;
                if (currentRetry >= maxRetries) {
                    console.error("Maximum retry attempts reached. Model loading failed.");
                }
            }
        }
        console.log("Model loading completed with status:", recognizer ? "Success" : "Failed");
        return recognizer;
    }

    const predictWebcam = async () => {
        if (loopStoppedRef.current) return; // component unmounted; stop the loop
        const recognizer = gestureRecognizerRef.current;
        if (!recognizer) {
            rafIdRef.current = requestAnimationFrame(predictWebcam);
            return;
        }

        setupCanvas();
        if (video && video.videoHeight > 0 && video.videoWidth > 0) {
            try {
                results = await recognizer.recognizeForVideo(video, Date.now());

                const hasHand = results && results.landmarks && results.landmarks.length > 0;
                if (hasHand) {
                    // Always feed the rolling buffer so prediction has fresh data.
                    pushLiveFrame(results.landmarks[0]);
                    // Record the raw index-tip position for the on-screen trail.
                    const tip = results.landmarks[0][8];
                    if (tip) {
                        const trail = trailRef.current;
                        trail.push({ x: tip.x, y: tip.y });
                        if (trail.length > LIVE_WINDOW_FRAMES) trail.shift();
                    }
                } else if (!isRecordingRef.current) {
                    // Hand left the frame: drop stale frames and reset recognition
                    // state so a re-appearing hand starts from a clean window and
                    // the next gesture can re-trigger its sound.
                    liveBufferRef.current = [];
                    trailRef.current = [];
                    candidateLabelRef.current = "";
                    candidateCountRef.current = 0;
                    lastRecognizedRef.current = "";
                    displayLabelRef.current = "";
                    displayConfRef.current = 0;
                }

                // Custom mode: run the trained model and skip the automatic
                // detection so the two systems don't fight over the display/audio.
                // Normal mode: run the built-in gesture actions (drums, play/pause…).
                if (isCustomModeRef.current) {
                    if (classifierRef.current && hasHand && !isRecordingRef.current) {
                        recognizeGesture();
                    }
                    drawHands();
                } else {
                    drawHands();
                    performAction();
                }
            } catch (error) {
                console.error("Error during gesture recognition:", error);
            }
        }
        rafIdRef.current = requestAnimationFrame(predictWebcam);
    };

    const setAudioObjects = () => {
        soundManager.loadAllSounds();
    };

    const setupCanvas = () => {
        if (canvasCtx == undefined) {
            canvasElement = document.getElementById("output_canvas") as HTMLCanvasElement;
            canvasCtx = canvasElement.getContext("2d");
            canvasElement.style.height = videoHeight;
            canvasElement.style.width = videoWidth;
        }
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    };

    const drawHands = () => {
        const drawingUtils = new DrawingUtils(canvasCtx);
        if (results && results.landmarks.length > 0) {
            for (const landmarks of results.landmarks) {
                drawingUtils.drawConnectors(
                    landmarks,
                    GestureRecognizer.HAND_CONNECTIONS,
                    { color: "#FFFFFF", lineWidth: 5 }
                );
                drawingUtils.drawLandmarks(landmarks, { color: "#B01EB0", lineWidth: 2 });
            }
        }
        if (isCustomModeRef.current || predictCapturingRef.current) {
            drawCustomOverlay();
        }
        canvasCtx.restore();
    };

    // Custom-mode visuals: the index-finger trail (what the model tracks over the
    // sliding window) plus a big label + confidence read-out on the video.
    const drawCustomOverlay = () => {
        const w = canvasElement?.width ?? 1280;
        const h = canvasElement?.height ?? 720;

        // Trail: connect the recent index-tip positions, fading from old to new.
        const trail = trailRef.current;
        if (trail.length > 1) {
            for (let i = 1; i < trail.length; i++) {
                const a = trail[i - 1];
                const b = trail[i];
                canvasCtx.beginPath();
                canvasCtx.moveTo(a.x * w, a.y * h);
                canvasCtx.lineTo(b.x * w, b.y * h);
                canvasCtx.strokeStyle = `rgba(0, 200, 255, ${i / trail.length})`;
                canvasCtx.lineWidth = 6;
                canvasCtx.stroke();
            }
            const last = trail[trail.length - 1];
            canvasCtx.beginPath();
            canvasCtx.arc(last.x * w, last.y * h, 10, 0, Math.PI * 2);
            canvasCtx.fillStyle = "#00C8FF";
            canvasCtx.fill();
        }

        // Label + confidence overlay (top-left).
        const label = displayLabelRef.current;
        if (label) {
            const conf = Math.round(displayConfRef.current * 100);
            const confident = displayConfRef.current >= CONFIDENCE_THRESHOLD;
            canvasCtx.font = "bold 48px sans-serif";
            canvasCtx.fillStyle = "rgba(0,0,0,0.55)";
            canvasCtx.fillRect(20, 20, 520, 80);
            canvasCtx.fillStyle = confident ? "#3DDC84" : "#E0A030";
            canvasCtx.fillText(`${label}  ${conf}%`, 40, 78);
        }
    };

    const performAction = () => {
        if (results && results.gestures.length == 0) {
            setCurrentGestureText("wave");
        }

        if (results && results.gestures.length > 0) {
            for (let i = 0; i < results.gestures.length; i++) {
                const categoryName = results.gestures[i][0].categoryName;
                const handedness = results.handednesses[i][0].displayName;

                detectAction(categoryName, handedness, results.landmarks[i]);
                handleDrums(handedness, results.landmarks[i]);
                handlePlayPause();
                handleEffects(handedness, results.landmarks[i]);
                handleRegions();
                handleVolume(results.landmarks[i]);
            }
        }
    };

    const detectAction = (categoryName: string, handedness: string, landmarks: any) => {
        model.updateFSMStates(categoryName, handedness, landmarks, getCurrentGestureEl(), model.wsRegions);
        setGestureMesssage();
    };

    const setGestureMesssage = () => {
        const cutText = model.getCutText();
        if (cutText) {
            setCurrentGestureText(cutText);
        }
    };

    const handleDrums = (handedness: string, landmarks: any) => {
        if (handedness == "Left") {
            const sound = model.getDrumSound(landmarks);
            if (sound) {
                ReactGA.event({ category: 'User Interaction', action: 'gesture', label: sound });
                soundManager.playSound(sound);
                setCurrentGestureText("drum");
            }
        }
    };

    const handlePlayPause = () => {
        if (waveform && model.runPlayPause()) {
            waveform.playPause();
        }
    };

    const handleEffects = (handedness: string, landmarks: any) => {
        const speedText = model.getSpeedText(landmarks, handedness);
        if (speedText) {
            setCurrentGestureText(speedText);
            waveform?.setPlaybackRate(soundManager.getSpeedValue());
        }
    };

    const handleRegions = () => {
        if (waveform) {
            model.handleLoopRegions(waveform.getCurrentTime());
        }
    };

    const handleVolume = (landmarks: any) => {
        if (model.isVolumeStarted()) {
            const currentVolume: number = 1 - landmarks[8].x;
            setVolume(Math.min(100, parseFloat((currentVolume * 100).toFixed(0))));
            setIsVolumeVisible(true);
            waveform?.setVolume(currentVolume);

            if (volumeTimer != null) {
                clearTimeout(volumeTimer);
            }
            volumeTimer = setTimeout(() => {
                setIsVolumeVisible(false);
                volumeTimer = null;
            }, 3000);
        }
    };

    // ---------------------------------------------------------------------------
    // Render
    // ---------------------------------------------------------------------------

    return (
        <>
            <div style={{ marginTop: "20px" }}>
                <p id='current_gesture' className="currGesture">wave</p>
                <p className="tooltipGesture">Current gesture</p>
            </div>
            <div className="volumeProgressBar" style={{ display: isVolumeVisible ? "block" : "none" }}>
                <VolumeProgressBar volume={volume}></VolumeProgressBar>
            </div>
            <div>
                <canvas className="output_canvas" id="output_canvas" width="1280" height="720" style={{ position: 'absolute', top: 0, left: 0, zIndex: 1 }} />
            </div>
            <button
                onClick={() => setIsCustomMode(m => !m)}
                style={{
                    backgroundColor: isCustomMode ? '#00C8FF' : '#444',
                    color: 'white', padding: '10px 14px', borderRadius: '5px',
                    border: 'none', position: 'relative', zIndex: 2, marginBottom: '10px',
                    fontWeight: 'bold', display: 'block'
                }}
            >
                {isCustomMode
                    ? "🎯 Custom Test Mode: ON (automatic detection OFF)"
                    : "🎛️ Custom Test Mode: OFF (normal app gestures)"}
            </button>
            <button
                onClick={() => setSonifyOn(s => !s)}
                style={{
                    backgroundColor: sonifyOn ? '#3DDC84' : '#444',
                    color: 'white', padding: '10px 14px', borderRadius: '5px',
                    border: 'none', position: 'relative', zIndex: 2, marginBottom: '10px',
                    fontWeight: 'bold', display: 'block'
                }}
                title="The explanation IS the sound: missing atoms mute notes of a chord; confidence sets brightness."
            >
                {sonifyOn
                    ? "🔊 Sonify Explanation: ON (hear the model's certainty)"
                    : "🔈 Sonify Explanation: OFF"}
            </button>
            <button
                onClick={handleRecordButtonClick}
                style={{ backgroundColor: isRecording ? 'green' : 'blue', color: 'white', padding: '10px', borderRadius: '5px', position: 'relative', zIndex: 2 }}
            >
                {isRecording ? "Stop Recording" : "Start Recording"}
            </button>
            <input
                type="text"
                value={gestureLabel}
                onChange={handleLabelChange}
                placeholder="Enter Gesture Label"
                style={{
                    padding: '10px',
                    borderRadius: '5px',
                    marginRight: '10px',
                    zIndex: 2
                }}
            />
            <label>Current Label: {gestureLabel}</label>
            <p style={{ position: 'relative', zIndex: 2 }}>
                Recorded samples: {recordedGestures.length}
                {recordedGestures.length > 0 && (
                    <> ({Object.entries(sampleCounts(recordedGestures)).map(([l, n]) => `${l}: ${n}`).join(", ")})</>
                )}
                {" "}| Model: {isModelTrained ? "trained ✓" : "not trained"}
            </p>
            <p style={{
                position: 'relative', zIndex: 2, fontWeight: 'bold',
                color: isTraining ? '#b8860b' : (isModelTrained ? 'green' : '#555')
            }}>
                {trainingStatus}
            </p>

            {/* Gesture -> Sound mapping */}
            <div style={{ marginTop: '20px', padding: '10px', border: '1px solid #ccc', borderRadius: '5px', position: 'relative', zIndex: 2 }}>
                <h3>Map Gestures to Sounds</h3>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <select
                        value={mapGesture}
                        onChange={(e) => setMapGesture(e.target.value)}
                        style={{ padding: '8px', borderRadius: '4px' }}
                    >
                        <option value="">Select Gesture</option>
                        {Array.from(new Set(recordedGestures.map(g => g.y))).map(gesture => (
                            <option key={gesture} value={gesture}>{gesture}</option>
                        ))}
                    </select>

                    <select
                        value={mapSound}
                        onChange={(e) => setMapSound(e.target.value)}
                        style={{ padding: '8px', borderRadius: '4px' }}
                    >
                        <option value="">Select Sound</option>
                        <option value="Drum">Drum</option>
                        <option value="Loop">Loop</option>
                        <option value="Melody">Melody</option>
                        <option value="Percussion">Percussion</option>
                    </select>

                    <button
                        onClick={() => {
                            if (mapGesture && mapSound) {
                                setGestureSoundMap({ ...gestureSoundMap, [mapGesture]: mapSound });
                                setMapGesture("");
                                setMapSound("");
                            }
                        }}
                        style={{
                            backgroundColor: 'blue',
                            color: 'white',
                            padding: '8px',
                            borderRadius: '5px',
                            border: 'none'
                        }}
                    >
                        Map
                    </button>
                </div>

                {Object.keys(gestureSoundMap).length > 0 && (
                    <div style={{ marginTop: '10px' }}>
                        <h4>Current Mappings:</h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                            {Object.entries(gestureSoundMap).map(([gesture, sound]) => (
                                <div key={gesture} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
                                    <span>{gesture} -&gt; {sound}</span>
                                    <button
                                        onClick={() => {
                                            const newMap = { ...gestureSoundMap };
                                            delete newMap[gesture];
                                            setGestureSoundMap(newMap);
                                        }}
                                        style={{ backgroundColor: '#ff4d4d', color: 'white', border: 'none', borderRadius: '4px', padding: '2px 6px' }}
                                    >
                                        X
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            <div style={{ marginTop: '10px', padding: '10px', border: '1px solid #ccc', borderRadius: '5px', position: 'relative', zIndex: 2 }}>
                <h3 style={{ margin: '0 0 8px' }}>Test a Gesture</h3>
                <button
                    onClick={predictGesture}
                    disabled={!isModelTrained || isCapturingPredict}
                    style={{
                        backgroundColor: isCapturingPredict ? '#00C8FF' : (isModelTrained ? '#6200EE' : '#999'),
                        color: 'white', padding: '10px 16px', borderRadius: '5px',
                        border: 'none', fontWeight: 'bold'
                    }}
                >
                    {isCapturingPredict ? "✏️ Drawing… draw your shape!" : "▶ Predict a Gesture (draw on click)"}
                </button>
                {!isModelTrained && <span style={{ marginLeft: '10px', color: '#888' }}>Train a model first.</span>}
                {predictStatus && (
                    <p style={{
                        fontSize: '1.3em', fontWeight: 'bold', marginTop: '10px',
                        color: predictedConfidence >= CONFIDENCE_THRESHOLD ? 'green' : '#b8860b'
                    }}>
                        {predictStatus}
                    </p>
                )}
            </div>

            <button
                onClick={() => {
                    console.log("Train Model button clicked");
                    trainModel();
                }}
                disabled={isTraining}
                style={{ backgroundColor: isTraining ? '#999' : 'blue', color: 'white', padding: '10px', borderRadius: '5px', position: 'relative', zIndex: 2 }}
            >
                {isTraining ? "Training…" : "Train Model"}
            </button>

            <AmlPanel
                getTrainingData={getAmlTrainingData}
                captureStroke={captureStroke}
                getSonifier={ensureSonifier}
                sonifyOn={sonifyOn}
                onPredictedLabel={playMappedSound}
            />
        </>
    );
};

export default GestureComponent;
