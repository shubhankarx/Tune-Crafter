import React, { useState, useEffect } from "react";
import ReactGA from 'react-ga4';
import RegionsPlugin from 'wavesurfer.js/src/plugin/regions';
import 'bootstrap/dist/css/bootstrap.css';

import { GestureRecognizer, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import WaveSurfer from "wavesurfer.js";
import * as tf from '@tensorflow/tfjs';
import { GestureModel } from "../models/GestureModel";
import { AudioManager } from "../AudioManager";
import VolumeProgressBar from "./VolumeProgressBar";


export interface Coordinates {
    x: number;
    y: number;
}

interface GestureComponentProps {
    video: HTMLVideoElement | null,
    waveform: WaveSurfer | null,
    soundManager: AudioManager
}

const GestureComponent = (props: GestureComponentProps) => {
    const { video, waveform, soundManager } = props;
    let gestureRecognizer: GestureRecognizer | null = null;

    let canvasElement: any | null = null;
    let canvasCtx: any | null = null;
    let results: any = undefined;
    const videoHeight = "100vh";
    const videoWidth = "auto";
    let volumeTimer: any = null;

    const model: GestureModel = new GestureModel(soundManager);

    const [volume, setVolume] = useState<number>(50);
    const [isVolumeVisible, setIsVolumeVisible] = useState<boolean>(false);
    const [isRecording, setIsRecording] = useState<boolean>(false);

    const handleRecordButtonClick = () => {
        setIsRecording(!isRecording);
        console.log(isRecording ? "Stopped Recording" : "Started Recording");
    };

    const [recordedGestures, setRecordedGestures] = useState<any[]>([]);
    const [classifier, setClassifier] = useState<any>(null);
    const [buttonColor, setButtonColor] = useState<string>('blue');

    useEffect(() => {
        if (video && waveform && gestureRecognizer == null) {
            //console.log("Initializing gesture recognizer...");
            createGestureRecognizer().then(() => {
                //console.log("Gesture recognizer initialized.");
                video?.addEventListener("loadeddata", predictWebcam);
                requestAnimationFrame(() => {
                    predictWebcam();
                });
            });
            setAudioObjects();
        }
    }, [video, waveform]);

    const createGestureRecognizer = async () => {
        let recognizer = await loadModelWithRetry();
        if (recognizer) {
            gestureRecognizer = recognizer;
            //console.log("Gesture recognizer created successfully.");
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
        let maxRetries = 3;
        let currentRetry = 0;
        let recognizer;

        while (currentRetry < maxRetries) {
            try {
                //(`Loading model, attempt #${currentRetry + 1}`);
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
        if (gestureRecognizer) {
            setupCanvas();
            if (video && video.videoHeight > 0 && video.videoWidth > 0) {
                //console.log("Video is loaded and has dimensions:", video.videoHeight, video.videoWidth);
                try {
                    results = await gestureRecognizer.recognizeForVideo(video, Date.now());
                    //console.log("Gesture recognizer returned results:", results);
                    if (isRecording) {
                        storeGesture(results);
                    }
                    if (classifier) {
                        recognizeGesture(results.landmarks);
                    }
                    drawHands();
                    performAction();
                } catch (error) {
                    console.error("Error during gesture recognition:", error);
                }
            } else {
                console.log("Video not ready or dimensions not available.");
            }
            requestAnimationFrame(predictWebcam);
        } else {
            console.log("Gesture recognizer not initialized.");
        }
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
        //console.log("Drawing hands...");
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
        } else {
            //console.log("No landmarks detected.");
        }
        canvasCtx.restore();
    };

    const performAction = () => {
        //console.log("Performing action based on gesture...");
        if (results && results.gestures.length == 0) {
            let current_gesture = document.getElementById('current_gesture') as HTMLOutputElement;
            current_gesture.innerText = "🙌";
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
        let current_gesture = document.getElementById('current_gesture') as HTMLOutputElement;
        model.updateFSMStates(categoryName, handedness, landmarks, current_gesture, model.wsRegions);
        setGestureMesssage();
    };

    const setGestureMesssage = () => {
        let current_gesture = document.getElementById('current_gesture') as HTMLOutputElement;
        let cutText = model.getCutText();
        if (cutText) {
            current_gesture.innerText = cutText;
        }
    };

    const handleDrums = (handedness: string, landmarks: any) => {
        if (handedness == "Left") {
            let sound = model.getDrumSound(landmarks);
            if (sound) {
                ReactGA.event({ category: 'User Interaction', action: 'gesture', label: sound });
                soundManager.playSound(sound);
                let current_gesture = document.getElementById('current_gesture') as HTMLOutputElement;
                current_gesture.innerText = "🥁 ✅";
            }
        }
    };

    const handlePlayPause = () => {
        if (waveform && model.runPlayPause()) {
            waveform.playPause();
        }
    };

    const handleEffects = (handedness: string, landmarks: any) => {
        let speedText = model.getSpeedText(landmarks, handedness);
        if (speedText) {
            let current_gesture = document.getElementById('current_gesture') as HTMLOutputElement;
            current_gesture.innerText = speedText;
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
            let currentVolume: number = 1 - landmarks[8].x;
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

    const storeGesture = (results: any) => {
        if (results && results.landmarks) {
            const landmarks = results.landmarks[0];
            const features = landmarks.map((point: any) => [point.x, point.y, point.z]);
            setRecordedGestures(prevData => {
                const updatedGestures = [...prevData, { x: features, y: "gestureLabel" }];
                console.log("Gesture recorded:", updatedGestures);
                return updatedGestures;
            });
            console.log("Gesture data stored:", { x: features.flat(), y: "gestureLabel" });
        }
    };

    const trainModel = () => {
        if (recordedGestures.length > 0) {
            const inputs = recordedGestures.map(data => data.x);
            const labels = recordedGestures.map(data => data.y === "gestureLabel" ? [1, 0, 0, 0] : [0, 1, 0, 0]);

            const xs = tf.tensor2d(inputs);
            const ys = tf.tensor2d(labels);

            const model = tf.sequential();
            model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [inputs[0].length] }));
            model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

            model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy'],
            });

            model.fit(xs, ys, {
                epochs: 20,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch}: loss = ${logs?.loss}`);
                    }
                }
            }).then(() => {
                setClassifier(model);
                console.log("Model trained");
            });
        } else {
            console.log("No recorded gestures to train on.");
        }
    };

    const recognizeGesture = (landmarks: any) => {
        if (classifier) {
            const features = landmarks.map((point: any) => [point.x, point.y, point.z]).flat();
            const input = tf.tensor2d([features]);
            classifier.predict(input).array().then((predictions: any) => {
                console.log("Gesture recognized:", predictions);
                // Implement sound playback based on the recognized gesture
            });
        } else {
            console.log("Classifier not available.");
        }
    };

    return (
        <>
            <div style={{ marginTop: "20px" }}>
                <p id='current_gesture' className="currGesture">🙌</p>
                <p className="tooltipGesture">Current gesture</p>
            </div>
            <div className="volumeProgressBar" style={{ display: isVolumeVisible ? "block" : "none" }}>
                <VolumeProgressBar volume={volume}></VolumeProgressBar>
            </div>
            <div>
            <canvas className="output_canvas" id="output_canvas" width="1280" height="720" style={{ position: 'absolute', top: 0, left: 0, zIndex: 1 }} />
            </div>
            <button
                onClick={handleRecordButtonClick}
                style={{ backgroundColor: isRecording ? 'green' : 'blue', color: 'white', padding: '10px', borderRadius: '5px', position: 'relative', zIndex: 2 }}
>
                {isRecording ? "Stop Recording" : "Start Recording"}
            </button>
            <button onClick={() => {
                console.log("Train Model button clicked");
                trainModel();
            }}style={{ backgroundColor: 'blue', color: 'white', padding: '10px', borderRadius: '5px', position: 'relative', zIndex: 2 }}
>  
                Train Model</button>
        </>
    );
};

export default GestureComponent;
