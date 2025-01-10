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
    //const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);

    let canvasElement: any | null = null;
    let canvasCtx: any | null = null;
    let results: any = undefined;
    const videoHeight = "100vh";
    const videoWidth = "auto";
    let volumeTimer: any = null;

    const model: GestureModel = new GestureModel(soundManager);

    const [volume, setVolume] = useState<number>(50);
    const [isVolumeVisible, setIsVolumeVisible] = useState<boolean>(false);
    const [isTraining, setIsTraining] = useState<boolean>(false);
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [recordingStartTime, setRecordingStartTime] = useState<number | null>(null); //msx: manage recording time
    const [frameBuffer, setFrameBuffer] = useState<any[]>([]); //msxL manage frame buffer
    const isRecordingRef = useRef(isRecording); // useRef to track the recording state
    const SLIDING_WINDOW_SIZE = 10; // Number of frames to keep in the sliding window

    const updateLabelMappings = (newLabel: string) => {
        setGestureLabels(prevLabels => {
            if (!prevLabels.includes(newLabel)) {
                const updatedLabels = [...prevLabels, newLabel];
                setLabelToIndex(prevMapping => ({
                    ...prevMapping,
                    [newLabel]: updatedLabels.length - 1
                }));
                return updatedLabels;
            }
            return prevLabels;
        });
    };


    const handleRecordButtonClick = () => {
        setIsRecording(prevIsRecording => {
            const newRecordingState = !prevIsRecording;
            console.log(newRecordingState ? "Started Recording" : "Stopped Recording");
            isRecordingRef.current = newRecordingState;
            if (newRecordingState) {
                startRecording(); // Start recording if the new state is true
            } else {
                stopRecording(); // Stop recording if the new state is false
            }
            
            return newRecordingState;
        });
    };
    
    const [recordedGestures, setRecordedGestures] = useState<any[]>([]);
    const [classifier, setClassifier] = useState<any>(null);
    const [buttonColor, setButtonColor] = useState<string>('blue');
    const [gestureLabel, setGestureLabel] = useState<string>("");// Changed to string
    const [labelToIndex, setLabelToIndex] = useState<{ [key: string]: number }>({});
    const [modelTrained, setModelTrained] = useState<boolean>(false); // New state

    // New state for mode
    const [mode, setMode] = useState<'predefined' | 'self-created'>('predefined');

    //msx: Handle the gesture label change
    const handleLabelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setGestureLabel(event.target.value);
    };
    
    const MAX_RECORDING_DURATION = 2000; // 2 seconds

    const normalizeCoordinates = (landmarks: any) => {
        const basePoint = landmarks[0]; // Assuming wrist as the base point
        return landmarks.map((point: any) => ({
            x: (point.x - basePoint.x),
            y: (point.y - basePoint.y),
            z: point.z ? (point.z - basePoint.z) : 0.0
        }));
    };

    //msx: This function starts the recording process by initializing the buffer and setting the recording state
    const startRecording = () => {
        setIsRecording(true);
        setIsTraining(true);
        setFrameBuffer([]); // Clear any previous data
        setRecordingStartTime(Date.now()); // Record the start time
        isRecordingRef.current = true; // Ensure the ref is updated immediately
        console.log("Recording started");
        // Automatically stop recording after the max duration
        const timeoutID = setTimeout(() => {
            console.log("Inside setTimeout");
            if (isRecordingRef.current){ 
                console.log("Timeout triggered after 2 seconds");
                handleRecordButtonClick(); // Trigger the same function as the button click
                //setIsRecording(false); // Update the state to stop recording
        }
        }, MAX_RECORDING_DURATION);

        // Optional: Clear timeout if component unmounts or recording stops before timeout
        return () => clearTimeout(timeoutID);
    };

    //msx: Captures a single frame of gesture data and adds it to the frame buffer
    const captureFrame = (landmarks: any) => {
        if (isRecording && recordingStartTime) {
            //const features = extractGestureFeatures(landmarks);
            console.log("Inside Capture Frame");
            const normalizedLandmarks = normalizeCoordinates(landmarks);
            const features = extractGestureFeatures(normalizedLandmarks);
            console.log("Captured Frame:", features);

            //Need to change the sliding window
            setFrameBuffer(prevBuffer => {
                const updatedBuffer = [...prevBuffer, features];
                if (updatedBuffer.length > SLIDING_WINDOW_SIZE) {
                    updatedBuffer.shift(); // Remove the oldest frame to maintain the sliding window size
                }
                return updatedBuffer;
            });

        }
    };

    //msx: Feature extraction from the sliding window
    const extractFeatures = (buffer: any[]) => {
        const feats_per_t = 3;  // 2D (x, y) + 3D (z) = 3 features per time step
        const normal_seq_len = 10;  // Sliding window size
        const features = new Array(feats_per_t * normal_seq_len).fill(0.0);

        //(depreacted) for (let t = 0; t < buffer.length; t++) { 
        for (let t = 0; t < buffer.length && t < normal_seq_len; t++) {    
            const point = buffer[t];
            console.log("Point:", point);
            if (point.length > 0) {
                for (let f = 0; f < point.length; f++) {
                    features[feats_per_t * t + f] = point[f];
                    
                }
            }
        }

        return features;
    };

    //msx: Stops the recording process, stores the gesture, and resets the recording state
    const stopRecording = () => {
        if (isRecording) {
            setIsRecording(false);
            isRecordingRef.current = false; // Ensure the ref is updated immediately
            const duration = Date.now() - (recordingStartTime || 0); // Calculate the duration of the recording
            if (frameBuffer.length > 0) {
                const features = extractFeatures(frameBuffer);
                const gesture = { x: features, y: gestureLabel, duration }; // Store the gesture along with its duration (Changed from FrameBuffer to features)
                setRecordedGestures(prevGestures => {
                const updatedGestures = [...prevGestures, gesture];
                console.log("Gesture recorded:", gesture);
                console.log("All Recorded Gestures:", updatedGestures); // Log all gestures for verification
                return updatedGestures;
                //console.log("Gesture recorded:", gesture);
            });
                setFrameBuffer([]); // Clear frame buffer (optional)
            }
            else {
                console.log("No frames to capture");
            }
            
            setRecordingStartTime(null); // Reset the recording start time
        }
    };



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
        isRecordingRef.current = isRecording; //Update the ref with latest recording state 
    }, [video, waveform, isRecording]);

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
        if (!gestureRecognizer) return;
        setupCanvas();
        if (video && video.videoHeight > 0 && video.videoWidth > 0) {
            try {
                results = await gestureRecognizer.recognizeForVideo(video, Date.now());
                if (isRecordingRef.current) {
                    storeGesture(results);
                } else {
                    // Skip triggering music functions if recording
                    if (modelTrained && classifier) {
                        recognizeGesture(results.landmarks);
                    }
                    performAction();
                }
                drawHands();
            } catch (error) {
                console.error("Error during gesture recognition:", error);
            }
        }
        requestAnimationFrame(predictWebcam);
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

    //msx: Extract gesture features using a consistent function
    const extractGestureFeatures = (landmarks: any) => {
        if (!landmarks || landmarks.length === 0) return [0.0, 0.0, 0.0];
        const keypoint = landmarks[8]; // Assume index 8 for a specific keypoint, Index Finger here, can be adjusted
        //const keypoint3d = landmarks3d[8]; // Assume index 8 for a specific keypoint, Index Finger here, can be adjusted
        return [keypoint.x, keypoint.y, keypoint.z || 0.0]; // Consider x, y, and possibly z if available
    };
    
    const storeGesture = (results: any) => {
        console.log("Gesture recognizer results:", results); // Check what `results` contains
        
        if (results && results.landmarks && results.landmarks.length > 0) {
            captureFrame(results.landmarks[0]);
            //const landmarks = results.landmarks[8];
            //const features = extractGestureFeatures(results.landmarks[0]);
            
            //landmarks.map((point: any) => [point.x, point.y, point.z]);
            /* 
            
            
            console.log("Storing gesture features:", features); 
            setRecordedGestures(prevData => {
                const updatedGestures = [...prevData, { x: features, y: "gestureLabel" }];
                console.log("Gesture recorded:", updatedGestures);
                return updatedGestures;
            });
            console.log("Gesture data stored:", { x: features.flat(), y: "gestureLabel" });
            */

        }
    };

    const trainModel = async () => { // Changed to async
        console.log("Start Training");
        if (recordedGestures.length > 0) {
            // Normalize inputs
            const inputs = recordedGestures.map(data => {
                return extractFeatures(data.x);
                //data.x.flat(); // Flatten the input features 
                // A: The model expects a 1D array of features, so we flatten the 2D array  
                // Check if the data is in the expected format
                //if (Array.isArray(data.x) && data.x.every(Number.isFinite)) {
                   
                //} else {
                    //console.error("Invalid data format", data);
                    //return null;
                //}
            }).filter(input => input !== null);
            console.log("Inputs:", inputs);
            const labels = recordedGestures.map(data => data.y === "gestureLabel" ? [1, 0, 0, 0] : [0, 1, 0, 0]);

            const xs = tf.tensor2d(inputs);
            console.log("Tensor Shape:", xs.shape);
            const ys = tf.tensor2d(labels);

            const tfmodel = tf.sequential();
            tfmodel.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [inputs[0].length] }));
            tfmodel.add(tf.layers.dense({ units: 32, activation: 'relu' }));
            tfmodel.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

            tfmodel.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy'],
            });

            // Train the model
            try {
                await tfmodel.fit(xs, ys, {
                    epochs: 30, // Increased epochs
                    validationSplit: 0.2, // Added validation split
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss}, accuracy = ${logs?.acc}`);
                        }
                    }
                });
                console.log("Training complete");
                setClassifier(tfmodel);
                setModelTrained(true); // Set model trained
                setMode('self-created'); // Switch to self-created gestures after training
            } catch (error) {
                console.error("Error during training:", error);
            } finally {
                setIsTraining(false);
            }
        } else {
            console.log("No recorded gestures to train on.");
        }
    };

    const recognizeGesture = (landmarks: any) => {
        if (!classifier) return;
        // Flatten last frames
        const features = extractFeatures(frameBuffer);
        const inputTensor = tf.tensor2d([features]);
        classifier.predict(inputTensor).array().then((predictions: any) => {
            console.log("Gesture Predictions:", predictions[0]);
            // Implement your label mapping here
            const gestureLabels = ["Gesture1", "Gesture2", "Gesture3", "Gesture4"];
            const predictedIndex = predictions[0].indexOf(Math.max(...predictions[0]));
            const predictedGesture = gestureLabels[predictedIndex];
            console.log("Detected Gesture:", predictedGesture);
            // Trigger actions based on `predictedGesture` if in self-created mode
            if (mode === 'self-created') {
                // Implement your action mapping here
            }
        });
    };

    return (
        <>
            <div style={{ marginTop: "20px", display: 'flex', gap: '10px', alignItems: 'center' }}>
                <button
                    onClick={() => setMode('self-created')}
                    className={`btn ${mode === 'self-created' ? 'btn-primary' : 'btn-outline-primary'}`}
                >
                    Self Created Gestures
                </button>
                <button
                    onClick={() => setMode('predefined')}
                    className={`btn ${mode === 'predefined' ? 'btn-primary' : 'btn-outline-primary'}`}
                >
                    Pre Defined Gestures
                </button>
            </div>
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
            <button onClick={() => {
                console.log("Train Model button clicked");
                trainModel();
            }} style={{ backgroundColor: 'blue', color: 'white', padding: '10px', borderRadius: '5px', position: 'relative', zIndex: 2 }}>
                Train Model
            </button>
        </>
    );
};

export default GestureComponent;