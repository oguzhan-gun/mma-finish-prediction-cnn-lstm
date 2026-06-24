import numpy as np
from tensorflow.keras.models import load_model
import cv2

model = load_model(
    "train/model.h5",
    compile=False
)

MAX_FRAMES = 32
FRAME_SIZE = (128, 128)
WINDOW_SECONDS = 20


def preprocess_frame(frame):

    frame = cv2.resize(
        frame,
        FRAME_SIZE
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    frame = (
        frame.astype(np.float32)
        / 255.0
    )

    frame = np.expand_dims(
        frame,
        axis=-1
    )

    return frame


def predict_from_frames(frames):

    data = np.array(
        frames,
        dtype=np.float32
    )

    data = np.expand_dims(
        data,
        axis=0
    )

    pred = model.predict(
        data,
        verbose=0
    )[0][0]

    return pred


def predict_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    window_size = int(
        fps * WINDOW_SECONDS
    )

    frame_buffer = []

    predictions = []

    window_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = preprocess_frame(
            frame
        )

        frame_buffer.append(
            frame
        )

        if len(frame_buffer) >= window_size:

            window_number += 1

            indices = np.linspace(
                0,
                len(frame_buffer) - 1,
                MAX_FRAMES
            ).astype(int)

            sampled_frames = [
                frame_buffer[i]
                for i in indices
            ]

            pred = predict_from_frames(
                sampled_frames
            )

            predictions.append(
                pred
            )

            start_time = (
                (window_number - 1)
                * WINDOW_SECONDS
            )

            end_time = (
                window_number
                * WINDOW_SECONDS
            )

            print(
                f"{start_time}-{end_time}s | "
                f"Prediction: {pred:.3f}"
            )

            frame_buffer = []

    cap.release()

    if len(predictions) == 0:

        print(
            "Video çok kısa."
        )

        return 0.0

    final_prediction = max(
        predictions
    )

    confidence = (
        final_prediction
        if final_prediction > 0.5
        else 1 - final_prediction
    )

    label = (
        "FINISHED"
        if final_prediction > 0.5
        else "UNFINISHED"
    )

    print("\n==========")
    print(
        f"FINAL RESULT: {label}"
    )
    print(
        f"CONFIDENCE: %{confidence * 100:.1f}"
    )
    print("==========")

    return final_prediction

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        help="Path to video file"
    )

    args = parser.parse_args()

    predict_video(
        args.video
    )