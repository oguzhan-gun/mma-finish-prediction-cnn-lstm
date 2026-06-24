import numpy as np
from tensorflow.keras.models import load_model
import cv2
import argparse


MAX_FRAMES = 32
FRAME_SIZE = (128, 128)
WINDOW_SECONDS = 20


model = load_model(
    "train/model.h5",
    compile=False
)
dummy = np.zeros(
    (1, MAX_FRAMES, 128, 128, 1),
    dtype=np.float32
)

model.predict(
    dummy,
    verbose=0
)

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

    current_pred = 0.0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = preprocess_frame(
            frame
        )

        frame_buffer.append(
            processed_frame
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

            current_pred = pred

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

        display_frame = cv2.resize(
            frame,
            (960, 540)
        )

        label_text = (
            "FINISHED"
            if current_pred > 0.5
            else "UNFINISHED"
        )

        current_time = (
            cap.get(cv2.CAP_PROP_POS_MSEC)
            / 1000.0
        )


        cv2.rectangle(
            display_frame,
            (10, 10),
            (320, 110),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            display_frame,
            f"Prediction: {current_pred:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            display_frame,
            f"Status: {label_text}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            display_frame,
            f"Time: {current_time:.1f}s",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "MMA Finish Prediction",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    
    if len(frame_buffer) > 0:
    
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
    
        predictions.append(pred)
    
        start_time = (
            window_number * WINDOW_SECONDS
        )
    
        duration = len(frame_buffer) / fps
    
        end_time = (
            start_time + duration
        )
    
        print(
            f"{start_time:.1f}-{end_time:.1f}s | "
            f"Prediction: {pred:.3f}"
        )

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
    
    result_frame = display_frame.copy()
    
    while True:
    
        cv2.putText(
            result_frame,
            "Press Q to Exit",
            (700, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )    
    
        cv2.rectangle(
            result_frame,
            (10, 10),
            (420, 140),
            (0, 0, 0),
            -1
        )
    
        cv2.putText(
            result_frame,
            f"Final Prediction: {final_prediction:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    
        cv2.putText(
            result_frame,
            f"Final Status: {label}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    
        cv2.putText(
            result_frame,
            f"Confidence: %{confidence * 100:.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    
        cv2.imshow(
            "MMA Finish Prediction",
            result_frame
        )
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows()
    
    print("\n==========")
    print(
        f"FINAL RESULT: {label}"
    )
    print(
        f"CONFIDENCE: %{confidence * 100:.1f}"
    )
    print("==========")

    return final_prediction


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