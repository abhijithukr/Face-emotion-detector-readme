import cv2
import argparse
from deepface import DeepFace
import numpy as np

# Emotion to color mapping (BGR)
EMOTION_COLORS = {
    "happy":     (0, 255, 128),
    "sad":       (255, 100, 50),
    "angry":     (0, 0, 255),
    "surprise":  (0, 200, 255),
    "fear":      (180, 0, 255),
    "disgust":   (0, 180, 100),
    "neutral":   (200, 200, 200),
}

def draw_label(frame, text, x, y, color):
    """Draw a filled label box with text above the face."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 6), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def analyze_frame(frame):
    """Run DeepFace emotion analysis on a single frame."""
    try:
        results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        return results if isinstance(results, list) else [results]
    except Exception:
        return []


def process_webcam():
    """Run real-time emotion detection on webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    frame_count = 0
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze every 5 frames for performance
        if frame_count % 5 == 0:
            last_results = analyze_frame(frame)

        for face in last_results:
            region = face.get("region", {})
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            emotion = face.get("dominant_emotion", "neutral")
            confidence = face.get("emotion", {}).get(emotion, 0)
            color = EMOTION_COLORS.get(emotion, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = f"{emotion.capitalize()} ({confidence:.0f}%)"
            draw_label(frame, label, x, y, color)

        cv2.imshow("Face Emotion Detector — press Q to quit", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(path):
    """Run emotion detection on a single image file."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"Error: Could not read image at '{path}'")
        return

    results = analyze_frame(frame)

    if not results:
        print("No faces detected.")
    else:
        for face in results:
            region = face.get("region", {})
            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
            emotion = face.get("dominant_emotion", "neutral")
            confidence = face.get("emotion", {}).get(emotion, 0)
            color = EMOTION_COLORS.get(emotion, (200, 200, 200))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{emotion.capitalize()} ({confidence:.0f}%)"
            draw_label(frame, label, x, y, color)

            print(f"Detected: {emotion} ({confidence:.1f}%) at region {region}")

    cv2.imshow("Face Emotion Detector", frame)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    output_path = path.replace(".", "_result.")
    cv2.imwrite(output_path, frame)
    print(f"Result saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Face Emotion Detector using DeepFace")
    parser.add_argument("--source", choices=["webcam", "image"], default="webcam",
                        help="Input source: 'webcam' or 'image'")
    parser.add_argument("--path", type=str, default="",
                        help="Path to image file (required if --source=image)")
    args = parser.parse_args()

    if args.source == "webcam":
        process_webcam()
    elif args.source == "image":
        if not args.path:
            print("Error: --path is required when using --source=image")
        else:
            process_image(args.path)


if __name__ == "__main__":
    main()
