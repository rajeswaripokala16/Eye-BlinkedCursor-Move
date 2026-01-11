import cv2
import mediapipe as mp
import pyautogui
from collections import deque
import time

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE_IDX = [33, 133, 159, 145]  # outer, inner, top, bottom


def get_eye_center(landmarks, frame_w, frame_h):
    left_corner = landmarks[33]
    right_corner = landmarks[133]
    cx = (left_corner.x + right_corner.x) / 2.0
    cy = (left_corner.y + right_corner.y) / 2.0
    return int(cx * frame_w), int(cy * frame_h)


def eye_open_ratio(landmarks):
    top = landmarks[159].y
    bottom = landmarks[145].y
    return abs(top - bottom)


def main():
    cap = cv2.VideoCapture(0)

    # smoothing window for cursor position
    smooth_window = 5
    pos_history = deque(maxlen=smooth_window)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh:

        blink_threshold = 0.004  # tune this
        eye_closed = False
        eye_close_start = 0.0

        dragging = False  # whether mouse is currently held down

        # blink duration thresholds (seconds)
        short_blink_max = 0.25   # <= 0.25s => left click
        long_blink_max = 0.60    # 0.25–0.60s => right click
        drag_blink_min = 0.80    # >= 0.80s => start drag

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                for idx in RIGHT_EYE_IDX:
                    x = int(face_landmarks[idx].x * w)
                    y = int(face_landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                cx, cy = get_eye_center(face_landmarks, w, h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # smooth cursor position
                norm_x = cx / w
                norm_y = cy / h
                screen_x = SCREEN_W * norm_x
                screen_y = SCREEN_H * norm_y
                pos_history.append((screen_x, screen_y))
                avg_x = sum(p[0] for p in pos_history) / len(pos_history)
                avg_y = sum(p[1] for p in pos_history) / len(pos_history)

                pyautogui.moveTo(avg_x, avg_y, duration=0.01)

                # blink detection with timing
                ear = eye_open_ratio(face_landmarks)
                cv2.putText(
                    frame,
                    f"OpenRatio: {ear:.4f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                now = time.time()
                if ear < blink_threshold:
                    # eye is closed
                    if not eye_closed:
                        eye_closed = True
                        eye_close_start = now
                else:
                    # eye open
                    if eye_closed:
                        duration = now - eye_close_start

                        if dragging:
                            # if dragging, opening eye stops drag
                            pyautogui.mouseUp()
                            dragging = False
                        else:
                            if duration <= short_blink_max:
                                pyautogui.click()  # left click
                            elif duration <= long_blink_max:
                                pyautogui.click(button="right")
                            elif duration >= drag_blink_min:
                                pyautogui.mouseDown()
                                dragging = True

                    eye_closed = False

                status = "DRAG" if dragging else "MOVE"
                cv2.putText(
                    frame,
                    f"Mode: {status}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Eye Ball Cursor Movement - Enhanced", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
