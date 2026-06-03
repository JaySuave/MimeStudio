import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

camera_index = 2
cap = cv2.VideoCapture(camera_index)

FINGER_TIPS = [4, 8, 12, 16, 20]        # polegar, indicador, médio, anelar, mindinho
FINGER_PIPS = [3, 6, 10, 14, 18]        # articulações de referência
FINGER_NAMES = ["Polegar", "Indicador", "Medio", "Anelar", "Mindinho"]

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        if camera_index == 2:
            frame = cv2.flip(frame, 0)  # Flip vertical

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        total_fingers_up = 0
        all_hands_status = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape
                landmarks = hand_landmarks.landmark

                # coords em pixeis (para texto, etc.)
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                fingers_up = 0
                fingers_status = []
                finger_states = []

                # ---- NOVA LÓGICA AQUI ----
                for finger_idx in range(5):
                    tip_idx = FINGER_TIPS[finger_idx]
                    pip_idx = FINGER_PIPS[finger_idx]

                    if finger_idx == 0:
                        # -------- POLEGAR --------
                        # Usamos coordenadas NORMALIZADAS (0..1)
                        tip = landmarks[tip_idx]
                        pip = landmarks[pip_idx]
                        wrist = landmarks[0]

                        # distâncias wrist -> tip e wrist -> pip
                        tip_dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
                        pip_dist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)

                        # quanto mais “para fora” está a ponta em relação ao IP
                        extra_extension = tip_dist - pip_dist

                        # Polegar mais horizontal do que vertical (apontado para o lado)
                        horizontal = abs(tip.x - pip.x) > abs(tip.y - pip.y)

                        # thresholds em coordenadas normalizadas
                        # 0.03–0.05 costuma funcionar, afina se necessário
                        is_up = extra_extension > 0.035 and horizontal

                        finger_states.append(is_up)

                    else:
                        # -------- OUTROS 4 DEDOS (para cima) --------
                        tip_x, tip_y = coords[tip_idx]
                        pip_x, pip_y = coords[pip_idx]
                        finger_states.append(tip_y < pip_y)

                # Contar dedos levantados
                for i, is_up in enumerate(finger_states):
                    if is_up:
                        fingers_up += 1
                        fingers_status.append(FINGER_NAMES[i])

                total_fingers_up += fingers_up
                hand_name = f"Mao {hand_idx + 1}"
                if fingers_status:
                    hand_info = f"{hand_name}: {', '.join(fingers_status)}"
                else:
                    hand_info = f"{hand_name}: Nenhum dedo"
                all_hands_status.append(hand_info)

        cv2.putText(
            frame,
            f"Total dedos levantados: {total_fingers_up}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        if all_hands_status:
            for i, hand_info in enumerate(all_hands_status):
                y_position = 80 + (i * 30)
                cv2.putText(
                    frame,
                    hand_info,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )
        else:
            cv2.putText(
                frame,
                "Nenhuma mao detectada",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Contador de dedos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
