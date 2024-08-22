import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Elleri başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

draw_color = (0, 0, 255)  # Kırmızı renk (BGR formatında)
erase_color = (0, 0, 0)  # Siyah renk (silme için)

# Webcam başlat
cap = cv2.VideoCapture(0)

# Çizim tuvali oluştur
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Önceki pozisyon değişkenlerini başlat
prev_x, prev_y = 0, 0

# Çizgileri kare üzerinde çizen fonksiyon
def draw_line(canvas, start, end, color, thickness=5):
    cv2.line(canvas, start, end, color, thickness)

# Çizilen alanları kare üzerinde silen fonksiyon
def erase_area(canvas, center, radius, color):
    cv2.circle(canvas, center, radius, color, -1)

# Ana döngü
while True:
    # Webcam'den kareyi oku
    ret, frame = cap.read()
    if not ret:
        break

    # Kareyi yatay olarak çevir
    frame = cv2.flip(frame, 1)

    # Kareyi RGB'ye çevir (MediaPipe için)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El işaretlerini tespit et
    results = hands.process(frame_rgb)

    # El işaretlerini çiz ve el pozisyonlarını al
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # El işaretlerini çizin
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Elin işaretlerini al
            landmarks = hand_landmarks.landmark

            # İşaret parmağı ucunun koordinatlarını al
            index_tip_x = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            index_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            # El bileğinin (el merkezinin) koordinatlarını al
            palm_x = int(landmarks[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
            palm_y = int(landmarks[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # Sol el ve el ayası tespit edilmiş mi diye kontrol et
            if results.multi_handedness[idx].classification[0].label == 'Left' and landmarks[
                mp_hands.HandLandmark.WRIST].x < 0.5:
                # Avuç içi ile sil
                erase_area(canvas, (palm_x, palm_y), 50, erase_color)
            else:
                # İşaret parmağı ile çiz
                if prev_x != 0 and prev_y != 0:
                    draw_line(canvas, (prev_x, prev_y), (index_tip_x, index_tip_y), draw_color, thickness=5)
                prev_x, prev_y = index_tip_x, index_tip_y
    else:
        # El tespit edilmediğinde önceki pozisyonu sıfırla
        prev_x, prev_y = 0, 0

    # Kare ve kanvası birleştir
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Kareyi göster
    cv2.imshow('Frame', combined_frame)

    # Çıkmak için tuşa basılmasını kontrol et
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
