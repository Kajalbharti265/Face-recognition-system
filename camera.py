import cv2
import os
import numpy as np
import pandas as pd
import datetime

# ---------- ATTENDANCE ----------
def mark_attendance(name):
    import pandas as pd
    import os
    import datetime

    file = "attendance.csv"
    today = datetime.date.today().strftime("%Y-%m-%d")
    time_now = datetime.datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(file):
        df = pd.DataFrame({
            "Name": [],
            "Date": [],
            "Punch In": [],
            "Punch Out": []
        })
        df.to_csv(file, index=False)

    df = pd.read_csv(file, dtype=str)   # 🔥 FORCE STRING TYPE

    record = df[(df["Name"] == name) & (df["Date"] == today)]

    if record.empty:
        # ✅ PUNCH IN
        df.loc[len(df)] = [name, today, time_now, ""]
        print("✅ Punch In:", name)

    else:
        idx = record.index[0]
        if df.loc[idx, "Punch Out"] == "" or pd.isna(df.loc[idx, "Punch Out"]):
            # ✅ PUNCH OUT
            df.loc[idx, "Punch Out"] = time_now
            print("✅ Punch Out:", name)

    df.to_csv(file, index=False)



# ---------- REGISTER FACE ----------
def register_face(name):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0
    os.makedirs("dataset", exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"dataset/{name}_{count}.jpg", face_img)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == 27 or count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("✅ Face registered")


# ---------- TRAIN MODEL ----------
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for file in os.listdir("dataset"):
        if file.endswith(".jpg"):
            name = file.split("_")[0]

            if name not in label_map:
                label_map[name] = current_id
                current_id += 1

            img = cv2.imread(os.path.join("dataset", file), cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_map[name])

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    np.save("labels.npy", label_map)

    print("✅ Training completed")


# ---------- VIDEO STREAM ----------
def video_stream():
    session_marked = set()   # resets every camera open

    if not os.path.exists("trainer.yml"):
        print("⚠️ Train model first")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    labels = np.load("labels.npy", allow_pickle=True).item()
    labels = {v: k for k, v in labels.items()}

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(face_gray)

            if conf < 80:
                name = labels[id_]

                if name not in session_marked:
                    mark_attendance(name)
                    session_marked.add(name)

                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        _, jpeg = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")