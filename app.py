from flask import Flask, render_template, Response
from camera import video_stream, register_face, train_model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/register/<name>")
def register(name):
    register_face(name)
    return f"{name} registered successfully!"

@app.route("/train")
def train():
    train_model()
    return "Model trained successfully!"

if __name__ == "__main__":
    app.run(debug=True)