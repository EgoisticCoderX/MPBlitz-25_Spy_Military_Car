"""
Image Uploader – Port 5001
No more Internal-Server-Error
"""

import os, cv2, base64, requests, tempfile
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY missing in .env")

# same MODELS dict as above (copy-paste it here)
MODELS = { ... }   # ← paste from live_detection.py

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)
UPLOAD_DIR = tempfile.mkdtemp()

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def detect_img(im_b64, groups):
    dets = []
    for g in groups:
        cfg = MODELS[g]
        for m in cfg["models"]:
            res = requests.post(
                f"https://detect.roboflow.com/{m['id']}?api_key={API_KEY}",
                params={"confidence": cfg["conf"], "overlap": 30, "format": "json"},
                data=im_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            ).json()
            if res and "predictions" in res:
                for p in res["predictions"]:
                    dets.append({
                        "class": p["class"],
                        "confidence": p["confidence"] * m["weight"],
                        "x": p["x"], "y": p["y"],
                        "width": p["width"], "height": p["height"],
                        "group": g
                    })
    return dets

def draw(img, dets):
    for d in dets:
        color = MODELS[d["group"]]["color"]
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["width"]), int(d["height"])
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 18), (x1 + 140, y1), color, -1)
        cv2.putText(img, f"{d['class']} {d['confidence']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            file = request.files['file']
            groups = request.form.getlist('models') or list(MODELS.keys())
            if not file:
                return jsonify({"error": "No file"}), 400
            path = os.path.join(UPLOAD_DIR, secure_filename(file.filename))
            file.save(path)
            dets = detect_img(b64(path), groups)
            img = cv2.imread(path)
            out = draw(img, dets)
            out_path = os.path.join(UPLOAD_DIR, "out_" + secure_filename(file.filename))
            cv2.imwrite(out_path, out)
            return jsonify({"download": "/download/" + os.path.basename(out_path),
                            "detections": dets})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template('upload.html', models=MODELS)

@app.route('/download/<fname>')
def download(fname):
    return send_file(os.path.join(UPLOAD_DIR, fname), as_attachment=True)

if __name__ == '__main__':
    print("Uploader: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False) 