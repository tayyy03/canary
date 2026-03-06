import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import time
import glob

tf.get_logger().setLevel("ERROR")

st.set_page_config(page_title="Test Model", layout="wide")
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {padding: 0.5rem;}
</style>
""", unsafe_allow_html=True)

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
INPUT_SIZE     = 960
CAM_W, CAM_H   = 480, 360
SKIP_FRAME     = 3
CLASSES        = ["Buah"]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(max(len(CLASSES), 80), 3), dtype=np.uint8)


def find_model():
    for p in ["*.tflite", "**/*.tflite"]:
        found = glob.glob(p, recursive=True)
        if found:
            return found[0]
    return "model.tflite"


def find_cameras(max_test=6):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


@st.cache_resource
def get_cameras():
    return find_cameras()


@st.cache_resource
def load_model(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=4)
    interp.allocate_tensors()
    return interp


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis]


def postprocess(output, orig_h, orig_w):
    preds = output[0]
    num_classes = preds.shape[0] - 4
    boxes  = preds[:4, :].T
    scores = preds[4:, :].T

    if num_classes == 1:
        confs     = scores[:, 0]
        class_ids = np.zeros(len(confs), dtype=np.int32)
    else:
        class_ids = np.argmax(scores, axis=1).astype(np.int32)
        confs     = scores[np.arange(len(scores)), class_ids]

    mask = confs > CONF_THRESHOLD
    if not mask.any():
        return []

    boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    if cx.max() <= 2.0:
        sx, sy = float(orig_w), float(orig_h)
    else:
        sx, sy = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE

    x1 = np.clip((cx - bw / 2) * sx, 0, orig_w - 1).astype(np.int32)
    y1 = np.clip((cy - bh / 2) * sy, 0, orig_h - 1).astype(np.int32)
    x2 = np.clip((cx + bw / 2) * sx, 0, orig_w - 1).astype(np.int32)
    y2 = np.clip((cy + bh / 2) * sy, 0, orig_h - 1).astype(np.int32)

    valid = (x2 - x1 > 2) & (y2 - y1 > 2)
    if not valid.any():
        return []

    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    confs, class_ids = confs[valid], class_ids[valid]
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(xyxy.tolist(), confs.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
    if len(indices) == 0:
        return []

    return [{"box": xyxy[i], "conf": float(confs[i]), "class_id": int(class_ids[i])}
            for i in indices.flatten()]


def draw(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        label  = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        color  = [int(c) for c in COLORS[cls_id % len(COLORS)]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 5, 0)), (x1 + tw + 3, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, max(y1 - 3, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


with st.sidebar:
    model_path = st.text_input("Model (.tflite)", value=find_model())

    with st.spinner("Scanning kamera..."):
        cameras = get_cameras()

    if cameras:
        cam_options = {f"Kamera {i}": i for i in cameras}
        cam_label   = st.selectbox("Pilih Kamera", list(cam_options.keys()))
        cam_index   = cam_options[cam_label]
        st.success(f"{len(cameras)} kamera: {cameras}")
    else:
        st.error("Tidak ada kamera")
        cam_index = None

    stop_btn = st.button("Stop", use_container_width=True)

st.markdown("Test Model Deteksi")

if not os.path.exists(model_path):
    st.error(f"Model tidak ditemukan: `{model_path}`")
    st.stop()

if cam_index is None:
    st.warning("Tidak ada kamera tersedia.")
    st.stop()

interpreter    = load_model(model_path)
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_idx         = input_details[0]['index']
out_idx        = output_details[0]['index']


def run_inference(frame):
    interpreter.set_tensor(in_idx, preprocess(frame))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)


col_vid, col_info = st.columns([3, 1])
frame_slot = col_vid.empty()

with col_info:
    fps_slot   = st.empty()
    count_slot = st.empty()
    st.markdown("---")
    det_slot   = st.empty()

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    st.error(f"Gagal membuka kamera index {cam_index}.")
    st.stop()

last_detections = []
frame_count     = 0
t_prev          = time.time()

while not stop_btn:
    ret, frame = cap.read()
    if not ret:
        st.error("Kamera terputus.")
        break

    frame_count += 1

    if frame_count % SKIP_FRAME == 0:
        output          = run_inference(frame)
        last_detections = postprocess(output, frame.shape[0], frame.shape[1])

    draw(frame, last_detections)

    now    = time.time()
    fps    = 1 / (now - t_prev + 1e-9)
    t_prev = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 80), 2)

    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     channels="RGB", use_container_width=True)
    fps_slot.metric("FPS", f"{fps:.1f}")
    count_slot.metric("Objek", len(last_detections))

    if last_detections:
        lines = [
            f"**{i+1}. {CLASSES[d['class_id']] if d['class_id'] < len(CLASSES) else d['class_id']}** `{d['conf']:.2f}`"
            for i, d in enumerate(last_detections)
        ]
        det_slot.markdown("\n\n".join(lines))
    else:
        det_slot.markdown("_Tidak ada deteksi_")

cap.release()