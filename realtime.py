import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import cv2
import tensorflow as tf
import glob
import gradio as gr

tf.get_logger().setLevel("ERROR")

CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.45
INPUT_SIZE     = 960
SKIP_FRAME     = 2
CLASSES        = ["Buah"]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(max(len(CLASSES), 80), 3), dtype=np.uint8)


def find_model():
    for p in ["*.tflite", "**/*.tflite"]:
        found = glob.glob(p, recursive=True)
        if found:
            return found[0]
    return "model.tflite"


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


interpreter    = load_model(find_model())
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_idx         = input_details[0]['index']
out_idx        = output_details[0]['index']

frame_count     = 0
last_detections = []


def run_inference(frame):
    interpreter.set_tensor(in_idx, preprocess(frame))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)


def detect(image):
    global frame_count, last_detections

    if image is None:
        return None, "Tidak ada input"

    frame_count += 1
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w  = frame.shape[:2]

    if frame_count % SKIP_FRAME == 0:
        output          = run_inference(frame)
        last_detections = postprocess(output, h, w)

    draw(frame, last_detections)

    if last_detections:
        lines = [
            f"{i+1}. {CLASSES[d['class_id']] if d['class_id'] < len(CLASSES) else d['class_id']} {d['conf']:.2f}"
            for i, d in enumerate(last_detections)
        ]
        info = "\n".join(lines)
    else:
        info = "Tidak ada deteksi"

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), info


PORT = int(os.environ.get("PORT", 7860))

with gr.Blocks(title="Test Model Deteksi") as demo:
    with gr.Row():
        cam_in  = gr.Image(sources=["webcam"], streaming=True)
        cam_out = gr.Image()
    det_out = gr.Textbox(lines=4)
    cam_in.stream(fn=detect, inputs=cam_in, outputs=[cam_out, det_out])

demo.launch(server_name="0.0.0.0", server_port=PORT)
