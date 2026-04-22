"""
Streamlit front-end for the Real-Estate Image Classifier.

Run with:
    streamlit run app/streamlit_app.py
"""
import io
import os
import threading
import time

import av
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ── Page config (MUST be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="Real-Estate Room Classifier",
    page_icon="🏠",
    layout="centered",
)

API_URL = st.sidebar.text_input("API URL", value=os.getenv("API_URL", "http://localhost:8000"))


def fmt_metric(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"

st.title("🏠 Real-Estate Image Classifier")
st.markdown(
    "Upload a property photo and the model will predict the **scene type**."
)

AVAILABLE_CLASSES = [
    "Bedroom", "Coast", "Forest", "Highway", "Industrial",
    "Inside city", "Kitchen", "Living room", "Mountain", "Office",
    "Open country", "Store", "Street", "Suburb", "Tall building",
]
with st.expander("📋 Clases disponibles (15 escenas)"):
    cols = st.columns(3)
    for i, cls in enumerate(AVAILABLE_CLASSES):
        cols[i % 3].write(f"• {cls}")

# ── Health check ──────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health.get("model_loaded"):
        st.sidebar.success(f"API connected — model on **{health['device']}**")
        st.sidebar.markdown(
            f"**Modelo activo**: {health.get('model_name', 'desconocido')}  \n"
            f"**Backbone**: {health.get('backbone', 'desconocido')}  \n"
            f"**Resolución**: {health.get('img_size', 'desconocida')} px"
        )
    else:
        st.sidebar.warning("API connected but **no model loaded**. Train a model first.")
except requests.exceptions.ConnectionError:
    st.sidebar.error("Cannot reach API. Start the FastAPI server first.")
    health = None
except requests.exceptions.RequestException as exc:
    st.sidebar.error(f"API request failed: {exc}")
    health = None

models_payload = None
if health and health.get("model_loaded"):
    try:
        models_payload = requests.get(f"{API_URL}/models", timeout=5).json()
    except requests.exceptions.RequestException:
        st.sidebar.warning("No se pudo cargar la lista de modelos disponibles.")

if health and health.get("model_loaded"):
    st.info(
        f"Modelo en uso: {health.get('model_name', 'desconocido')} "
        f"({health.get('backbone', 'desconocido')}, {health.get('img_size', 'desconocida')} px)"
    )

if models_payload and models_payload.get("models"):
    models = models_payload["models"]
    active_model = models_payload.get("active_model")
    model_names = [item["name"] for item in models]
    default_index = model_names.index(active_model) if active_model in model_names else 0

    selected_model = st.sidebar.selectbox(
        "Selector de modelos",
        options=model_names,
        index=default_index,
        format_func=lambda name: name.replace("_", " "),
    )

    selected_meta = next(item for item in models if item["name"] == selected_model)
    st.sidebar.caption(
        f"Val: {fmt_metric(selected_meta.get('val_acc'))} | "
        f"Test: {fmt_metric(selected_meta.get('test_acc'))} | "
        f"TTA: {fmt_metric(selected_meta.get('test_tta_acc'))}"
    )

    if selected_model != active_model:
        st.sidebar.warning(f"Modelo seleccionado pendiente de activar: {selected_model}")
    if st.sidebar.button("Cambiar modelo", use_container_width=True, disabled=(selected_model == active_model)):
        try:
            response = requests.post(
                f"{API_URL}/models/select",
                json={"name": selected_model},
                timeout=60,
            )
            if response.status_code == 200:
                st.sidebar.success(f"Modelo cambiado a {selected_model}")
                st.rerun()
            else:
                st.sidebar.error(response.json().get("detail", "No se pudo cambiar el modelo"))
        except requests.exceptions.RequestException as exc:
            st.sidebar.error(f"Error al cambiar el modelo: {exc}")


# ══════════════════════════════════════════════════════════════════════
# TABS: Subir imagen  |  Cámara en vivo
# ══════════════════════════════════════════════════════════════════════
tab_upload, tab_camera = st.tabs(["📷 Subir imagen", "🎥 Cámara en vivo"])

# ── TAB 1: Image upload ──────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Choose a property image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Max 10 MB",
    )

    col1, col2 = st.columns([1, 1])

    if uploaded is not None:
        image = Image.open(uploaded)
        col1.image(image, caption="Uploaded image", width="stretch")

        if col1.button("🔍 Classify", type="primary", use_container_width=True):
            with st.spinner("Running inference..."):
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                try:
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                except requests.exceptions.RequestException as exc:
                    st.error(f"Cannot reach the API. Error: {exc}")
                    st.stop()

                if resp.status_code != 200:
                    st.error(f"API error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
                    st.stop()

                result = resp.json()

            # ── Display results ───────────────────────────────────────
            predicted = result["predicted_class"]
            confidence = result["confidence"]
            probs = result["probabilities"]
            prediction_model = result.get("model_name", health.get("model_name", "desconocido") if health else "desconocido")

            col2.markdown("### Prediction")
            col2.caption(f"Modelo usado: {prediction_model}")
            col2.metric(label="Room Type", value=predicted.replace("_", " ").title())
            col2.metric(label="Confidence", value=f"{confidence:.1%}")

            if confidence < 0.50:
                col2.warning(
                    "Confianza baja. Puede que la imagen no pertenezca a ninguna de las 15 clases disponibles "
                    "(por ejemplo, baños no están en el dataset)."
                )

            col2.markdown("---")
            col2.markdown("#### Class Probabilities")
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                label = cls.replace("_", " ").title()
                col2.progress(prob, text=f"{label}: {prob:.1%}")


# ── TAB 2: Live camera ───────────────────────────────────────────────
class SceneClassifier(VideoProcessorBase):
    """Process webcam frames: classify periodically via the API and overlay results."""

    api_url: str = ""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_result = None
        self._last_call = 0.0
        self._interval = 1.0  # seconds between API calls

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # Send frame to API every _interval seconds
        if now - self._last_call >= self._interval:
            self._last_call = now
            try:
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                resp = requests.post(
                    f"{SceneClassifier.api_url}/predict",
                    files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=5,
                )
                if resp.status_code == 200:
                    with self._lock:
                        self._last_result = resp.json()
            except Exception:
                pass

        # Read latest prediction
        with self._lock:
            pred = self._last_result

        # Overlay prediction on the frame
        if pred:
            label = pred["predicted_class"].replace("_", " ").title()
            conf = pred["confidence"]
            probs = pred.get("probabilities", {})
            sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:5]

            h, w = img.shape[:2]
            # Background panel
            panel_h = 40 + len(sorted_probs) * 28
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (min(420, w), panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

            # Main prediction
            color = (0, 255, 0) if conf >= 0.5 else (0, 200, 255)
            cv2.putText(img, f"{label}: {conf:.1%}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

            # Top-5 probabilities
            for i, (cls, prob) in enumerate(sorted_probs):
                y = 58 + i * 28
                cls_label = cls.replace("_", " ").title()
                # Draw bar
                bar_w = int(prob * 250)
                cv2.rectangle(img, (12, y - 12), (12 + bar_w, y + 6), color, -1)
                cv2.putText(img, f"{cls_label}: {prob:.1%}", (14, y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


with tab_camera:
    st.markdown(
        "Activa tu cámara para clasificar escenas **en tiempo real**. "
        "El modelo analiza un fotograma cada segundo y muestra la predicción superpuesta."
    )

    cam_interval = st.slider(
        "Intervalo de clasificación (segundos)", 0.5, 5.0, 1.0, 0.5,
        help="Cada cuántos segundos se envía un fotograma al modelo",
    )

    # Set the API URL for the processor class
    SceneClassifier.api_url = API_URL

    ctx = webrtc_streamer(
        key="scene-classifier",
        video_processor_factory=SceneClassifier,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    if ctx.video_processor:
        ctx.video_processor._interval = cam_interval

    # Show latest prediction in sidebar-style below the video
    if ctx.state.playing and ctx.video_processor:
        result_placeholder = st.empty()
        st.markdown("---")
        st.caption(
            "💡 Las predicciones se muestran superpuestas en el vídeo. "
            "Apunta la cámara a diferentes habitaciones/escenas."
        )


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Real-Estate Classifier — ML2 Final Project | "
    f"[API Docs]({API_URL}/docs)"
)
