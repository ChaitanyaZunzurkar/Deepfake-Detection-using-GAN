"""
Inference pipeline for GAN-based Deepfake Detection.

Models:
    M1_original      : EfficientNet-B0 on original images
    M2_gan_enhanced  : EfficientNet-B0 on GAN-enhanced images
    M3_feature_fusion: FusionCNN on original and enhanced images

Class mapping from the detection notebook:
    0 = REAL
    1 = FAKE
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from model_definitions import (
    SingleCNN,
    FusionCNN,
    EnhancementGenerator,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Confirmed from the detection notebook.
CLASS_NAMES = {
    0: "REAL",
    1: "FAKE",
}

REAL_CLASS_INDEX = 0
FAKE_CLASS_INDEX = 1

DECISION_THRESHOLD = 0.4

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm"
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

GAN_PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def extract_state_dict(checkpoint):
    """Extract a model state dictionary from common checkpoint formats."""

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected a checkpoint dictionary, "
            f"received {type(checkpoint).__name__}."
        )

    possible_keys = (
        "state_dict",
        "model_state_dict",
        "generator_state_dict",
        "model",
        "G",
    )

    for key in possible_keys:
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value

    # The checkpoint itself may already be a state dictionary.
    return checkpoint


def remove_module_prefix(state_dict):
    """Remove DataParallel's 'module.' prefix when present."""

    if state_dict and all(
        key.startswith("module.")
        for key in state_dict
    ):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def load_checkpoint(path):
    """Load a checkpoint onto the CPU before transferring the model."""

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    return remove_module_prefix(
        extract_state_dict(checkpoint)
    )


def load_classifier(path, model):
    """Load classifier weights strictly and prepare inference."""

    state_dict = load_checkpoint(path)

    # Strict loading intentionally exposes architecture mismatches.
    model.load_state_dict(state_dict, strict=True)

    model = model.to(DEVICE)
    model.eval()

    return model


def load_gan(path):
    """Load the trained enhancement generator."""

    state_dict = load_checkpoint(path)

    generator = EnhancementGenerator()
    generator.load_state_dict(state_dict, strict=True)

    generator = generator.to(DEVICE)
    generator.eval()

    return generator


# ============================================================
# DETECTOR
# ============================================================

class Detector:
    def __init__(self):
        self.models = {}
        self.gan = None
        self.model_errors = {}
        self._face_detector = None

        # IMPORTANT:
        # Use the class names from model_definitions.py.
        specifications = {
            "M1_original": (
                SingleCNN,
                "original_cnn.pth",
            ),
            "M2_gan_enhanced": (
                SingleCNN,
                "gan_enhanced_cnn.pth",
            ),
            "M3_feature_fusion": (
                FusionCNN,
                "feature_fusion.pth",
            ),
        }

        # Load each classifier independently.
        for model_name, (model_class, filename) in (
            specifications.items()
        ):
            checkpoint_path = MODEL_DIR / filename

            if not checkpoint_path.exists():
                self.model_errors[model_name] = (
                    f"Checkpoint not found: {checkpoint_path}"
                )
                continue

            try:
                self.models[model_name] = load_classifier(
                    checkpoint_path,
                    model_class(),
                )

                print(
                    f"[OK] Loaded {model_name} "
                    f"from {checkpoint_path.name}"
                )

            except Exception as exc:
                self.model_errors[model_name] = str(exc)

                print(
                    f"[ERROR] Could not load {model_name}: {exc}"
                )

        # The notebook may use either filename.
        gan_candidates = [
            MODEL_DIR / "gan_enhancement_best.pth",
            MODEL_DIR / "enhancement_best.pth",
        ]

        gan_path = next(
            (
                path
                for path in gan_candidates
                if path.exists()
            ),
            None,
        )

        if gan_path is None:
            self.model_errors["GAN"] = (
                "GAN checkpoint not found. Expected one of: "
                + ", ".join(
                    path.name for path in gan_candidates
                )
            )
        else:
            try:
                self.gan = load_gan(gan_path)

                print(
                    f"[OK] Loaded GAN from {gan_path.name}"
                )

            except Exception as exc:
                self.model_errors["GAN"] = str(exc)

                print(
                    f"[ERROR] Could not load GAN: {exc}"
                )

        print(f"Inference device: {DEVICE}")
        print(f"Available classifiers: {self.available_models()}")

    # ========================================================
    # MODEL STATUS
    # ========================================================

    def available_models(self):
        """Return the classifiers that loaded successfully."""

        return list(self.models.keys())

    def get_model_errors(self):
        """Return checkpoint-loading errors."""

        return dict(self.model_errors)

    # ========================================================
    # FACE CROPPING
    # ========================================================

    def _crop_face(self, image):
        """
        Crop the largest detected face if facenet-pytorch is
        installed and face detection succeeds.

        Otherwise, use the entire image.
        """

        image = image.convert("RGB")

        try:
            from facenet_pytorch import MTCNN

            if self._face_detector is None:
                self._face_detector = MTCNN(
                    keep_all=True,
                    device=DEVICE,
                    post_process=False,
                )

            boxes, _ = self._face_detector.detect(image)

            if boxes is not None and len(boxes) > 0:
                box = max(
                    boxes,
                    key=lambda b: max(
                        0, b[2] - b[0]
                    ) * max(
                        0, b[3] - b[1]
                    ),
                )

                x1, y1, x2, y2 = map(float, box)

                padding = 0.25 * max(
                    x2 - x1,
                    y2 - y1,
                )

                x1 = max(0, int(x1 - padding))
                y1 = max(0, int(y1 - padding))
                x2 = min(image.width, int(x2 + padding))
                y2 = min(image.height, int(y2 + padding))

                if x2 > x1 and y2 > y1:
                    return image.crop(
                        (x1, y1, x2, y2)
                    ).resize(
                        (256, 256),
                        Image.Resampling.BICUBIC,
                    )

        except ImportError:
            pass

        except Exception as exc:
            print(f"[WARNING] Face detection failed: {exc}")

        # Fallback: resize the entire image.
        return image.resize(
            (256, 256),
            Image.Resampling.BICUBIC,
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    @staticmethod
    def _interpret_logits(logits):
        """Convert two-class logits into probabilities and labels."""

        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "The classifier must return a PyTorch tensor."
            )

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(
                "Expected logits with shape [batch_size, 2], "
                f"received {tuple(logits.shape)}. "
                "Check the model architecture and checkpoint."
            )

        if logits.shape[0] != 1:
            raise RuntimeError(
                "Expected exactly one prediction at a time."
            )

        probabilities = torch.softmax(logits, dim=1)[0]

        real_probability = float(
            probabilities[REAL_CLASS_INDEX].item()
        )

        fake_probability = float(
            probabilities[FAKE_CLASS_INDEX].item()
        )

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        return {
            "prediction": CLASS_NAMES[predicted_class],
            "predicted_class": predicted_class,
            "fake_probability": round(fake_probability, 6),
            "real_probability": round(real_probability, 6),
        }

    # ========================================================
    # IMAGE PREDICTION
    # ========================================================

    @torch.inference_mode()
    def predict_image(
        self,
        image,
        model_name="M1_original",
    ):
        """Predict whether an image is REAL or FAKE."""

        if model_name not in self.models:
            error = self.model_errors.get(model_name)

            if error:
                raise RuntimeError(
                    f"Model '{model_name}' failed to load: {error}"
                )

            raise ValueError(
                f"Unknown or unavailable model '{model_name}'. "
                f"Available models: {self.available_models()}"
            )

        image = image.convert("RGB")
        crop = self._crop_face(image)

        raw = PREPROCESS(crop).unsqueeze(0).to(DEVICE)

        enhanced = None

        # Generate an enhanced image if the GAN loaded.
        if self.gan is not None:
            gan_input = GAN_PREPROCESS(
                crop
            ).unsqueeze(0).to(DEVICE)

            gan_output = self.gan(
                gan_input
            ).clamp(0.0, 1.0)[0]

            enhanced_image = transforms.ToPILImage()(
                gan_output.cpu()
            )

            enhanced = PREPROCESS(
                enhanced_image
            ).unsqueeze(0).to(DEVICE)

        model = self.models[model_name]

        if model_name == "M1_original":
            logits = model(raw)

        elif model_name == "M2_gan_enhanced":
            if enhanced is None:
                raise RuntimeError(
                    "M2 requires a successfully loaded GAN."
                )

            logits = model(enhanced)

        elif model_name == "M3_feature_fusion":
            if enhanced is None:
                raise RuntimeError(
                    "M3 requires a successfully loaded GAN."
                )

            logits = model(raw, enhanced)

        else:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        result = self._interpret_logits(logits)

        return {
            "model": model_name,
            **result,
            "threshold": DECISION_THRESHOLD,
            "device": str(DEVICE),
            "note": (
                "Research estimate only; not a forensic guarantee. "
                "Class mapping is set to REAL=0 and FAKE=1 "
                "according to the detection notebook."
            ),
        }

    # ========================================================
    # VIDEO PREDICTION
    # ========================================================

    @torch.inference_mode()
    def predict_video(
        self,
        video_path,
        model_name="M1_original",
        max_frames=12,
    ):
        """Sample video frames and aggregate their predictions."""

        if max_frames < 1:
            raise ValueError(
                "max_frames must be at least 1."
            )

        if model_name not in self.models:
            error = self.model_errors.get(model_name)

            if error:
                raise RuntimeError(
                    f"Model '{model_name}' failed to load: {error}"
                )

            raise ValueError(
                f"Unknown or unavailable model '{model_name}'. "
                f"Available models: {self.available_models()}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(
                "Could not open the uploaded video."
            )

        frame_results = []

        try:
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                raise ValueError(
                    "The video contains no readable frames."
                )

            indices = np.unique(
                np.linspace(
                    0,
                    total_frames - 1,
                    min(max_frames, total_frames),
                ).astype(int)
            )

            for index in indices:
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(index),
                )

                success, frame = cap.read()

                if not success:
                    continue

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                result = self.predict_image(
                    Image.fromarray(rgb),
                    model_name=model_name,
                )

                frame_results.append({
                    "frame": int(index),
                    **result,
                })

        finally:
            cap.release()

        if not frame_results:
            raise ValueError(
                "No readable frames could be sampled."
            )

        # Aggregate frame-level fake probabilities.
        mean_fake_probability = float(
            np.mean([
                row["fake_probability"]
                for row in frame_results
            ])
        )

        mean_real_probability = float(
            np.mean([
                row["real_probability"]
                for row in frame_results
            ])
        )

        prediction = (
            "FAKE"
            if mean_fake_probability >= DECISION_THRESHOLD
            else "REAL"
        )

        predicted_class = (
            FAKE_CLASS_INDEX
            if prediction == "FAKE"
            else REAL_CLASS_INDEX
        )

        return {
            "model": model_name,
            "prediction": prediction,
            "predicted_class": predicted_class,
            "fake_probability": round(
                mean_fake_probability, 6
            ),
            "real_probability": round(
                mean_real_probability, 6
            ),
            "threshold": DECISION_THRESHOLD,
            "frames_analyzed": len(frame_results),
            "frame_results": frame_results,
            "aggregation": (
                "Mean of sampled-frame probabilities; "
                "not a calibrated video-level probability."
            ),
            "device": str(DEVICE),
            "note": (
                "Research estimate only; not a forensic guarantee."
            ),
        }

    # ========================================================
    # UNIFIED FILE PREDICTION
    # ========================================================

    def predict_file(
        self,
        uploaded_file,
        model_name="M1_original",
        max_frames=12,
    ):
        """Route an image or video path to the correct predictor."""

        path = Path(uploaded_file)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        extension = path.suffix.lower()

        if extension in IMAGE_EXTENSIONS:
            with Image.open(path) as image:
                image.load()

                return self.predict_image(
                    image.convert("RGB"),
                    model_name=model_name,
                )

        if extension in VIDEO_EXTENSIONS:
            return self.predict_video(
                path,
                model_name=model_name,
                max_frames=max_frames,
            )

        raise ValueError(
            f"Unsupported file extension: {extension}"
        )
