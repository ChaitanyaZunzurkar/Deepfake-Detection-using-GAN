from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model_definitions import EnhancementGenerator, SingleInputClassifier, FusionClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
GAN_PREPROCESS = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

def _load_state(path, model):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("model_state_dict", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model

class Detector:
    def __init__(self):
        self.models = {}
        self.gan = None
        specs = {
            "M1_original": (SingleInputClassifier, "M1_original_best.pth"),
            "M2_gan_enhanced": (SingleInputClassifier, "M2_gan_enhanced_best.pth"),
            "M3_feature_fusion": (FusionClassifier, "M3_feature_fusion_best.pth"),
        }
        for name, (factory, filename) in specs.items():
            p = MODEL_DIR / filename
            if p.exists():
                self.models[name] = _load_state(p, factory())
        gan_path = MODEL_DIR / "enhancement_best.pth"
        if gan_path.exists():
            checkpoint = torch.load(gan_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("G", checkpoint.get("generator_state_dict", checkpoint))
            self.gan = EnhancementGenerator().to(DEVICE)
            self.gan.load_state_dict(state, strict=True)
            self.gan.eval()

    def available_models(self):
        return list(self.models.keys())

    def _crop_face(self, image):
        # Try the same MTCNN family as training; fall back to full image if not installed/detection fails.
        try:
            from facenet_pytorch import MTCNN
            if not hasattr(self, "_face_detector"):
                self._face_detector = MTCNN(keep_all=True, device=DEVICE, post_process=False)
            boxes, _ = self._face_detector.detect(image)
            if boxes is not None and len(boxes):
                box = max(boxes, key=lambda b: max(0, b[2]-b[0]) * max(0, b[3]-b[1]))
                x1, y1, x2, y2 = map(float, box)
                pad = 0.25 * max(x2-x1, y2-y1)
                x1, y1 = max(0, int(x1-pad)), max(0, int(y1-pad))
                x2, y2 = min(image.width, int(x2+pad)), min(image.height, int(y2+pad))
                if x2 > x1 and y2 > y1:
                    return image.crop((x1, y1, x2, y2)).resize((256, 256), Image.Resampling.BICUBIC)
        except Exception:
            pass
        return image.resize((256, 256), Image.Resampling.BICUBIC)

    @torch.inference_mode()
    def predict_image(self, image, model_name="M1_original"):
        if model_name not in self.models:
            raise FileNotFoundError(f"Model '{model_name}' is not loaded. Check the matching checkpoint in {MODEL_DIR}.")
        image = image.convert("RGB")
        crop = self._crop_face(image)
        raw = PREPROCESS(crop).unsqueeze(0).to(DEVICE)
        enhanced_crop = crop
        if self.gan is not None:
            g = GAN_PREPROCESS(crop).unsqueeze(0).to(DEVICE)
            out = self.gan(g).clamp(0, 1)[0].cpu()
            enhanced_crop = transforms.ToPILImage()(out)
        enhanced = PREPROCESS(enhanced_crop).unsqueeze(0).to(DEVICE)
        model = self.models[model_name]
        if model_name == "M1_original":
            logits = model(raw)
        elif model_name == "M2_gan_enhanced":
            if self.gan is None:
                raise FileNotFoundError("M2 needs models/enhancement_best.pth (GAN checkpoint).")
            logits = model(enhanced)
        else:
            if self.gan is None:
                raise FileNotFoundError("M3 needs models/enhancement_best.pth (GAN checkpoint).")
            logits = model(raw, enhanced)
        p_fake = float(torch.sigmoid(logits).reshape(-1)[0].cpu())
        return {
            "model": model_name,
            "prediction": "FAKE" if p_fake >= 0.5 else "REAL",
            "fake_probability": round(p_fake, 6),
            "real_probability": round(1.0 - p_fake, 6),
            "threshold": 0.5,
            "device": str(DEVICE),
            "note": "Research estimate only; not a forensic guarantee."
        }

    def predict_video(self, video_path, model_name="M1_original", max_frames=12):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open the uploaded video.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames.")
        indices = np.unique(np.linspace(0, total - 1, min(max_frames, total)).astype(int))
        rows = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.predict_image(Image.fromarray(rgb), model_name)
            rows.append({"frame": int(idx), **result})
        cap.release()
        if not rows:
            raise ValueError("No readable frames were sampled.")
        probs = [r["fake_probability"] for r in rows]
        mean_p = float(np.mean(probs))
        return {
            "model": model_name,
            "prediction": "FAKE" if mean_p >= 0.5 else "REAL",
            "mean_fake_probability": round(mean_p, 6),
            "frames_analyzed": len(rows),
            "frame_results": rows,
            "aggregation": "mean of sampled-frame fake probabilities (not calibrated)"
        }
