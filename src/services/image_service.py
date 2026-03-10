import gc
import numpy as np
import requests
import io
import base64
from PIL import Image
from ..config import ImageGenerationConfig

class ImageService:
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self.api_host = "https://api.stability.ai"

    def load(self):
        """Khởi tạo dịch vụ ảnh (Lightweight - No CLIP needed)."""
        print(f"✓ Analysis tools ready (Using Stability AI Engine: {self.config.engine_id})")

    def unload(self):
        """Dọn dẹp bộ nhớ."""
        gc.collect()

    def _call_stability_api(self, prompt: str, init_image: Image.Image = None, seed: int = 42) -> Image.Image:
        """Gọi Stability AI API (Text-to-Image hoặc Image-to-Image)."""
        if not self.config.stability_api_key or "your_" in self.config.stability_api_key:
            print(f"❌ Error: STABILITY_API_KEY is invalid or placeholder! Key starts with: [{self.config.stability_api_key[:5] if self.config.stability_api_key else 'None'}]")
            return Image.new("RGB", (self.config.image_width, self.config.image_height), (100, 100, 100))

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.stability_api_key}"
        }
        
        try:
            if init_image:
                print(f"Stability AI: Requesting Img2Img (Engine: {self.config.engine_id})...")
                url = f"{self.api_host}/v1/generation/{self.config.engine_id}/image-to-image"
                
                # Chuyển ảnh sang bytes
                img_byte_arr = io.BytesIO()
                init_image.save(img_byte_arr, format='PNG')
                
                # Multipart data PHẢI là string
                files = {"init_image": ("init_image.png", img_byte_arr.getvalue(), "image/png")}
                data = {
                    "text_prompts[0][text]": str(prompt),
                    "cfg_scale": str(self.config.guidance_scale),
                    "samples": "1",
                    "steps": str(self.config.num_inference_steps),
                    "seed": str(seed),
                    "image_strength": str(self.config.refiner_strength),
                    "init_image_mode": "IMAGE_STRENGTH"
                }
                response = requests.post(url, headers=headers, files=files, data=data)
            else:
                print(f"Stability AI: Requesting Text2Img (Engine: {self.config.engine_id})...")
                url = f"{self.api_host}/v1/generation/{self.config.engine_id}/text-to-image"
                payload = {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": self.config.guidance_scale,
                    "height": self.config.image_height,
                    "width": self.config.image_width,
                    "samples": 1,
                    "steps": self.config.num_inference_steps,
                    "seed": seed,
                }
                response = requests.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                raise Exception(f"Stability API Error ({response.status_code}): {response.text}")

            data = response.json()
            image_base64 = data["artifacts"][0]["base64"]
            return Image.open(io.BytesIO(base64.b64decode(image_base64)))

        except Exception as e:
            print(f"❌ Stability API failed: {e}")
            return Image.new("RGB", (self.config.image_width, self.config.image_height), (0, 0, 0))

    def generate(self, prompt: str, seed: int = 42):
        return self._call_stability_api(prompt, seed=seed)

    def refine(self, image: Image.Image, prompt: str, strength: float = 0.35, seed: int = 42):
        # Lưu strength vào config tạm thời để dùng trong _call_stability_api
        old_strength = self.config.refiner_strength
        self.config.refiner_strength = strength
        result = self._call_stability_api(prompt, init_image=image, seed=seed)
        self.config.refiner_strength = old_strength
        return result

    def get_stats(self, image: Image.Image):
        """Tính toán thống kê ảnh sử dụng NumPy (Lightweight)."""
        img_array = np.array(image).astype(np.float32) / 255.0
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=-1)
        return {
            "mean": float(np.mean(img_array)),
            "std": float(np.std(img_array)),
            "max": float(np.max(img_array))
        }

    def unload(self):
        """Dọn dẹp bộ nhớ."""
        gc.collect()
