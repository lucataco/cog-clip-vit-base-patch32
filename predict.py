from cog import BasePredictor, Path, Input
from PIL import Image
from typing import List
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_CACHE = "checkpoints"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = CLIPModel.from_pretrained(
           MODEL_NAME,
           cache_dir=MODEL_CACHE
        ).to("cuda")
        self.processor = CLIPProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        image: Path = Input(description="Input Image."),
        text: str = Input(
            description='Description of the image, separate different descriptions with "|"',
            default="a photo of a dog | a cat | two cats with remote controls"
        ),
    ) -> List[float]:

        image = Image.open(str(image))
        text = [t.strip() for t in text.split("|")]
        inputs = self.processor(
            text=text, images=image, return_tensors="pt", padding=True
        ).to("cuda")
        outputs = self.model(**inputs)
        # Image-text similarity score
        logits_per_image = outputs.logits_per_image
        # Softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1)  
        return probs.tolist()[0]
