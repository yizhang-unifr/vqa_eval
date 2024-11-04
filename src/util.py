from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from contextlib import ExitStack 
from pathlib import Path

import pandas as pd



class VisualQA():
    def __init__(self):
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    def extract(self, image_paths: str, query: str, batch_size:int = 10):
        """Retrieves images from the database."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            with ExitStack() as stack:
                images = [stack.enter_context(Image.open(image_path)) for image_path in image_paths[i: i + batch_size]]
                inputs = self.processor(images=images, text=query, return_tensors="pt", padding=True)
                outputs = self.model.generate(**inputs, max_length=20)

                results.extend([self.processor.decode(o, skip_special_tokens=True) for o in outputs])
        return results
    

def to_df(results, image_ids, image_paths, root_path=Path(__file__).parent.parent):
    """Converts the results to a pandas DataFrame."""
    df = pd.DataFrame(results, columns=["Result"])
    df["Image ID"] = image_ids
    df["Image Path"] = list(map(lambda x: x.relative_to(root_path) ,image_paths))
    df = df[["Image ID", "Image Path", "Result"]]
    return df

if __name__ == "__main__":
    visualqa = VisualQA()
    root = Path(__file__).parent.parent
    images = root / "images"
    artwork = images / "artwork"
    image_ids = [0, 5, 10]
    image_paths = [artwork / f"img_{str(image_id)}.jpg" for image_id in image_ids]
    query = "How many people are there in the painting?"
    results = visualqa.extract(image_paths, query)
    df = to_df(results, image_ids, image_paths)
    print(query)
    print(df)