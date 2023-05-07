import argparse
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


class vit_gpt2_image_captioning_model:
  def __init__(self, max_length=16, num_beams=4):
    print("Loading model from initializer")
    self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)

    self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


  def predict_step(self,image_paths):
    images = []
    for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

      images.append(i_image)

    pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(self.device)

    output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

    preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

  def predict_caption(self):
    # Open or create a file if exists and write the predicted caption
    with open("/output/ciclo_limite/caption.txt", "w+") as text_file:
      caption = self.predict_step(['/output/ciclo_limite/image_00000.png'])
      print("Generated Caption:" + caption[0])
      text_file.write(caption[0])