import argparse
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_length",
    type=int,
    nargs="?",
    default=16,
    help="max caption length"
)
parser.add_argument(
    "--num_beams",
    type=int,
    nargs="?",
    default=4,
    help="num beams"
)

opt = parser.parse_args()

max_length = opt.max_length
num_beams = opt.num_beams
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


# Open or create a file if exists and write the predicted caption
with open("/output/ciclo_limite/caption.txt", "w+") as text_file:
  caption = predict_step(['/output/ciclo_limite/image_00000.png'])
  text_file.write(caption[0])