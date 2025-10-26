
import pandas
import os
# decode raw image bytes to PIL Image
from io import BytesIO
from PIL import Image
import random
from urllib.request import urlopen

def decode_bytes(file_name, bytes_data):
    img = Image.open(BytesIO(bytes(bytes_data)))
    img.load() 
    img.save(f'{file_name}.png')        # ensure decoded
    return file_name

# Don't run if images folder already exists and contains files
if os.path.exists('images') and len(os.listdir('images')) > 0:
    print("Images folder already exists and is not empty. Skipping download.")
    exit(0)

# download tiny imagenet dataset for demo
url = "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet?download=true"
out_path = "imagenet.parquet"

with urlopen(url) as r, open(out_path, "wb") as f:
    f.write(r.read())

df = pandas.read_parquet('imagenet.parquet')

# ensure output directory exists
os.makedirs('images', exist_ok=True)

for i, img_field in enumerate(df['image']):
    if random.random() < 0.01:  # save 1% of images for demo
        decode_bytes(f'images/{i}', img_field['bytes'])
