# What this is
Downscale image, add noise on the edge and add perlin noise in the background.

Base image should be 512x512 px. 

|Base Image|Input Image|Mask Image|
|---|---|---|
|![input image](./images/base.png) | ![noised image](./images/init.png) | ![mask image](./images/mask.png)|


Results after inpainting:

![outpainting result](./result.png)



# Installation
Create and activate a python environnement, then:
```bash
pip install -r requirements.txt
```

## Script Usage:

```bash
python ./downscale_and_noise.py --input_image ./images/base.png --output_init ./images/init.png --output_mask ./images/mask.png
```

## Module Usage:

```python
from downscale_and_noise import get_init_mask_image

with open('./images/base.png', 'rb') as f:
    base_image = Image.open(f)
    base_image.load()

init_image, mask_image = get_init_mask_image(base_image)
init_image_arr = np.array(init_image)
mask_image_arr = np.array(mask_image)
```