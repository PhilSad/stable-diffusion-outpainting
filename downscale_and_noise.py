from PIL import Image
import numpy as np
import random
from perlin_noise import PerlinNoise
from sklearn.preprocessing import minmax_scale
import click


# todo recode from https://editor.p5js.org/Pi_p5/sketches/qAqoieAhx

def add_noise_to_img(img, dist_from_center=50, amount_of_noise=50):
  img_arr = np.array(img)
  mask = np.zeros_like(img_arr)

  dim = len(img_arr)
  prob_noise = 1
  dist_noise = 0.5
  x,y = dim,dim
  rx,ry = dim,dim
  rx -= rx % 4 
  ry -= ry % 4


  tr_size = dist_from_center/200.0  # [0,100]
  tr_strt = amount_of_noise  #[0,90]


  for i in range(rx):
    for j in range(int(ry*tr_size)):
      if random.randint(0,100) > tr_strt+(j/(ry*tr_size))*(100-tr_strt):
        mask[i][j] = [255,255,255]
        img_arr[i][j] = [0,0,0] 

  for i in range(int(rx*tr_size)):
    for j in range(ry):
      if random.randint(0,100) > tr_strt+(i/(rx*tr_size))*(100-tr_strt):
        mask[i][j] = [255,255,255]
        img_arr[i][j] = [0,0,0] 

  for i in range(rx-1, -1, -1):
    for j in range(ry-1,int(ry-(ry*tr_size)), -1):
      if random.randint(0,100) > tr_strt+((ry-j)/(ry*tr_size))*(100-tr_strt):
        img_arr[i][j] = [0,0,0] 
        mask[i][j] = [255,255,255]

  for i in range(rx-1,int(rx-(rx*tr_size)), -1):
    for j in range(ry-1,-1, -1):
      if random.randint(0,100) > tr_strt+((rx-i)/(rx*tr_size))*(100-tr_strt):
        img_arr[i][j] = [0,0,0] 
        mask[i][j] = [255,255,255]



  return img_arr, mask


downscale_factor = 4

noise11 = PerlinNoise(octaves=10)
noise12 = PerlinNoise(octaves=5)

noise21 = PerlinNoise(octaves=10)
noise22 = PerlinNoise(octaves=5)

noise31 = PerlinNoise(octaves=10)
noise32 = PerlinNoise(octaves=5)


def noise_mult_1(i,j, xpix=512,ypix=512):
  return noise11([i/xpix, j/ypix]) + 0.5 * noise12([i/xpix, j/ypix]) #+ 0.25 * noise3([i/xpix, j/ypix]) + 1.125 * noise4([i/xpix, j/ypix])

def noise_mult_2(i,j, xpix=512,ypix=512):
  return noise21([i/xpix, j/ypix]) + 0.5 * noise22([i/xpix, j/ypix]) #+ 0.25 * noise3([i/xpix, j/ypix]) + 1.125 * noise4([i/xpix, j/ypix])

def noise_mult_3(i,j, xpix=512,ypix=512):
  return noise31([i/xpix, j/ypix]) + 0.5 * noise32([i/xpix, j/ypix]) #+ 0.25 * noise3([i/xpix, j/ypix]) + 1.125 * noise4([i/xpix, j/ypix])

def get_mask_image(img, downscale_factor=4, noise_distance=20, noise_prob=0):

  img_downscaled = img.resize((int(img.size[0] / downscale_factor), int(img.size[1] / downscale_factor)))
  noised_img, mask = add_noise_to_img(img_downscaled, 20, 0)
  img_arr = np.array(img)
  xy1_mask_img = int(len(img_arr) - ( len(img_arr) / downscale_factor ) - ( ( len(img_arr) - (len(img_arr)/ downscale_factor)  ) / 2))
  xy2_mask_img = xy1_mask_img + ( len(img_arr) / downscale_factor )

  full_mask = np.zeros_like(np.array(img))
  full_mask.fill(255)
  for i in range(len(full_mask)):
    for j in range(len(full_mask)):
      if i >= xy1_mask_img and j > xy1_mask_img and i < xy2_mask_img and j < xy2_mask_img:
        full_mask[i][j] = mask[i-xy1_mask_img][j-xy1_mask_img]

  return full_mask, noised_img



def get_init_image(noised_img, full_mask, xpix=512,ypix=512):
  click.echo('Generating noise ...')
  pic = [[[noise_mult_1(i,j), noise_mult_2(i,j), noise_mult_3(i,j) ] for j in range(xpix)] for i in range(ypix)]
  click.echo('Noise generated !')
  scaled_noise = minmax_scale(np.array(pic).flatten(), (0,255)).reshape((512,512, 3))
  scaled_noise = scaled_noise.astype(np.uint8)


  init_image = scaled_noise.copy()
  noised_img_arr = np.array(noised_img)
  xy1_mask_img = int(len(init_image) - ( len(init_image) / downscale_factor ) - ( ( len(init_image) - (len(init_image)/ downscale_factor)  ) / 2))
  xy2_mask_img = xy1_mask_img + ( len(init_image) / downscale_factor )

  for i in range(len(scaled_noise)):
    for j in range(len(scaled_noise)):
      if i >= xy1_mask_img and j > xy1_mask_img and i < xy2_mask_img and j < xy2_mask_img and list(full_mask[i][j]) != [255,255,255]:
        init_image[i][j] = noised_img_arr[i-xy1_mask_img][j-xy1_mask_img]
  return init_image


def get_init_mask_image(img, downscale_factor=4, noise_distance=20, noise_prob=0):
  full_mask, noised_image = get_mask_image(img, downscale_factor,noise_distance, noise_prob)
  init_image = get_init_image(noised_image, full_mask)
  return init_image, full_mask

@click.command()
@click.option('--input_image', help='Image to use as input. (512x512)')
@click.option('--output_init', default="./init.png", help='Path to save init image.')
@click.option('--output_mask', default="./mask.png", help='Path to save mask image.')
@click.option('--downscale_factor', default=4)
@click.option('--noise_distance', default=20)
@click.option('--noise_prob', default=0)

def cmd_get_init_mask_image(input_image,output_init, output_mask, 
                            downscale_factor,noise_distance, noise_prob):
  with open(input_image, 'rb') as f:
    base_image = Image.open(f)
    base_image.load()
  
  init_image, mask = get_init_mask_image(base_image,downscale_factor, noise_distance, noise_prob)
  click.echo('Saving images ...')

  Image.fromarray(init_image).save(output_init)
  Image.fromarray(mask).save(output_mask) 

  click.echo('Done ! :)')



if __name__ == '__main__':
  cmd_get_init_mask_image()






