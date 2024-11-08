import cv2
import random
import numpy as np
from augraphy import *

# texture
def dirtydrum(image, **kwargs):
    #print("dirtydrum")
    direction = random.randint(1, 3)
    method = DirtyDrum(
        line_width_range=(1, 3),
        line_concentration=0.3,
        direction=direction,
        noise_intensity=0.2,
        noise_value=(0, 5),
        ksize=(3, 3),
        sigmaX=0,
        )
    
    return method(image)

# texture
def delaunay_pattern(image, **kwargs):
    #print("delaunay_pattern")
    method = DelaunayTessellation(
        n_points_range = (500, 800),
        n_horizontal_points_range=(50, 100),
        n_vertical_points_range=(50, 100),
        noise_type = "random")
    
    return method(image)

# texture
def dirtyscreen(image, **kwargs):
    #print("dirtyscreen")
    method = DirtyScreen(
        n_clusters = (50,100),
        n_samples = (2,20),
        std_range = (1,5),
        value_range = (150,250),
    )

    return method(image)

# texture
def dirthering(image, **kwargs):
    #print("dirthering")
    if random.random() < 0.95:
        method = Dithering(
            dither="ordered",
            )
    else:
        method = Dithering(
            dither="floyd",
            order=(1, 2),
            )
    return method(image)

# texture
def noise_texture(image, **kwargs):
    #print("noise_texture")
    method = NoiseTexturize(
        sigma_range=(2, 3),
        turbulence_range=(2, 5),
        texture_width_range=(50, 500),
        texture_height_range=(50, 500),
        )
    return method(image)

# words
def hollow(image, **kwargs):
    #print("hollow")
    method = Hollow(
        hollow_median_kernel_value_range = (101, 101),
        hollow_min_width_range=(1, 1),
        hollow_max_width_range=(200, 200),
        hollow_min_height_range=(1, 1),
        hollow_max_height_range=(200, 200),
        hollow_min_area_range=(10, 10),
        hollow_max_area_range=(5000, 5000),
        hollow_dilation_kernel_size_range = (3, 3),
        )
    return method(image)

# words
def dotmatrix(image, **kwargs):
    #print("dotmatrix")
    method = DotMatrix(
        dot_matrix_shape="circle",
        dot_matrix_dot_width_range=(2, 3),
        dot_matrix_dot_height_range=(2, 3),
        dot_matrix_min_width_range=(1, 1),
        dot_matrix_max_width_range=(50, 50),
        dot_matrix_min_height_range=(1, 1),
        dot_matrix_max_height_range=(50, 50),
        dot_matrix_min_area_range=(10, 10),
        dot_matrix_max_area_range=(800, 800),
        dot_matrix_median_kernel_value_range=(15, 15),
        dot_matrix_gaussian_kernel_value_range=(1, 1),
        dot_matrix_rotate_value_range=(0, 0)
        )
    return method(image)

# words
def inkbleed(image, **kwargs):
    #print("inkbleed")
    method = InkBleed(
        intensity_range=(0.4, 0.7),
        kernel_size=(5, 5),
        severity=(0.2, 0.4)
        )
    return method(image)

# words
def dilate(image, kernel_size=2, **kwargs):
    #print("dilate")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# words
def erode(image, kernel_size=2, **kwargs):
    #print("erode")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# def inkshift(image):
#     method = InkShifter(
#         text_shift_scale_range=(18, 27),
#         text_shift_factor_range=(1, 4),
#         text_fade_range=(0, 2),
#         noise_type = "random",
#         )
#     return method(image)

# def lensflare(image):
#     method = LensFlare(
#         lens_flare_location = "random",
#         lens_flare_color = "random",
#         lens_flare_size = (0.5, 5),
#         )
#     return method(image)

# brightness
def lighting_gradient_gaussian(image, **kwargs):
    #print("lighting_gradient_gaussian")
    method = LightingGradient(
        light_position=None,
        direction=90,
        max_brightness=255,
        min_brightness=0,
        mode="gaussian",
        transparency=0.5
        )
    return method(image)

# brightness
def lowlightness(image, **kwargs):
    #print("lowlightness")
    method = LowLightNoise(
        num_photons_range = (50, 100),
        alpha_range = (0.7, 0.9),
        beta_range = (10, 30),
        gamma_range = (1.0 , 1.8)
        )
    return method(image)

# brightness
def shadowcast(image, **kwargs):
    #print("shadowcast")
    method = ShadowCast(
        shadow_side = "bottom",
        shadow_vertices_range = (2, 3),
        shadow_width_range=(0.5, 0.8),
        shadow_height_range=(0.5, 0.8),
        shadow_color = (0, 0, 0),
        shadow_opacity_range=(0.5,0.6),
        shadow_iterations_range = (1,2),
        shadow_blur_kernel_range = (101, 301),
        )
    return method(image)

# line
def noisylines(image, **kwargs):
    #print("noisylines")
    method = NoisyLines(
        noisy_lines_direction = 0,
        noisy_lines_location = "random",
        noisy_lines_number_range = (3, 5),
        noisy_lines_color = (0, 0, 0),
        noisy_lines_thickness_range = (1, 2),
        noisy_lines_random_noise_intensity_range = (0.01, 0.1),
        noisy_lines_length_interval_range = (0, 20),
        noisy_lines_gaussian_kernel_value_range = (3, 3),
        noisy_lines_overlay_method = "ink_to_paper",
        )
    return method(image)