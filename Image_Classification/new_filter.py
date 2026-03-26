import os

import cv2
import numpy as np

DEFAULT_COLOR_STEP = 48
DEFAULT_SMOOTH_DIAMETER = 11
DEFAULT_SMOOTH_SIGMA = 160
DEFAULT_SATURATION_SCALE = 1.35
DEFAULT_CONTRAST_ALPHA = 1.15
DEFAULT_CONTRAST_BETA = 12
DEFAULT_EDGE_LOW = 70
DEFAULT_EDGE_HIGH = 160
DEFAULT_EDGE_THICKNESS = 2
DEFAULT_HALFTONE_CELL_SIZE = 10
DEFAULT_HALFTONE_STRENGTH = 0.18


def posterize_colors(image_array, color_step=DEFAULT_COLOR_STEP):
    posterized = (image_array // color_step) * color_step
    return posterized.astype(np.uint8)


def boost_comic_colors(image_array):
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * DEFAULT_SATURATION_SCALE, 0, 255)
    boosted_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return cv2.convertScaleAbs(
        boosted_image,
        alpha=DEFAULT_CONTRAST_ALPHA,
        beta=DEFAULT_CONTRAST_BETA
    )


def create_outline_mask(gray_image):
    softened_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(softened_gray, DEFAULT_EDGE_LOW, DEFAULT_EDGE_HIGH)
    kernel = np.ones((DEFAULT_EDGE_THICKNESS, DEFAULT_EDGE_THICKNESS), dtype=np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    return 255 - thick_edges


def add_halftone_shading(image_array, gray_image):
    height, width = gray_image.shape
    cell_size = DEFAULT_HALFTONE_CELL_SIZE
    small_width = max(1, width // cell_size)
    small_height = max(1, height // cell_size)

    reduced_gray = cv2.resize(
        gray_image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    darkness = 1.0 - (reduced_gray / 255.0)
    radius_map = cv2.resize(darkness, (width, height), interpolation=cv2.INTER_NEAREST)
    radius_map = radius_map * (cell_size * 0.45)

    y_positions = np.mod(np.arange(height, dtype=np.float32)[:, None], cell_size) - (cell_size / 2)
    x_positions = np.mod(np.arange(width, dtype=np.float32)[None, :], cell_size) - (cell_size / 2)
    distance_squared = (x_positions ** 2) + (y_positions ** 2)
    dot_mask = (radius_map > 1.2) & (distance_squared <= (radius_map ** 2))

    shaded_image = image_array.astype(np.float32)
    shadow_tint = np.array([28, 8, 36], dtype=np.float32)
    shaded_image[dot_mask] = np.clip(
        shaded_image[dot_mask] * (1.0 - DEFAULT_HALFTONE_STRENGTH) + shadow_tint,
        0,
        255
    )
    return shaded_image.astype(np.uint8)


def apply_comic_filter(image_path, output_path):
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError("Could not open image file.")

        color_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        smoothed_colors = cv2.bilateralFilter(
            color_image,
            d=DEFAULT_SMOOTH_DIAMETER,
            sigmaColor=DEFAULT_SMOOTH_SIGMA,
            sigmaSpace=DEFAULT_SMOOTH_SIGMA
        )
        posterized_colors = posterize_colors(smoothed_colors)
        boosted_colors = boost_comic_colors(posterized_colors)

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        edge_mask = create_outline_mask(gray_image)

        edge_mask_rgb = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2RGB)
        comic_image = cv2.bitwise_and(boosted_colors, edge_mask_rgb)
        comic_image = add_halftone_shading(comic_image, gray_image)
        cv2.imwrite(output_path, cv2.cvtColor(comic_image, cv2.COLOR_RGB2BGR))
        print(f"Processed image saved as '{output_path}'.")
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Comic Filter Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_comic{ext}"
        apply_comic_filter(image_path, output_file)
