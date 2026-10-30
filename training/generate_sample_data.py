"""
Procedural Sample Dataset Generator for AI Surface Defect Detection.
Generates realistic benchmark images for immediate training and validation:
- Material: Metal, Plastic, Other
- Cracks: Metal with surface cracks vs. Clean nominal metal

Allows immediate training and testing of both CNN models without needing
to download gigabytes of raw data first.
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

IMG_SIZE = (224, 224)


def create_metal_texture(width=224, height=224, has_crack=False) -> Image.Image:
    """Generates a realistic brushed metal / machined steel surface texture."""
    # Base grayscale aluminum/steel tone
    base_gray = random.randint(140, 200)
    img_arr = np.full((height, width, 3), base_gray, dtype=np.uint8)

    # Add brushed metal directional horizontal/diagonal streaks
    noise = np.random.normal(0, 15, (height, width)).astype(np.float32)
    # Brushed metal rolling effect
    kernel_size = random.randint(7, 15)
    brushed = np.zeros_like(noise)
    for k in range(kernel_size):
        brushed += np.roll(noise, k, axis=1) / kernel_size

    # Add lighting gradient (metallic sheen / specular reflection)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    angle = random.uniform(0, np.pi)
    gradient = np.sin(xv * np.cos(angle) + yv * np.sin(angle)) * random.randint(15, 35)

    composite = np.clip(img_arr[:, :, 0] + brushed * 1.5 + gradient, 30, 245).astype(np.uint8)
    # Slight cold blue/gray metallic tint
    r = composite
    g = np.clip(composite.astype(int) + random.randint(-4, 0), 0, 255).astype(np.uint8)
    b = np.clip(composite.astype(int) + random.randint(2, 8), 0, 255).astype(np.uint8)

    img = Image.fromarray(np.stack([r, g, b], axis=-1))

    if has_crack:
        draw = ImageDraw.Draw(img)
        # Generate jagged hairline crack with branching
        start_x = random.randint(30, width - 60)
        start_y = random.randint(30, height - 60)
        curr_x, curr_y = start_x, start_y
        steps = random.randint(18, 45)

        crack_points = [(curr_x, curr_y)]
        for _ in range(steps):
            curr_x += random.randint(-4, 6)
            curr_y += random.randint(2, 6)
            crack_points.append((curr_x, curr_y))

            # Random branching micro-crack
            if random.random() < 0.25 and len(crack_points) > 5:
                branch_x, branch_y = curr_x, curr_y
                branch_points = [(branch_x, branch_y)]
                for _ in range(random.randint(6, 15)):
                    branch_x += random.randint(-5, 5)
                    branch_y += random.randint(1, 4)
                    branch_points.append((branch_x, branch_y))
                # Draw branch
                draw.line(branch_points, fill=(random.randint(20, 50), random.randint(20, 50), random.randint(20, 50)), width=1)

        # Draw main dark crack line with slight shadow & highlight edge
        dark_color = (random.randint(10, 40), random.randint(10, 40), random.randint(10, 40))
        highlight_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))

        # Crack shadow/highlight for depth
        offset_points = [(p[0] + 1, p[1] + 1) for p in crack_points]
        draw.line(offset_points, fill=highlight_color, width=1)
        draw.line(crack_points, fill=dark_color, width=random.choice([1, 2, 2]))

    return img


def create_plastic_texture(width=224, height=224) -> Image.Image:
    """Generates colored matte or injection-molded polymer plastic texture."""
    # Common plastics: bright blue, safety orange, matte yellow, deep red, polypropylene green
    plastic_palettes = [
        (220, 60, 40),   # Red
        (30, 120, 220),  # Blue
        (240, 180, 20),  # Yellow
        (230, 100, 20),  # Orange
        (40, 160, 90),   # Green
        (240, 240, 240), # White matte
    ]
    base_color = random.choice(plastic_palettes)
    base_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        base_arr[:, :, c] = base_color[c]

    # Add soft smooth plastic specular gradient (smooth curve, no brushed streaks)
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    center_x, center_y = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    dist = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)
    sheen = np.exp(-dist * 2.5) * random.randint(40, 80)

    # Slight fine polymer grain
    grain = np.random.normal(0, 4, (height, width))

    for c in range(3):
        base_arr[:, :, c] = np.clip(base_arr[:, :, c] + sheen + grain, 0, 255).astype(np.uint8)

    return Image.fromarray(base_arr)


def create_other_texture(width=224, height=224) -> Image.Image:
    """Generates non-metal, non-plastic textures (wood grain, cardboard, fabric, desk)."""
    variety = random.choice(["wood", "cardboard", "fabric"])

    if variety == "wood":
        # Brown timber tone with annular growth rings
        base_brown = np.array([random.randint(140, 180), random.randint(90, 120), random.randint(50, 75)])
        img_arr = np.zeros((height, width, 3), dtype=np.uint8)
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        xv, yv = np.meshgrid(x, y)
        rings = np.sin(np.sqrt(xv**2 + 5 * yv**2) * 12) * 20
        noise = np.random.normal(0, 8, (height, width))
        for c in range(3):
            img_arr[:, :, c] = np.clip(base_brown[c] + rings + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    elif variety == "cardboard":
        # Rough textured kraft paper
        base_kraft = np.array([195, 160, 120])
        noise = np.random.normal(0, 14, (height, width))
        img_arr = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            img_arr[:, :, c] = np.clip(base_kraft[c] + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    else:
        # Weave / fabric / desktop surface
        base_tone = np.random.randint(80, 180, 3)
        pattern = ((np.arange(width) % 4 == 0)[:, None] ^ (np.arange(height) % 4 == 0)[None, :]) * 15
        noise = np.random.normal(0, 10, (height, width))
        img_arr = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            img_arr[:, :, c] = np.clip(base_tone[c] + pattern.T + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)


def generate_dataset(samples_per_class=60):
    """
    Generates balanced datasets in:
    dataset/material/metal, plastic, other
    dataset/cracks/crack, no_crack
    """
    print(f"Generating synthetic benchmark datasets ({samples_per_class} samples per class)...")

    # 1. Material Dataset
    mat_dirs = {
        "metal": os.path.join(DATASET_DIR, "material", "metal"),
        "plastic": os.path.join(DATASET_DIR, "material", "plastic"),
        "other": os.path.join(DATASET_DIR, "material", "other"),
    }
    for p in mat_dirs.values():
        os.makedirs(p, exist_ok=True)

    # 2. Crack Dataset
    crack_dirs = {
        "crack": os.path.join(DATASET_DIR, "cracks", "crack"),
        "no_crack": os.path.join(DATASET_DIR, "cracks", "no_crack"),
    }
    for p in crack_dirs.values():
        os.makedirs(p, exist_ok=True)

    # Generate Metal (clean & with crack)
    for i in range(samples_per_class):
        # 50% clean, 50% with surface micro-crack
        has_crack = (i % 2 == 1)
        metal_img = create_metal_texture(224, 224, has_crack=has_crack)
        metal_img.save(os.path.join(mat_dirs["metal"], f"metal_{i:04d}.jpg"), quality=95)

    # Generate Plastic
    for i in range(samples_per_class):
        plastic_img = create_plastic_texture(224, 224)
        plastic_img.save(os.path.join(mat_dirs["plastic"], f"plastic_{i:04d}.jpg"), quality=95)

    # Generate Other
    for i in range(samples_per_class):
        other_img = create_other_texture(224, 224)
        other_img.save(os.path.join(mat_dirs["other"], f"other_{i:04d}.jpg"), quality=95)

    # Generate Crack Dataset (crack vs no_crack)
    for i in range(samples_per_class):
        crack_img = create_metal_texture(224, 224, has_crack=True)
        crack_img.save(os.path.join(crack_dirs["crack"], f"crack_{i:04d}.jpg"), quality=95)

        no_crack_img = create_metal_texture(224, 224, has_crack=False)
        no_crack_img.save(os.path.join(crack_dirs["no_crack"], f"clean_metal_{i:04d}.jpg"), quality=95)

    print("\nDataset Generation Complete!")
    print(f" - Material (Metal, Plastic, Other): {samples_per_class * 3} total images in 'dataset/material/'")
    print(f" - Cracks (Crack, No Crack): {samples_per_class * 2} total images in 'dataset/cracks/'")


if __name__ == "__main__":
    generate_dataset(samples_per_class=60)
