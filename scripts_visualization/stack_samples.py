import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from PIL import Image

output_dir = "/home/birdortyedi/inr-retouching/FINAL_VISUALIZATIONS_OVERLAY"

# Previous samples:
# samples = ["sample105", "sample177", "sample163", "sample179"]

groups = {
    "darker": ['sample144', 'sample203', 'sample149', 'sample180', 'sample161', 'sample104', 'sample112', 'sample204'],
    "brighter": ['sample182', 'sample126', 'sample175', 'sample109', 'sample107', 'sample132', 'sample174', 'sample117']
}

for group_name, samples in groups.items():
    images = []
    for sample in samples:
        path = os.path.join(output_dir, f"overlay_row_{sample}.jpg")
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            print(f"Warning: {path} not found.")

    if not images:
        print(f"No images found to stack for {group_name}.")
        continue

    widths, heights = zip(*(i.size for i in images))
    min_width = min(widths)

    resized_images = []
    for im in images:
        new_h = int(im.height * (min_width / im.width))
        resized_images.append(im.resize((min_width, new_h), Image.LANCZOS))

    heights = [i.height for i in resized_images]
    total_height = sum(heights)

    GAP = 5
    total_height += GAP * (len(resized_images) - 1)

    new_im = Image.new('RGB', (min_width, total_height), (255, 255, 255))

    y_offset = 0
    for im in resized_images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.height + GAP

    output_path = os.path.join(output_dir, f"stacked_comparison_{group_name}.png")
    new_im.save(output_path, format="PNG", optimize=False)
    print(f"Successfully saved {group_name} stacked image to {output_path}")
