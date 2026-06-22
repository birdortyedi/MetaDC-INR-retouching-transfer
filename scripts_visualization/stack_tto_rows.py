import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from PIL import Image
import numpy as np

def stack_images(input_paths, output_path, gap=5):
    images = [Image.open(p).convert('RGB') for p in input_paths]
    
    # Find max width
    max_w = max(img.width for img in images)
    
    resized_images = []
    for img in images:
        if img.width != max_w:
            new_h = int(round(img.height * max_w / img.width))
            img = img.resize((max_w, new_h), Image.LANCZOS)
        resized_images.append(img)
    
    total_h = sum(img.height for img in resized_images) + gap * (len(resized_images) - 1)
    
    canvas = Image.new('RGB', (max_w, total_h), (255, 255, 255))
    
    curr_y = 0
    for img in resized_images:
        canvas.paste(img, (0, curr_y))
        curr_y += img.height + gap
        
    canvas.save(output_path, quality=95)
    print(f"Combined image saved to {output_path}")

if __name__ == '__main__':
    base_dir = '/home/birdortyedi/inr-retouching/tto_visuals_final_font'
    samples = ['sample166', 'sample146', 'sample125', 'sample149']
    paths = [os.path.join(base_dir, f"{s}_tto_progress.png") for s in samples]
    
    output_dir = '/home/birdortyedi/inr-retouching/tto_visuals_combined'
    os.makedirs(output_dir, exist_ok=True)
    stack_images(paths, os.path.join(output_dir, 'tto_progress_stacked.png'))
