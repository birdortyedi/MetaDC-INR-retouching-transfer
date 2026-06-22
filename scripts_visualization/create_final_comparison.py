import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from PIL import Image, ImageDraw, ImageFont

# Configuration
DATA_DIR = "/home/birdortyedi/inr-retouching/data/Subjective_Evaluation_Data"
M1_DIR = "/home/birdortyedi/inr-retouching/QUALITATIVE_RESULTS/M1"
M2_DIR = "/home/birdortyedi/inr-retouching/QUALITATIVE_RESULTS/M2"
OURS_DIR = "/home/birdortyedi/inr-retouching/results/subj_eval_results_500"
OUTPUT_DIR = "/home/birdortyedi/inr-retouching/FINAL_VISUALIZATIONS_OVERLAY"

# Get all samples automatically
SAMPLES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
TARGET_HEIGHT = 512
SPACING = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_font(size):
    serif_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
    ]
    for fp in serif_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

def draw_overlay(img, label, font):
    """Draw semi-transparent black overlay with white text on the image."""
    img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    padding_x, padding_y = 12, 8
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    box_w = tw + padding_x * 2
    box_h = th + padding_y * 2
    
    # Position: top left
    bx0, by0 = 10, 10
    bx1, by1 = bx0 + box_w, by0 + box_h
    
    # Draw semi-transparent background box
    draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 160))
    
    # Draw text
    tx = bx0 + padding_x - bbox[0]
    ty = by0 + padding_y - bbox[1]
    draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
    
    return Image.alpha_composite(img, overlay).convert('RGB')

def draw_dashed_line(draw, x, height, color=(100, 100, 100), dash_length=10):
    for y in range(0, height, dash_length * 2):
        draw.line([x, y, x, min(y + dash_length, height)], fill=color, width=2)

def get_image_path(folder, sample_id, suffix, extensions=[".jpg", ".png"]):
    for ext in extensions:
        path = os.path.join(folder, f"{sample_id}{suffix}{ext}")
        if os.path.exists(path):
            return path
    return None

def process_sample(sample_id, font):
    print(f"Processing {sample_id}...")
    sample_dir = os.path.join(DATA_DIR, sample_id)
    
    # Image Paths
    paths = [
        os.path.join(sample_dir, f"{sample_id}_before.jpg"),
        os.path.join(sample_dir, f"{sample_id}_after.jpg"),
        os.path.join(sample_dir, f"{sample_id}_input.jpg"),
        get_image_path(M1_DIR, sample_id, ""),
        get_image_path(M2_DIR, sample_id, ""),
        get_image_path(OURS_DIR, sample_id, "_retouched") or get_image_path(OURS_DIR, sample_id, "_input")
    ]
    
    labels = ["Ref. Before", "Ref. After", "Input", "InRetouch", "Team A", "MetaDC-INR"]
    
    images = []
    for p, label in zip(paths, labels):
        if p and os.path.exists(p):
            img = Image.open(p).convert('RGB')
            w, h = img.size
            new_w = int(w * (TARGET_HEIGHT / h))
            img = img.resize((new_w, TARGET_HEIGHT), Image.LANCZOS)
            # Add overlay text ON image
            img = draw_overlay(img, label, font)
            images.append(img)
        else:
            images.append(Image.new('RGB', (TARGET_HEIGHT, TARGET_HEIGHT), (240, 240, 240)))

    # Calculate total dimensions
    SEPARATOR_WIDTH = 15
    total_width = sum(img.width for img in images) + SPACING * (len(images) - 2) + SEPARATOR_WIDTH
    
    canvas = Image.new('RGB', (total_width, TARGET_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    current_x = 0
    for i, img in enumerate(images):
        canvas.paste(img, (current_x, 0))
        
        if i == 1:
            # Draw dashed separator line after Ref. After (index 1)
            line_x = current_x + img.width + SEPARATOR_WIDTH // 2
            draw_dashed_line(draw, line_x, TARGET_HEIGHT)
            current_x += img.width + SEPARATOR_WIDTH
        else:
            current_x += img.width + SPACING

    output_path = os.path.join(OUTPUT_DIR, f"overlay_row_{sample_id}.jpg")
    canvas.save(output_path, quality=95)

if __name__ == "__main__":
    font = get_font(28)
    for s in SAMPLES:
        process_sample(s, font)
    print(f"\nOverlay visualization rows created in {OUTPUT_DIR}")
