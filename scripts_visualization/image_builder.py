import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image, ImageDraw

def create_diagonal_split(path_before, path_after, output_path):
    # 1. Load images and ensure they match in size
    img_before = Image.open(path_before).convert("RGBA")
    img_after = Image.open(path_after).convert("RGBA")
    
    # Optional: Resize img_after to match img_before if needed
    img_after = img_after.resize(img_before.size)
    w, h = img_before.size
    
    # 2. Create a grayscale mask for the diagonal split
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    
    # Define the polygon for the right side (the "After" part)
    # This creates a diagonal from slightly right of top-center to left of bottom-center
    slope_offset = w // 6 
    polygon = [
        (w // 2 + slope_offset, 0),  # Top point
        (w, 0),                      # Top right
        (w, h),                      # Bottom right
        (w // 2 - slope_offset, h)   # Bottom point
    ]
    draw.polygon(polygon, fill=255)
    
    # 3. Composite the two images using the mask
    composite = Image.composite(img_after, img_before, mask)
    
    # 4. Draw the crisp white separator line down the seam
    final_draw = ImageDraw.Draw(composite)
    final_draw.line(
        [(w // 2 + slope_offset, 0), (w // 2 - slope_offset, h)], 
        fill="white", 
        width=4 # Adjust thickness based on resolution
    )
    
    # 5. Save the result
    composite.convert("RGB").save(output_path)

# import os
# os.makedirs("outputs/with_metaload", exist_ok=True)
# os.makedirs("outputs/without_metaload", exist_ok=True)

# for i in range(101, 205):
#     path_before = f"data/Subjective_Evaluation_Data/sample{i:03d}/sample{i:03d}_input.jpg"

#     path_after_metaload = f"results/subj_eval_results_500/sample{i:03d}_retouched.png"
#     output_path_metaload = f"outputs/with_metaload/composite{i:03d}.png"
#     create_diagonal_split(path_before, path_after_metaload, output_path_metaload)
    
#     path_after_nometaload = f"results_nometaload/sample{i:03d}_retouched.png"
#     output_path_nometaload = f"outputs/without_metaload/composite{i:03d}.png"
#     create_diagonal_split(path_before, path_after_nometaload, output_path_nometaload)


create_diagonal_split(
    "data/Subjective_Evaluation_Data/sample184/sample184_before.jpg",
    "data/Subjective_Evaluation_Data/sample184/sample184_after.jpg",
    "reference_composite184.png"
)