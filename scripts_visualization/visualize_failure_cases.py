import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_overlay(img, label, font):
    """Draw semi-transparent black overlay at top-left with centered white text."""
    # Convert to RGBA
    img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    padding_x, padding_y = 15, 10
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    box_w = tw + padding_x * 2
    box_h = th + padding_y * 2
    
    # Position: top left
    bx0, by0 = 15, 15
    bx1, by1 = bx0 + box_w, by0 + box_h
    
    # Draw semi-transparent background box
    draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 160))
    
    # Draw text centered in the box (using the bbox to offset correctly)
    tx = bx0 + (box_w - tw) // 2 - bbox[0]
    ty = by0 + (box_h - th) // 2 - bbox[1]
    draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
    
    return Image.alpha_composite(img, overlay).convert('RGB')

def create_failure_analysis_figure(cases, output_path, font_size=48):
    """
    cases: list of dicts
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Load Fonts
    serif_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"
    ]
    font_header = None
    for fp in serif_paths:
        if os.path.exists(fp):
            font_header = ImageFont.truetype(fp, font_size)
            break
    if font_header is None: font_header = ImageFont.load_default()

    labels = ["Ref. Before", "Ref. After", "Input", "Output"]
    rows = []
    spacing = 3
    header_margin = 100
    
    for case in cases:
        img_paths = [case['ref_before'], case['ref_after'], case['input'], case['output']]
        panels = []
        
        for path in img_paths:
            if not os.path.exists(path):
                img = Image.new('RGB', (1024, 1024), (240, 240, 240))
            else:
                img = Image.open(path).convert('RGB')
            panels.append(img)
            
        h = panels[0].height
        resized_panels = []
        for p in panels:
            w = int(p.width * h / p.height)
            resized_panels.append(p.resize((w, h), Image.LANCZOS))
            
        row_w = sum(p.width for p in resized_panels) + spacing * (len(resized_panels) - 1)
        row_canvas = Image.new('RGB', (row_w, h), (255, 255, 255))
        curr_x = 0
        panel_x_starts = [] # To track for header alignment
        for p in resized_panels:
            panel_x_starts.append(curr_x)
            row_canvas.paste(p, (curr_x, 0))
            curr_x += p.width + spacing
        rows.append({'img': row_canvas, 'x_starts': panel_x_starts, 'panel_widths': [p.width for p in resized_panels]})

    if not rows: return
    
    max_w = max(r['img'].width for r in rows)
    total_h = sum(r['img'].height for r in rows) + spacing * (len(rows) - 1) + header_margin
    
    final_canvas = Image.new('RGB', (max_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(final_canvas)
    
    # Draw Headers once at the top
    first_row = rows[0]
    for x, w, label in zip(first_row['x_starts'], first_row['panel_widths'], labels):
        bbox = draw.textbbox((0, 0), label, font=font_header)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        # Center label above the panel
        lx = x + (w - tw) // 2
        ly = (header_margin - th) // 2 - bbox[1]
        draw.text((lx, ly), label, fill=(0, 0, 0), font=font_header)
        
    curr_y = header_margin
    for r in rows:
        final_canvas.paste(r['img'], (0, curr_y))
        curr_y += r['img'].height + spacing
        
    final_canvas.save(output_path, quality=95)
    print(f"Success: Failure analysis figure with XL headers saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case1_id', type=str, default='sample101')
    parser.add_argument('--case2_id', type=str, default='sample102')
    parser.add_argument('--results_dir', type=str, default='tto_visuals_final_font')
    parser.add_argument('--dataset_dir', type=str, default='data/Automatic_Evaluation_Data')
    parser.add_argument('--output_name', type=str, default='failure_analysis.png')
    args = parser.parse_args()

    # Construct paths for the two cases
    cases = []
    for sid in [args.case1_id, args.case2_id]:
        case = {
            'ref_before': os.path.join(args.dataset_dir, sid, f"{sid}_before.jpg"),
            'ref_after': os.path.join(args.dataset_dir, sid, f"{sid}_after.jpg"),
            'input': os.path.join(args.dataset_dir, sid, f"{sid}_input.jpg"),
            'output': os.path.join(args.results_dir, f"{sid}_retouched.png"),
            'sid': sid
        }
        # Fallback for output if retouched.png doesn't exist
        if not os.path.exists(case['output']):
             case['output'] = os.path.join(args.results_dir, sid, 'step_200.png')
             
        cases.append(case)

    create_failure_analysis_figure(cases, args.output_name)
