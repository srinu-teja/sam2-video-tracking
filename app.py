"""
SAM 2.1 Video Object Segmentation & Tracking Pipeline
Interactive Gradio Interface with Spatio-Temporal Memory Propagation
"""

import os
import cv2
import torch
import imageio
import numpy as np
import gradio as gr
from sam2.build_sam import build_sam2_video_predictor

# Configuration paths
CHECKPOINT_PATH = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
VIDEO_DIR = "DAVIS/JPEGImages/480p/lady-running"
OUTPUT_VIDEO_PATH = "davis_evaluation_tracking.mp4"

print("Loading SAM 2.1 Video Predictor...")
predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT_PATH)

# Verify video directory and load first frame
if os.path.exists(VIDEO_DIR):
    frame_names = sorted(os.listdir(VIDEO_DIR))
    first_frame_path = os.path.join(VIDEO_DIR, frame_names[0])
else:
    frame_names = []
    first_frame_path = None


def process_video(evt: gr.SelectData):
    """
    Callback fired when user clicks on the initial frame in Gradio.
    Extracts coordinates and propagates mask across all video frames.
    """
    click_x, click_y = evt.index[0], evt.index[1]
    clicked_points = np.array([[click_x, click_y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)  # Positive prompt

    print(f"Prompt captured: X={click_x}, Y={click_y}. Initializing inference...")
    writer = imageio.get_writer(OUTPUT_VIDEO_PATH, fps=24)

    # Mixed-precision inference without gradients
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=VIDEO_DIR)
        
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=clicked_points,
            labels=labels
        )

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(state):
            frame_path = os.path.join(VIDEO_DIR, frame_names[out_frame_idx])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            color_mask = np.zeros_like(frame)
            color_mask[mask] = [0, 255, 0]  # Green overlay

            overlay = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)
            writer.append_data(overlay)

    writer.close()
    print("Video processing complete. Output saved.")
    return OUTPUT_VIDEO_PATH


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 SAM 2.1 Video Segmentation & Tracking Pipeline")
    gr.Markdown("**Instructions:** Click on the target object in the initial frame. The model propagates the segmentation mask across the entire video sequence.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(value=first_frame_path, label="1. Click target object in frame 0")
        with gr.Column():
            output_video = gr.Video(label="2. Propagated Segmentation Video Output")

    input_image.select(process_video, inputs=None, outputs=output_video)

if __name__ == "__main__":
    demo.launch()
