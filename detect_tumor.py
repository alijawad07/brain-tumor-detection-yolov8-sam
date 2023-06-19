from ultralytics import YOLO
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--source', required=True, help='Path to directory containing images')
    parser.add_argument('--output', required=True, help='Path to save the inference result')
    parser.add_argument('--weights',required=True, help='Path to checkpoint file')
    return parser.parse_args()
    
args = parse_args()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='white', facecolor=(0,0,0,0), lw=3))    

# Set the device for GPU acceleration
device = "cuda"

# Load the YOLOv8 model
model = YOLO(args.weights)

#create output directory if it doesnot exist
os.makedirs(args.output, exist_ok=True)

# Open the video file
video_path = args.source
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    sys.exit(1)

# Get the video properties
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Create an output video file
output_path = os.path.join(args.output, "inferred.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, output_size)

# Check if the video writer was initialized successfully
if not out.isOpened():
    print("Error initializing video writer")
    sys.exit(1)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection on the frame
    results = model.predict(image, conf=0.25)

    # Find the detection with the highest confidence
    max_confidence = 0
    max_bbox = None
    for result in results:
        boxes = result.boxes
        confidences = result.boxes.conf
        for bbox, confidence in zip(boxes.xyxy.tolist(), confidences.tolist()):
            if confidence > max_confidence:
                max_confidence = confidence
                max_bbox = bbox

    if max_bbox is not None:
        # Extract the coordinates of the highest confidence bounding box
        input_box = np.array(max_bbox)

        # Perform segmentation on the frame using the highest confidence bounding box
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # Display the frame with the segmentation mask and bounding box
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis('off')  # Turn off axes
        plt.tight_layout()  # Remove padding
        plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)  # Save the figure without extra margins
        plt.close()
        # Read the saved image with the overlay
        overlay_image = cv2.imread('temp.png')

        # Write the frame with the overlay to the output video
        out.write(overlay_image)


# Release the video file and output video
cap.release()
out.release()

# Remove the temporary image file
os.remove('temp.png')

print("Inference completed successfully. Output video saved as inferred.mp4")
