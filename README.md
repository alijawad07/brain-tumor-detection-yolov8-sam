# Brain Tumor Detection using YOLOv8 and SAM

This repository contains code for a brain tumor detection project using the YOLOv8 object detection model and the Segment Anything by Meta (SAM) library for segmentation. The project aims to detect brain tumors in medical images and provide potential applications in medical imaging for improved patient care.

## Prerequisites
- Python 3.x
- ultralytics library
- OpenCV (cv2) library
- numpy library
- segment_anything library
- SAM model checkpoints (e.g., sam_vit_h_4b8939.pth)

## Installation
1. Clone the repository:

```bash
git clone https://github.com/alijawad07/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

3. Download the SAM model checkpoint file (`sam_vit_h_4b8939.pth`) and place it in the project directory.
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage
1. Prepare your input data:
   - Organize your brain tumor images in a directory.
   - Make sure you have the path to the directory containing the images (`--source` argument).
   - Specify the output directory path for saving the inference results (`--output` argument).
   - Provide the path to the YOLOv8 model checkpoint file (`--weights` argument).

2. Run the inference on the test images:

```bash
python detect_tumor.py --source /path/to/images/directory --output /path/to/save/results --weights /path/to/yolov8/checkpoint
```

3. The script will process the images, perform object detection using YOLOv8, and extract the highest confidence bounding box. It will then apply segmentation on the detected region using SAM and generate the inferred images with the segmentation mask and bounding box overlaid.

4. Once the inference is complete, an output video named `inferred.mp4` will be saved in the specified output directory.

## Notes
- Ensure that you have the necessary GPU support and CUDA drivers installed for efficient execution.
- Adjust the confidence threshold (`conf`) in the YOLOv8 model prediction to suit your requirements.
- Make sure to download the SAM model checkpoint file compatible with your SAM version.
- Remove the temporary image files generated during the inference process if needed.

Feel free to customize and extend this codebase according to your specific requirements and use cases.
