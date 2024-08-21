# edit-object-pose

## Problem Statement: Edit pose of an object in a scene
Recent advancement in generative AI has led to a development of a lot of creative workflows. One
such workflow is to use generative AI techniques for editing product photographs after they have
been shot at a studio, for example to polish the scene for displaying the product on an e-commerce
website. One such post-production editing requirement could be editing the pose of the object by
rotating it within the same scene.
This problem statement involves two tasks - for the eventual goal of developing technology for a
user-friendly pose edit functionality. The first task is to segment an object (defined by a user given
class prompt) in a given scene. This enables the ‘user-friendly’ part of the problem statement. The
second task is to edit the pose of the object by taking user poses (e.g. Azimuth +10 degrees, Polar -5
degrees). The final generated scene should look realistic and composite.

## Tasks and Respective Status:
1. Task1 (SUCCESS). This task is to write an executable code that takes the input scene and the text prompt
from the command line argument and outputs an image with a red mask on all pixels where
the object (denoted in the text prompt) was present.
(e.g. python run.py --image ./example.jpg --class "chair" --output
./generated.png)

2. Task2. The second task is to change the pose of the segmented object by the relative angles
given by the user. You can use a consistent direction as positive azimuth and polar angle
change and mention what you used.
(e.g. python run.py --image ./example.jpg --class "chair" --azimuth
+72 --polar +0 --output ./generated.png)
The generated image:
a. Should preserve the scene (background) (FAIL)
b. Should adhere to the relative angles given by the use (SUCCESS)

### **Prerequisites**

To set up the environment and download the necessary pre-trained models for the script, follow these steps:

```bash
# Create the models directory
mkdir models/

# Navigate to the models directory
cd models/

# Download the SAM (Segment Anything Model) large model
wget -O sam_l.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/sam_l.pt

# Download the YOLOv8x model
wget -O yolov8x.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# Download the YOLOv8x World model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt
```

These commands will create a `models/` directory and download the necessary models for YOLO and SAM, which are required to run the script.
### Detailed Description of the Approach 1 (run.py)

The provided code implements a multi-step process to segment an object from an image, rotate it, and blend it back into the original scene using inpainting techniques. This approach leverages several advanced deep learning models, including YOLO for object detection, SAM for segmentation, and Zero123/Stable Diffusion pipelines for novel view synthesis and inpainting. Below is a detailed breakdown of each step in the approach:

---

### **1. Input Handling and Argument Parsing**
- **Purpose:** This part of the code handles the command-line arguments, allowing the user to specify the input image, target object class, rotation angles (azimuth and polar), and output directory. If the angles are not provided, only segmentation is performed. Otherwise, the full pipeline, including rotation and inpainting, is executed.

- **Key Arguments:**
  - `image`: Path to the input image.
  - `class_name`: Class name of the object to be segmented.
  - `azimuth` and `polar`: Optional angles for rotating the object in 3D space.
  - `output_dir`: Directory to save the output images.

---

### **2. Object Detection using YOLOv8**
- **Purpose:** The code uses YOLOv8, a state-of-the-art object detection model, to identify and localize objects in the image. The model outputs bounding boxes and class names for detected objects.

- **Process:**
  - The YOLO model is loaded with pre-trained weights (`yolov8x.pt`).
  - Inference is run on the input image, resulting in a list of detected objects with bounding boxes and confidence scores.
  - Objects with a confidence score above a threshold (0.5 in this case) are selected for further processing.

---

### **3. Semantic Similarity and Class Matching**
- **Purpose:** Given a user-defined target class, the code finds the most semantically similar detected class using a pre-trained language model (BERT).

- **Process:**
  - The BERT model generates feature vectors for both the target class and detected classes.
  - Cosine similarity is computed between these vectors to find the best match.
  - The most similar class is selected for segmentation.

---

### **4. Object Segmentation using SAM (Segment Anything Model)**
- **Purpose:** The Segment Anything Model (SAM) is used to generate a precise mask for the object corresponding to the selected class.

- **Process:**
  - The bounding box of the matched class is passed to SAM to extract the object's mask.
  - The mask is applied to the image to generate:
    1. **Red Binary Mask:** A mask image highlighting the segmented object in red.
    2. **Segmented Image:** An image with only the segmented object on a black background.
    3. **Background with Cutout:** The original image with the object area blacked out.

- **Output:** The red binary mask, segmented image, and background with cutout are saved.

---

### **5. Novel View Synthesis with Zero123**
- **Purpose:** This step involves generating a novel view of the segmented object by rotating it in 3D space using the Zero123 pipeline.

- **Process:**
  - The segmented image is passed through the Zero123StableDiffusionPipeline.
  - The pipeline generates a rotated view of the object based on the provided azimuth and polar angles.
  - The output is a rotated object image, which is saved for further processing.

---

### **6. Inpainting to Blend the Rotated Object Back into the Scene**
- **Purpose:** The rotated object is blended back into the original scene using inpainting techniques to ensure seamless integration.

- **Process:**
  - The resized mask is used to extract the rotated object.
  - The rotated object is resized to fit the cutout area in the background.
  - The `StableDiffusionInpaintPipeline` is employed to inpaint and blend the object back into the original scene, using the resized mask to guide the inpainting process.
  - The final inpainted image is saved.

---

### **7. Conditional Execution**
- **Purpose:** Depending on whether the user has provided rotation angles, the code either performs only segmentation or the full pipeline including rotation and inpainting.

- **Execution Path:**
  - **Task1:** If no angles are provided, only the segmentation task is performed, generating the red binary mask and other related images.
  - **Task2:** If angles are provided, the segmented object is rotated, and the novel view is blended into the original scene using inpainting.

---

### **Key Takeway**
The initial assumption was that yolov8x model will be capable of detecting every object in the scene, but the class names came out to be different than the class prompt, hence I performed a similarity check to match it with the nearest class detected.

For eg: when sofa image was fed, it gave me label "couch". Which is similar to sofa in the semantic sense of natural language, hence it processed the object sofa effectively.
