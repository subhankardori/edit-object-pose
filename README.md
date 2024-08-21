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

### **Tasks and Respective Status**

#### **Task 1: Red Mask Generation (SUCCESS)**
- **Description:** 
  - This task involves writing executable code that takes an input image and a text prompt (representing an object class) from the command line. The code outputs an image with a red mask on all pixels where the object specified in the text prompt is present.
- **Execution Example:**
  - Command: `python run.py --image ./example.jpg --class "chair" --output ./generated.png`
  - **Status:** SUCCESS
  - The output image correctly identifies the object (e.g., a chair) and applies a red mask over it.

#### **Task 2: Pose Change of the Segmented Object**
- **Description:** 
  - This task requires altering the pose of the segmented object based on relative angles (azimuth and polar) provided by the user. The positive direction for azimuth and polar angles should be consistent and explicitly mentioned.
- **Execution Example:**
  - Command: `python run.py --image ./example.jpg --class "chair" --azimuth +72 --polar +0 --output ./generated.png`
  
- **Subtasks and Status:**
  - **a. Scene Preservation (Background) (FAIL):**
    - The generated image should preserve the original scene, meaning the background should remain intact after altering the object's pose.
    - **Status:** FAIL
    - **Issue:** The background scene was not preserved accurately during the pose change, leading to inconsistencies or artifacts in the final image.
  
  - **b. Adherence to Relative Angles (SUCCESS):**
    - The object's pose should be changed according to the specified relative angles (azimuth and polar) provided by the user.
    - **Status:** SUCCESS
    - The object was correctly rotated according to the specified angles.

### **Summary:**
- **Task 1** was successfully completed, with the code accurately generating a red mask over the specified object.
- **Task 2** had mixed results. While the object was correctly rotated according to the relative angles, the background scene was not preserved as required, leading to a failure in maintaining scene integrity during the pose transformation.

### **Setting Up the Python Environment**

To set up a Python virtual environment and install the required dependencies, follow these steps:

```bash
# Create a Python virtual environment named 'object-pose-env'
python3 -m venv object-pose-env

# Activate the virtual environment
source object-pose-env/bin/activate
cd edit-object-pose/
# Install the required Python packages from the requirements.txt file
pip install -r requirements.txt
```

These commands will create and activate a virtual environment called `object-pose-env` and install all the necessary Python packages as specified in your `requirements.txt` file. This environment helps in managing dependencies and ensuring that your project runs consistently across different setups.
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
#### CLI for Script Execution
```bash
python run.py --image pose_editing_examples/chair2.jpeg --class_name "chair" --output_dir output/ # Task 1
python run.py --image pose_editing_examples/chair2.jpeg --class_name "chair" --output_dir output/ --azimuth +72 --polar +0 #Task 2
```
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

**Key Addon done in `run_v2.py`:**
1. **Object Detection with YOLOWorld:**
   - **Model Initialization:** The script uses YOLOWorld, a model designed to handle dynamic class definitions, which allows for the detection of objects based on custom prompts.
   - **Detection Process:** The model is configured to detect objects within an image using the specified class name provided by the user. The detected bounding boxes for the relevant objects are collected.
   - **Advantage:** YOLOWorld's dynamic class capability enables the detection of a wide variety of objects without needing to retrain the model, making it flexible and adaptable.

### Failed Cases
```
Failed Case: Laptop
Command:
python run2.py --image pose_editing_examples/laptop.jpg --class_name "laptop" --output_dir output/ --azimuth +60 --polar -20
```

**Log Output:**

```
(zero123-env) dori@nemo:~/subhankar/zero123-hf$ python run2.py --image pose_editing_examples/laptop.jpg --class_name "laptop" --output_dir output/ --azimuth +60 --polar -20
/home/dori/subhankar/zero123-hf/zero123-env/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/home/dori/subhankar/zero123-hf/zero123-env/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
==========================================================================================================
Task2: Novel Synthesis of Projection View at a given pose followed by inpainting with the original scene background.
==========================================================================================================

image 1/1 /home/dori/subhankar/zero123-hf/pose_editing_examples/laptop.jpg: 640x640 1 laptop, 33.0ms
Speed: 4.9ms preprocess, 33.0ms inference, 10.4ms postprocess per image at shape (1, 3, 640, 640)
No object detected for class name: laptop
```

**Description:**

The script failed to detect the object when using the "laptop" class with the specified image. This issue might be related to the detection model's limitations with certain classes or specific images. The failure to detect the object means the subsequent steps in the pipeline could not be executed.

**Possible SOlutions**: need to have a better view of the object while capturing, try lowering the YOLO detection confidence 

**All the Results can be found in the results folder.**
