import os
import cv2
import torch
import numpy as np
from ultralytics import SAM, YOLOWorld
from transformers import pipeline
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from gradio_new import preprocess_image, create_carvekit_interface
import PIL.Image as Image
import argparse

def segment_object(image_path, class_name, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Initialize a YOLO-World model
    model = YOLOWorld("models/yolov8x-worldv2.pt")

    # Set the custom classes (you can dynamically specify classes based on user input)
    model.set_classes([class_name])

    # Move the model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Run YOLO-World inference to detect objects
    results = model.predict(image_path)

    detected_boxes = []

    # Loop through the detections to collect bounding boxes for the specified class
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_index = int(box.cls)  # class index
            cls_name = model.names[cls_index]  # class name
            score = box.conf  # confidence score
            if score > 0.5 and cls_name == class_name:  # Filter by confidence score and class name
                detected_boxes.append(box.xyxy[0].cpu().numpy())

    if detected_boxes:
        # Assuming the first detected box is the most relevant one
        bbox = detected_boxes[0]

        # Load the SAM model for segmentation
        sam_model = SAM('models/sam_l.pt')
        sam_model.to(device)

        # Use the bounding box with SAM to segment the object
        results_sam = sam_model(image, bboxes=[bbox])

        # Extract the mask from the results_sam object
        masks = results_sam[0].masks.data  # Extract the mask data
        mask_array = masks.cpu().numpy().squeeze()  # Convert to numpy and remove any singleton dimensions

        # Create a blank image for the red mask and a black background for the segmented object
        red_mask_image = np.zeros_like(image)
        black_background = np.zeros_like(image)  # Black background

        # Apply the red color to all pixels where the object is detected
        red_mask_image[mask_array > 0] = [0, 0, 255]  # Red color for the mask

        # Copy the segmented object onto the black background
        segmented_image = black_background.copy()
        segmented_image[mask_array > 0] = image[mask_array > 0]

        # Save the red mask image
        red_mask_path = os.path.join(output_dir, "red_binary_mask.jpg")
        cv2.imwrite(red_mask_path, red_mask_image)
        print(f"Red mask image saved to {red_mask_path}")

        # Create a background image with the object area cut out (black for cutout)
        background_with_cutout = image.copy()
        background_with_cutout[mask_array > 0] = [0, 0, 0]  # Black for cutout

        # Save the background with cutout
        background_cutout_path = os.path.join(output_dir, "background_with_cutout.jpg")
        cv2.imwrite(background_cutout_path, background_with_cutout)
        print(f"Background with cutout saved to {background_cutout_path}")

        return segmented_image, background_with_cutout
    else:
        print(f"No object detected for class name: {class_name}")
        return None, None


def process_rotated_object(rotated_object_np):
    # Convert the image to grayscale to create a mask
    gray_image = cv2.cvtColor(rotated_object_np, cv2.COLOR_RGB2GRAY)
    
    # Create a binary mask where white is converted to black
    _, binary_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Apply the binary mask to set non-object pixels to black
    rotated_object_np[binary_mask == 0] = [0, 0, 0]  # Set background to black

    return rotated_object_np

def rotate_and_inpaint(segmented_image, background_with_cutout, original_image_path, model_id="kxic/stable-zero123", pose=[0.0, -72.0, 0.0], output_dir="new_logs"):
    # Initialize the Zero123 pipeline for generating the rotated view
    pipe = Zero1to3StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if "stable" in model_id:
        pipe.stable_zero123 = True
    
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")

    num_images_per_prompt = 1

    # Convert numpy array to PIL Image if needed
    if isinstance(segmented_image, np.ndarray):
        segmented_image = Image.fromarray(segmented_image)

    # Prepare input images and poses for the pipeline
    input_images = [segmented_image]
    query_poses = [pose]

    # Preprocess the images using Carvekit (assuming this is part of your original process)
    pre_images = []
    models = dict()
    models['carvekit'] = create_carvekit_interface()
    
    for raw_im in input_images:
        input_im = preprocess_image(models, raw_im, True)
        H, W = input_im.shape[:2]  # Define H and W based on the processed image dimensions
        pre_images.append(Image.fromarray((input_im * 255.0).astype(np.uint8)))
    
    input_images = pre_images

    # Run the pipeline for inference to generate the rotated view
    images = pipe(input_imgs=input_images, prompt_imgs=input_images, poses=query_poses, height=H, width=W,
                  guidance_scale=3.0, num_images_per_prompt=num_images_per_prompt, num_inference_steps=50).images

    # Extract the rotated object from the image generated by Zero123
    rotated_object = images[0]
    rotated_object_np = np.array(rotated_object)
    
    # Save the rotated object image
    rotated_object_path = os.path.join(output_dir, "rotated_object.jpg")
    rotated_object.save(rotated_object_path)
    print(f"Rotated object image saved to {rotated_object_path}")

    # Process the rotated object to remove the background
    processed_object_np = process_rotated_object(rotated_object_np)

    # Create a binary mask for the black void in the processed object
    gray_image = cv2.cvtColor(processed_object_np, cv2.COLOR_RGB2GRAY)
    _, black_void_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    black_void_mask = black_void_mask.astype(np.uint8)
    
    # Calculate the bounding box around the object in the processed object
    x, y, w, h = cv2.boundingRect(black_void_mask)
    
    # Resize the processed object to match the size of the bounding box
    resized_object = cv2.resize(processed_object_np[y:y+h, x:x+w], (w, h), interpolation=cv2.INTER_AREA)

    # Adjust the position to ensure the object is resting on the correct surface
    y_offset = y + (h - resized_object.shape[0])

    # Place the resized object on the background image using a mask
    roi = background_with_cutout[y_offset:y_offset+resized_object.shape[0], x:x+resized_object.shape[1]]
    object_mask = cv2.cvtColor(resized_object, cv2.COLOR_RGB2GRAY) > 1  # Create a mask for the object
    roi[object_mask] = resized_object[object_mask]  # Apply the object to the region of interest

    # Save the merged image
    final_image_path = os.path.join(output_dir, "final_merged_image.jpg")
    cv2.imwrite(final_image_path, background_with_cutout)
    print(f"Merged image saved to {final_image_path}")

    return background_with_cutout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an object in an image, rotate it, and perform inpainting to blend it into the scene.")
    parser.add_argument('--image', required=True, help='Path to the input image file')
    parser.add_argument('--class_name', required=True, help='Class name of the object to segment')
    parser.add_argument('--azimuth', type=float, help='Azimuth angle')
    parser.add_argument('--polar', type=float, help='Polar angle')
    parser.add_argument('--output_dir', default="new_logs", help='Directory to save the rotated and inpainted images')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.azimuth is None or args.polar is None:
        print("==========================================================================================================")
        print(f"Task1: Red Binary Mask generation of the desired object={args.class_name}")
        print("==========================================================================================================")
        # Only perform segmentation and save the red binary mask, no rotation or inpainting
        segment_object(args.image, args.class_name, args.output_dir)
    else:
        print("==========================================================================================================")
        print(f"Task2: Novel Synthesis of Projection View at a given pose followed by inpainting with the original scene background.")
        print("==========================================================================================================")
        # Segment the object and get the segmented image and background with cutout
        segmented_image, background_with_cutout = segment_object(args.image, args.class_name, args.output_dir)

        # Perform rotation and inpainting if segmentation was successful
        if segmented_image is not None:
            rotate_and_inpaint(
                segmented_image=segmented_image,
                background_with_cutout=background_with_cutout,
                original_image_path=args.image,
                pose=[-args.polar, -args.azimuth, 0.0],
                output_dir=args.output_dir
            )
