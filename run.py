import os
import cv2
import torch
import numpy as np
from ultralytics import SAM, YOLO
from transformers import pipeline
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from gradio_new import preprocess_image, create_carvekit_interface
import PIL.Image as Image
import argparse

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar_class(detected_classes, target_class, similarity_model):
    target_vector = np.mean(similarity_model(target_class), axis=1)
    max_similarity = -1
    best_class = None

    for detected_class in detected_classes:
        detected_vector = np.mean(similarity_model(detected_class), axis=1)
        similarity = cosine_similarity(target_vector[0], detected_vector[0])

        if similarity > max_similarity:
            max_similarity = similarity
            best_class = detected_class

    return best_class


def segment_object(image_path, class_name, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Load the YOLOv8 and SAM models
    model = YOLO('models/yolov8x.pt')  # YOLOv8 large model for detection
    sam_model = SAM('models/sam_l.pt')  # Load SAM model

    # Move the models to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    sam_model.to(device)

    # Load a pre-trained language model for semantic similarity
    similarity_model = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased", device=0 if device == 'cuda' else -1)

    # Run YOLOv8 inference to detect objects
    results = model(image)
    
    detected_classes = []
    detected_boxes = []

    # Loop through the detections to collect classes and bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls.item()  # class index
            score = box.conf.item()  # confidence score
            if score > 0.5:  # Filter by confidence score
                detected_class_name = model.names[int(cls)]
                detected_classes.append(detected_class_name)
                detected_boxes.append(box.xyxy[0].cpu().numpy())

    # Find the most similar detected class to the user-provided class
    best_class = find_most_similar_class(detected_classes, class_name, similarity_model)
    if best_class:
        best_index = detected_classes.index(best_class)
        bbox = detected_boxes[best_index]

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

        return segmented_image, background_with_cutout, mask_array
    else:
        print(f"No similar object detected for class name: {class_name}")
        return None, None, None


def rotate_and_inpaint(segmented_image, background_with_cutout, original_image_path, mask_array, model_id="kxic/stable-zero123", pose=[0.0, -72.0, 0.0], output_dir="new_logs"):
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

    # Preprocess the images using Carvekit
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
    
    # Convert mask_array to uint8 if it's not already
    mask_array_uint8 = (mask_array * 255).astype(np.uint8) if mask_array.dtype != np.uint8 else mask_array

    # Resize the mask array to match the size of the rotated object
    resized_mask_array = cv2.resize(mask_array_uint8, (rotated_object_np.shape[1], rotated_object_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure the mask is in grayscale mode
    mask_pil = Image.fromarray(resized_mask_array).convert("L")

    # Apply the resized mask on the rotated object to extract only the rotated object
    extracted_rotated_object = np.where(resized_mask_array[:, :, None] > 128, rotated_object_np, 255)  # Object on white background

    # Convert to PIL for compositing
    extracted_rotated_object_pil = Image.fromarray(extracted_rotated_object.astype(np.uint8)).convert("RGB")
    
    # Resize the extracted rotated object to fit the void in the background
    rotated_object_resized = extracted_rotated_object_pil.resize(background_with_cutout.shape[1::-1])  # Corrected to get (width, height)

    # Convert the background with cutout to PIL Image for pasting
    background_with_cutout_pil = Image.fromarray(background_with_cutout).convert("RGB")

    # Ensure the mask is the same size as the background
    if mask_pil.size != background_with_cutout_pil.size:
        mask_pil = mask_pil.resize(background_with_cutout_pil.size, Image.NEAREST)

    # Ensure both images have the same mode and size
    if rotated_object_resized.size != background_with_cutout_pil.size:
        rotated_object_resized = rotated_object_resized.resize(background_with_cutout_pil.size, Image.ANTIALIAS)

    # Paste the rotated object back into the scene using the mask
    composite_image = background_with_cutout_pil.copy()
    composite_image.paste(rotated_object_resized, (0, 0), mask_pil)

    # Now apply inpainting to blend the object into the scene
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to("cuda")
    
    prompt = "background scene is preserved, composite and real. Like post-production editing or photoshop."
    generator = torch.Generator(device="cuda").manual_seed(0)
    inpainted_images = inpaint_pipe(
        prompt=prompt,
        image=composite_image,
        mask_image=mask_pil,
        guidance_scale=7.5,
        generator=generator
    ).images

    # Save the final inpainted image
    inpainted_image_path = os.path.join(output_dir, "inpainted_image.jpg")
    inpainted_images[0].save(inpainted_image_path)
    print(f"Inpainted image saved to {inpainted_image_path}")



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
        print(f"Task1: Red Binary Mask generation of the desired object={args.class_name}")
        # Only perform segmentation and save the red binary mask, no rotation or inpainting
        segment_object(args.image, args.class_name, args.output_dir)
    else:
        print(f"Task2: Novel Synthesis of Projection View at a given pose followed by inpainting with the original scene background.")
        # Segment the object and get the segmented image, background with cutout, and mask array
        segmented_image, background_with_cutout, mask_array = segment_object(args.image, args.class_name, args.output_dir)

        # Perform rotation and inpainting if segmentation was successful
        if segmented_image is not None:
            rotate_and_inpaint(segmented_image, background_with_cutout, args.image, mask_array, pose=[-args.polar, -args.azimuth, 0.0], output_dir=args.output_dir)
