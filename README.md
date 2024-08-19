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
