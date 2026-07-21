## 1. Problem Statement
1. Back in hyperverge, we use to go through the failed samples from production as much as possible. And try to identify the failure cases.
2. Then, we will try to generate those sample types and include them in our next training iteration.
3. With that we'll test that version. If it covers those failed cases. We are good. If not, we would again do this whole process again.

So, here instead of this training testing loop.
I have tried to skip the training step and try quantify the distributional gap between the real and the synthetic dataset.

--> In HV, this was my workflow, for all the projects i worked on. Like tamper detection, quality check model, signature detection.



## 2. Task and Data

The underline task is to detect objects in images. 
Where objects with different settings like, texture visibility, HDRI, noise level, camera pose, object pose, occlusion - might make it difficult for the model to detect the objects.
We try to cover those edge cases with a synth dataset. 

Now we try to score that gap we have between the real and the generated synth dataset. And decide if we need to improve the edge cases in the synth generated.

--- 

Why DINOv2 Embeddings?
    DINOv2 embeddings are advanced numerical blueprints for images, created by a powerful computer vision AI developed by Meta AI.
    While text embeddings map the meanings of words, DINOv2 embeddings map the visual details, structures, and textures of pictures. It looks at an image and translates it into a list of numbers (a vector) that captures exactly what is happening visually.

### Phase 0:
    1. Done, downloading the CAD models (21 classes), the BOP test dataset(12 classes), and related jsons.
    2. And we just picked 8 target classes from the models. 
        (   because, 21 classes are too much, its not the point. Reduced the classes to save any extra overhead.
            Why those 8, picking both low-texture and high texture items, to have good variations)
    3. And we create a YOLO friendly ground-truth dataset.

    a. data/ycvb/models/ --> CAD models data from BOP
    b. data/ycvb/test/ --> test samples from BOP for YCB_V
    c. data/yolo_dataset/test --> YOLO labels ~ version of BOP labels from data/test => prep from prep_ycbv_test.ipynb

### Phase 1 (Establish the baseline):
    1. Now we are generating a single "low-randomization" synth dataset using blenderproc
    2. And train the yolo model on that basic synth dataset.
    3. Then run that model against the real world YOLO formatted test set and log the initial mAP score to wandb.
    
    
