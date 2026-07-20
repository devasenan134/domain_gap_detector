## 1. Problem Statement
    1. Back in hyperverge, we use to go through the failed samples from production as much as possible. And try to identify the failure cases.
    2. Then, we will try to generate those sample types and include them in our next training iteration.
    3. With that we'll test that version. If it covers those failed cases. We are good. If not, we would again do this whole process again.

    So, here instead of this training testing loop.
    I have tried to skip the training step and try quantify the distributional gap between the real and the synthetic dataset.

    --> In HV, this was my workflow, for all the projects i worked on. Like tamper detection, quality check model, signature detection.


## 2. Task and Data
