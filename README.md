# GContourPose
### Extension of ContourPose trained on ClearPose dataset of glass objects.

[Click Here for Final Paper](https://github.com/samk271/GContourPose/blob/main/IREP%20-%20Sam%20Klemic%20Final%20Paper.pdf)

Run the following to install all necessary packages
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy matplotlib scipy pillow opencv-python open3d
```
#### Overview and User Guide
- main.py is called to run either training or evaluation
    - Has 5 switches related to training and eval
    - --train will put the code into training mode
        - Every 10 epochs it stores that state of the model to `./trained_models/obj` as `GContourPose_epoch.pkl`
    - --obj specifies which object to train for, searches in `./data/model/`
    - --epochs specifies the number of epochs to run training for
    - --resume says whether or not to resume training
        - Searches in `./trained_models/obj/` for the most recent saved network state to resume training on
    - --blender specifies whether to use blender rendered countours for gt or project densely sampled points. If switched to true searches in `./data/set2/set2/sceneX/renders/` for contours.
        - Currently projected dense points are generated in real time, but to make training faster, making a dedicated dataset of these contours would need to be done
    - If --train is set to False, or isn't included, the code goes into evaluation mode. It will take the most trained model from `./trained_models/obj/` to use.
        - As of right now, the evaluation function is commented out
            - For evaluation, the object trained needs a diameter, there is a function in keypoints.py to find and return it. Put the resulting diameter in `diameter_dictionary.json` formatted as `obj: diameter`.
        - After evaluation, a matplotlib window will appear including:
            - A randomly selected RGB image
            - The ground truth contour and keypoints
            - The prediced keypoints
            - A projection of the predicted pose on the rgb input
            - The overlap between the predicted contour and the ground truth
- The organization of the data should go as follows:
    - `./data/models/obj/` must include `obj_3D_contour.txt`, `obj_keypoints.txt`, `obj_points.txt`, and `obj.ply`. If they do not exist they can be generated using helper functions in keypoints.py
    - `./data/set2/set2/sceneX/data` must include a `rgb` folder with the rgb inputs to train and eval with
    - `./data/set2/set2/sceneX/renders` must include the blender generated renders if you wish to train with them.
    - They can be downloaded, along with their corresponding RGB images from the following link: (https://mega.nz/folder/6coX3IrB#YnjWdBtWh6LX-cOiT8w27w)
    - The renders have an unconventional naming structure:
        - For scene 1: in `./renders/x.png` corresponds to `./data/rgb/((x*10)-1).png`
        - For scene 3: in `./renders/x.png` corresponds to `./data/rgb/(x-1).png`
    - Dataset.py handles associating rgb to render for you
- keypoints.py has useful tools for generating both dense contour points and sparse keypoints
    - `generate_keypoints()` uses FPS to generate the keypoints for each model in `./data/model/` and saves them to `./data/model/obj/obj_keypoints.txt`
    - `count_contour_points()` uses dataset to acquire poses, projects the param obj point cloud in that pose, does edge detection, and counts how often a 3d point falls on an edge. It returns an array with indicies matching the obj point cloud.
    - `corresponding_keypoint_mapper()` does the dense contour point sampling. It uses `count_contour_points()` and saves the array to a file. if the file already exists it reads from it. It then takes every point which fell on an edge over 100 times and stores it in `./data/model/obj/obj_3D_contour.txt`. It also visualizes the resulting dense points, so you can manually verify if the threshold was accurate.
    - `find_diameter` finds the furthest point between any two keypoints for param obj. 

#### Docker
For running on a server or containerizing the code, there is a `Dockerfile` and `gcon_pose.sh` as a shell script.

