# Intraoperative Registration via Cross-Modal Inverse Neural Rendering
This repository is the accompanying code to the paper "Intraoperative Registration via Cross-Modal Inverse Neural Rendering". It contains the code for the multistyle NeRF and the pose estimation method. Our NeRF is based on the Nerfstudio implementation of Instant-NGP and the actual registration is done using Parallel Inversion. Since there is no pytorch implementation of a SOTA solver, we use the NeRF to infer the target style and retrain the model in the Parallel Inversion environment to register that NeRF to the intraoperative image.

## Requirements

- NVIDIA GPU

## Installation

To configure the environment to work with Nerfstudio, Parallel Inversion, and other dependencies, follow the instructions below.
```bash
cd nerfstudio-method-template
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 conda env create -f environment.yaml
conda activate style-ngp
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Register the method in Nerfstudio to later run our method in the Nerfstudio environment.
```bash
pip install -e .
ns-install-cli
```

## Getting Started

This section provides a simple example and some toy data for you to play with. It describes how to interact with Nerfstudio, including commands like `ns-train` and `ns-render`.
Execute all commands from the repo root.

## Training the NeRF

Train the NeRF & hypernet in the Nerfstudio environment. Follow instructions in the terminal to observe training via localhost.
```bash
ns-train style-ngp --data data/datasets/000_mra
```
or **recommended**: learn structure also from one of the synthetic craniotomy datasets. Training the density/structure part of the NeRF on the MR data is possible but will not yield the most pleasing visual results as the vessels are much more salient than in the (synthetic) craniotomy images that we learn appearance from. Therefore, we recommend to pick one of the synthetic craniotomy datasets to learn structure from.
```bash
ns-train style-ngp --data data/datasets/0_065_cat5_2.0
```
Tipps:
- Zoom slightly in and out to get frequent updates
- You can click on training images to move the camera to that POV. To replicate the intrinsics of the virtual camera that was used to generate the training data, go to 'Render' and set the 'Default FOV' to 30 degrees.
- Note how after some number of steps, the NeRF tumbles into visual chaos and then switches through styles. This is the transition between the 2 training stages when the density will be fixed, but now we flick through multiple datasets and style images and the hypernetwork tries to set the weights of the RGB network within the NeRF to reconstruct the corresponding images in that dataset.
- The file `style_ngp_pipeline.py` gives you control over how long the density training should be (1st stage) and how many steps the hypernetwork should be trained on each style (2nd stage).
- The file `style_ngp_config.py` lets you control how many steps in total to train for.


## Inspect the Model (after training)
Inspect the model on test data. Note that you can control styles/appearances in the UI! If you scroll down, you can choose 'Style' in 'Custom Elements'. Do not change this during training! You should run all the following commands from the root of the repo.

Important notes:
- To switch from training to test mode, you need to set self.train_styles = False in style_ngp_pipeline.py and vice versa. Otherwise, you can only select styles from the training distribution in the UI.
- As of now, which folders constitute the training and test datasets (and styles/textures) is hardcoded in style_ngp_pipeline.py and can be changed there.
```bash
ns-viewer --load-config <path to config of model, shown when training finishes>
```

## Perform Registration
Since there is no SOTA solver implemented in PyTorch, we need to run the target style through our NeRF, regenerate the training dataset with that inferred style, and retrain instant-ngp in the Parallel Inversion environment.
1. Based on the test style, generate a training set to retrain a model in the Parallel Inversion environment. The dataset will consists of images taken from the same poses as the training sets but now with the appearance infered by the hypernetwork based on the test style. Again, make sure to set self.train_styles = False and the dataset will be generated of the first test style, in case there are several. The rendering might take a while.
```bash
ns-render dataset --load-config <path to config yaml that is given after training> --output-path rendered_dataset --split train+test --image-format png --rendered-output-names rgb
mkdir rendered_dataset/images
cp -r rendered_dataset/train/rgb/* rendered_dataset/images && rm -r rendered_dataset/train
cp -r rendered_dataset/test/rgb/* rendered_dataset/images && rm -r rendered_dataset/test
```
2. Convert the dataset to the Parallel Inversion format. This will create a new transforms file that gives the camera poses in the format desired by the Parallel Inversion environment. We do that on the test set (as defined in `style_ngp_pipeline.py`) and then move it in the same folder as the rendered images to form a full NeRF dataset.
```bash
python data_generation/nerf_coords_transform.py --transforms_file data/datasets/0_207_cat5_2.0/transforms.json
mv data/datasets/0_207_cat5_2.0/transforms_ngp.json rendered_dataset/transforms.json
``` 
3. Get some dependencies that Parallel Inversion requires.
```bash
git submodule update --init --recursive
```
4. Setup another conda environment as described in the Parallel Inversion submodule (see their Readme). Do not forget to go through the pain of also completing the build, otherwise you will not have modules that the python scripts require (e.g. pyngp). In our case, that required lots of experimenting with cmake, setting for example -DOPENGL_opengl_LIBRARY explicity etc. Also, we had to deactivate the conda environment temporarily for the build so that conda packages do not clash with your system ones.
5. Navigate to the scripts folder in the submodule and run Parallel Inversion. This will train a new model based on the dataset and then register the NeRF to the intraoperative image. Note that I already provided the json file with a single target with the pose taken from the rerendered dataset and its transforms that you create in step 1 and step 2.
```bash
python run_pose_refinement.py --config config/example/example.yaml --targets_json_path ../../data/example_registration.json --render_aabb_scale 32
```

# Detailed Overview (Optional)
This section provides a full overview of the project and is intended to make your life a bit easier. However, above code snippets and config files that are used in them can be sufficient to just dig into the code straightaway. This section adds some details on data generation, registration via Parallel Inversion, and evaluation.

## Data Generation

### `generate_dataset.py`
Generates a NeRF dataset of images and poses from a mesh and textures. This will yield the data in the format we already provided as an example (see `data`) with each dataset having a set of images and a transforms.json file indicating the poses.
Example usage:
```bash
TODO
```

### `generate_gt.py`
Allows for manual creation of a 6 DoF ground truth pose given a target image, a mesh (to be aligned to the target image), and a texture (to texture the mesh to help compare alignment with the target image).
Example usage:
```bash
TODO
```

### `manipulate_target.py`
Adjusts the color distribution of an image to match the color distribution of a folder of images.
Example usage:
```bash
TODO
```

#### `utils.py`
Contains utility functions specific to data generation.

## Style NGP

Contains our models. Check the Nerfstudio description of what fields, models, pipelines etc. are. We are mostly following their guide on how to implement models in their environment. We have to break with conventions in some cases though, e.g., to switch datasets during training.

### `field_components.py`
Contains several feature extractors whose output serves as input to the hypernetwork; HistogramExtractor was used in the paper.

### `style_ngp_config.py`
Some configs, most importantly number of iterations for training.

### `style_ngp_field.py`
Describes model architecture as well as behavior for switching to hypernetwork etc.

### `style_ngp_model.py`
Initializes model, using field (see `style_ngp_field`) and config (see `style_ngp_config`).

### `style_ngp_pipeline.py`
Controls main logic for training the NeRF and the hypernetwork, e.g., switching the style images and the corresponding datasets during training. Also defines where to find the different datasets, which input to the hypernetwork corresponds to which NeRF training set, and defines UI control.

### `template_datamanager.py`
Standard datamanager.

### `utils`
E.g., ImageNet preprocessing.

## Registration with Parallel Inversion
TODO

## Evaluation (eval), includes preparation of the data for registration with Parallel Inversion (e.g. coordinate system conversions)

### `compute_psnr.py`
Computes PSNR between two folders of images (assuming they correspond and reading them with os maintains the correspondence). Used here to compare to compare e.g. the test set generated directly from the stylized mesh against what NeRF renders with `ns-render` for the same poses when feeding the respective style image/texture.
Example usage:
```bash
TODO
```

### `generate_targets_and_inits.py`
Given a mesh and texture, this script generates registration target images (with poses) as well as perturbed poses to use for initialization of the pose solver. We used this to get a larger synthetic test set for registration as each clinical case in our MICCAI dataset only comes with one craniotomy image.
Example usage:
```bash
TODO
```

### `nerf_coords_transform.py`
This becomes necessary when performing registration using Parallel Inversion. We follow the heuristics they are using, which transforms the poses. The script can do that for a `transforms.json` describing a dataset but also for a ground truth pose file (see output of `generate_gt`) and a file containing registration targets and init poses (see output of `generated_targets_and_inits`).
Example usage:
```bash
TODO
```

### `pose_eval.py`
Given the results from Parallel Inversion and the mesh, this will compute the ADD error by applying predicted and ground truth pose to the mesh and computing average pairwise distance of the vertices. The result can also be visualized. See args for more details, a few more tricks necessary to go back from instant-ngp NVIDIA convention to VTK where the original scan and mesh live.
Example usage:
```bash
TODO
```

### `pose_registration.py`
Visualizes camera pose estimation results (init, pred over time, and gt pose) in the coordinate system of the mesh.
Example usage:
```bash
TODO
```

### `synthetic_target_eval.py`
Given the results from Parallel Inversion, this evaluates the synthetic targets generated by `generated_targets_and_inits.py`.
Example usage:
```bash
TODO
```

#### `utils.py`
Contains utility functions specific to evaluation.
