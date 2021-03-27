# DexYCB Toolkit

### Prerequisites

This code is tested with Python 3.7.

### Installation

1. Clone the repo with `--recursive` and and cd into it:

    ```Shell
    git clone --recursive ssh://git@gitlab-master.nvidia.com:12051/ychao/dex-ycb-toolkit.git
    cd dex-ycb-toolkit
    ```

2. Install Python package and dependencies:

    ```Shell
    # Install dex-ycb-toolkit
    pip install -e .

    # Install bop_toolkit dependencies
    cd bop_toolkit
    pip install -r requirements.txt
    cd ..

    # Install manopth
    cd manopth
    pip install -e .
    cd ..
    ```

3. Download the DexYCB dataset.

4. Set environment variable for dataset path:

    ```Shell
    export DEX_YCB_DIR=/path/to/dex-ycb
    ```

    `$DEX_YCB_PATH` should be a folder with the following structure:

    ```Shell
    ├── 20200709-weiy/
    ├── 20200813-ceppner/
    ├── ...
    ├── calibration/
    └── models/
    ```

5. Download example results:

    ```Shell
    ./results/fetch_example_results.sh
    ```

6. Download MANO models and codes (`mano_v1_2.zip`) from `https://mano.is.tue.mpg.de` and place the file under `manopath`. Unzip the file and create symlink:

    ```Shell
    cd manopth
    unzip mano_v1_2.zip
    cd mano
    ln -s ../mano_v1_2/models models
    cd ../..
    ```

### Running examples

1. Creating dataset:

    ```Shell
    python examples/create_dataset.py
    ```

2. Visualizing object and hand pose of one image sample:

    ```Shell
    python examples/visualize_pose.py
    ```

3. COCO evaluation:

    ```Shell
    python examples/evaluate_coco.py
    ```

4. BOP evaluation:

    ```Shell
    python examples/evaluate_bop.py
    ```

5. HPE evaluation:

    ```Shell
    python examples/evaluate_hpe.py
    ```

6. Grasp visualization:

    ```Shell
    python examples/visualize_grasps.py
    ```

7. Grasp evaluation:

    ```Shell
    python examples/evaluate_grasp.py
    python examples/evaluate_grasp.py --visualize
    PYOPENGL_PLATFORM=egl python examples/evaluate_grasp.py --visualize
    ```

8. Viewing a sequence:

    ```Shell
    python examples/view_sequence.py --name 20200709-weiy/20200709_141754
    python examples/view_sequence.py --name 20200709-weiy/20200709_141754 --device cpu
    python examples/view_sequence.py --name 20200709-weiy/20200709_141754 --no-preload
    ```

9. Rendering a sequence:

    ```Shell
    python examples/render_sequence.py --name 20200709-weiy/20200709_141754
    PYOPENGL_PLATFORM=egl python examples/render_sequence.py --name 20200709-weiy/20200709_141754
    ```
