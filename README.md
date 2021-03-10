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
    pip install -e .
    ```

    and dependencies for `bop_toolkit`:

    ```Shell
    cd bop_toolkit
    pip install -r requirements.txt
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

### Running examples

1. Creating dataset:

    ```Shell
    python examples/create_dataset.py
    ```

2. Visualizing:

    ```Shell
    python examples/visualize.py
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
