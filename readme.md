# Dataset Generation and Finetuning
## Scene Generation
Here's how you can generate the training scenes. First:
```bash
cd SceneGen/images/
python generate_scene.py
```
By default, this script will generate 100 base scene, and place buses on 15 different positions in each scene. So a total of 1500 scenes will be generated. Among each group of 15 pictures, 6 of them won't contain people. The generation parameters can be tuned inside the code, for example `NUM_BASE_SCENES = 100`, `BUS_VARIANTS_PER_SCENE = 15`,`NO_PERSON_VARIANTS_PER_SCENE = 6`.

Note that this script won't annotate the benches and animal groups. To do this, please
```bash
cd SceneGen/images/
python annotate_ids.py
```
Then you'll get a set of training images with labeled benches and animal crowds.

## GRAID Q/A pairs generation
Now you have the training images prepared, the next step is to create corresponding Q/A pairs w.r.t each image. To do this, first go to the GRAID folder:
```bash
cd GRAID/graid/
```

Then install the necessary environment:

0. Install uv (optional if you already have it): `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))
1. Create a virtual environment: `uv venv`
2. Activate it: `source .venv/bin/activate` (or use direnv with the provided .envrc)
3. Install dependencies: `uv sync`
4. Install all backends: `uv run install_all`

The configuration of generation is written in `zoo_bus_config.json`, where you can find the source image path and the questions to be generated. To use GRAID to generate data, run:
```bash
python run_zoo_bus.py zoo_bus_config.json
```
This will gengerate the HuggingFace format dataset. To convert the dataset into trainable image/QA pairs, run:
```bash
python export_dedup_jsonl.py \
  --dataset_dir datasets/zoo_bus_vqa \
  --split train
```
Then, a `train.jsonl` file and `a folder with source images` will be generated in the current file.

## Model finetuning
To finetune the llava model, first copy `the folder with source images` and the `train.json` into the `fineTune` folder. Then follow the `fineTuning.ipynb` for model finetuning and evaluation. Specifically, the finetuning script is `train_llava_next_fast.py`, 
and the evaluation script is `eval_before_after.py`.