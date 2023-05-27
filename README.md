# Stable Diffusion
Implementation of stable diffusion in pytorch

This repo implements the text-to-image model stable diffusion in pytorch. The code uses pretrained CLIPTextModel and CLIPToeknizer from huggingface with the rest models trained from scratch on [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) dataset.

## Prepare and running

to build env for your run, just simply create a conda env with python>3.7 recommended, and install the packages:

    git clone https://github.com/lwb2099/stable_diffusion_pytorch.git
    pip install -r requirements.txt

after which you can simply run by passing 
    
    accelerate launch --config_file place_for_your_yaml_file train.py --train_args

I pushed my vscode `launch.json` so that you can modify command line arguments more easily.

## Structure
The structure of the code has drew insight from a few awesome repositories: [fairseq](https://github.com/facebookresearch/fairseq), [transformers](https://github.com/huggingface/transformers) and it should looks like this:

    |-data
    |--dataset
    |--pretrained
    |-model
    |--checkpoint-{step/epoch}
    |-scripts
    |-stable_diffusion
    |--config
    |--models
    |--modules
    |-test
    |-utils
`data` stores downloaded dataset in `data/dataset` and pretrained CLIP model from [huggingface/models](https://huggingface.co/models) in `data/pretrained`. 

`model` is used to store training ckpts with name "checkpoint-{step}" if passed "--checkpointing-steps step" or "epoch-{epoch}" if passed "--checkpointing-steps epoch". 

`scripts` places code like txt2img.py to sample iamge, which still remains to be finished

`test` contains scripts to test code, currently only args because packages structure and import still confuses me

`utils` has helpful scripts for a successful run, includeing ckpt handling, model&data loading and arg parsing.

`stable_diffusion` is the main package that stores everything to build a model. `config` stores yaml files created by "accelerate config" command line, `models` stores assembled models while `module` contains nessecery blocks to build them.

## Problems remaining
Though it can possibly run successfully, several problems yet still remains to be solved.(or just things I have not figured out), and any guidance is appreciated

- I have searched the web yet python package dependencies and import rules, I think its a better way to learn in practice. 
- Structure of this repo combines transformers and fairseq together, but I'm seeking a better structure for smaller projects.
- Though I'v used dataclass, it is clearly a better way to build a model through config json file, and I'm 

## Reference:
Thanks to the following amazing repositories that helped me build this code:

### Model implementation:
[origin stable diffusion github](https://github.com/CompVis/stable-diffusion)

[labmlai annotated deep learning paper implementation](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion)

### Training script:
modified from [huggingface/diffusers](https://github.com/huggingface/diffusers)

### Arg parsing:
modify script from [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) 

### Diffusion Code Tutorial: 
[dome272/Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch)

[Aleksa's youtube tutorial](https://www.youtube.com/watch?v=y7J6sSO1k50&t=3197s)

More detailed references and links to .py files are in comments