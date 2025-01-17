# MyoChallenge

Code repository for [MyoChallenge](https://sites.google.com/view/myochallenge) by team _stiff fingers_. We have documented a summary of our approach including our key insights and intuitions along with all the training steps and hyperparameters [here](docs/summary.md).

## Requirements

Listed in `requirements.txt`. Note that there is a version error with some packages, e.g. `stable_baselines3`, requiring later versions of `gym` which `myosuite` is incompatible with. If your package manager automatically updates gym, do a `pip install gym==0.13.0` (or equivanlent with your package manager) at the end and this should work fine. If you experience any issues, feel free to open an issue on this repository or contact us via email.

## Usage

Run `python src/main_baoding.py` to start a training. Note that this starts training from one of the pre-trained models in our curriculum. You can find all the trained models along with the scripts used to train them and the environment configurations [here](trained_models). The full information about the training process can be found in the [summary](docs/summary.md).

To evaluate the best single policy network (see the [summary](docs/summary.md)), run `python src/main_eval.py`. To evaluate the final ensemble (55% score), run `python src/eval_mixture_of_ensembles.py`.
