# -*- coding: utf-8 -*-
import os
from rl_ppo import A2CPolicyNetwork
import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything
import argparse


def load_model_from_experiment(experiment_folder: str):
    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[1]
    print('checkpoint_path', checkpoint_path)
    model = A2CPolicyNetwork.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    model.freeze()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='new')
    parser.add_argument("--model_path", default="logs_rl/version_10", type=str)
    args = parser.parse_args()

    seed_everything(0, workers=True)
    device = 'cuda:0'
    print("Loading model...")
    model: A2CPolicyNetwork = load_model_from_experiment(args.model_path)
    model.to(device)

    test_counter = []
    for example in tqdm(range(0, 500), mininterval=60):  # context kws target
        finish = False
        dialog_context = model.env.context
        dialog_kws = model.env.kws
        target = model.env.target
        chosen_kws = []
        for i in range(1, 8):
            input_ids = model.state.to(device)
            pi, actions, value, res_pred, state_action_ids = model.forward(
                input_ids,
                input_ids.tolist()
            )
            next_state, reward, done, _ = model.env.step(res_pred)
            model.state = torch.tensor([next_state])
            if done == 1:
                finish = True
                break

        test_counter.append((dialog_context, dialog_kws, finish, target))
        model.env.reset()
        finish = False

    counter = []
    fout = open('test_result/self_play_dialog.txt', 'w')
    for dialog, kws, finish, target in test_counter:
        if finish is True:
            counter.append(len(dialog) / 2)  # context / 2 = turns
        for i in range(0, len(dialog)):
            fout.write(dialog[i] + '\n')
            # fout.write(dialog[i] + ' | ' + str(kws[i]) + '\n')
        fout.write(str(finish) + ' target: ' + target + '\n\n')
    fout.write("sucess rate: ", len(counter) / len(test_counter))
    fout.write("turns: ", str(torch.tensor(counter).float().mean().item()))
    fout.close()

# nohup python -u rl_test.py > common.log 2>&1 &
