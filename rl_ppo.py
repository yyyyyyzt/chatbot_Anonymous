import argparse
from typing import Callable, Iterator, List, Tuple
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, progress
import numpy as np
from rl_env import Env
from basemodel_gpt import End2EndModel
from load_utils import get_cos_similar_multi
import random


class Critic(nn.Module):
    def __init__(self, base_model=None, hidden_size=768):
        super().__init__()
        # self.base_model = base_model
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states, attention_mask=None):
        return self.critic_net(states)  # last_embed
        outputs = self.base_model(
            input_ids=states,
            attention_mask=attention_mask,
        )
        last_embed = outputs.hidden_states[-1][:, -1, :]  # n * 768
        return self.critic_net(last_embed)  # last_embed


class ActorContinous(nn.Module):
    """Policy network, for continous action spaces, which returns a distribution and an action given an
    observation."""

    def __init__(self, base_model, act_dim):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()
        self.base_model: End2EndModel = base_model
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float)
        self.log_std = nn.Parameter(log_std)

    def forward(self, action_ids, attention_mask=None):
        outputs = self.base_model.decoder(
            action_ids,
            attention_mask=attention_mask,
        )  # outputs.hidden_states 61 768
        actions = outputs.hidden_states[-1][:, -1, :]
        # mu = outputs.hidden_states[-1][:, -2, :]
        # mask = torch.zeros(action_ids.shape)
        # for i, row in enumerate(raw_ids):
        #     mask[i][len(row)] = 1
        # mu = outputs.hidden_states[-1][mask.bool()]
        std = torch.exp(self.log_std)
        # pi = Normal(loc=mu, scale=std)  # 取中间状态为均值，生成后的结果为action
        pi = Normal(loc=actions, scale=std)
        # actionss = pi.sample()
        return pi, actions

    def get_log_prob(self, pi: Normal, actions: torch.Tensor):
        """Takes in a distribution and actions and returns log prob of actions under the distribution.
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions).sum(axis=-1)


class ExperienceSourceDataset(IterableDataset):
    """Implementation from PyTorch Lightning Bolts: https://github.com/PyTorchLightning/lightning-
    bolts/blob/master/pl_bolts/datamodules/experience_source.py.

    Basic experience source dataset. Takes a generate_batch function that returns an iterator. The logic for the
    experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self):
        iterator = self.generate_batch()
        return iterator


class A2CPolicyNetwork(pl.LightningModule):
    def __init__(
        self,
        base_path: str,
        disc_path: str,
        data_path: str,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        max_episode_len: float = 200,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__()

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.save_hyperparameters()

        self.base_model: End2EndModel = End2EndModel.load_from_checkpoint(base_path)
        self.env = Env(self.base_model, disc_path, data_path)

        self.actor = ActorContinous(self.base_model, 768)
        self.critic = Critic(self.base_model)

        self.batch_states = []
        self.batch_states_action = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = torch.tensor(self.env.reset()).unsqueeze(0)

    def forward(self, states, raw_ids=None):
        """Passes in a state x through the network and returns the policy and a sampled action.

        Args:
            x: environment : context candidate target

        Returns:
            Tuple of policy and action
        """
        # mask = states.ne(-100).long()
        # if raw_ids is None:  # 只有state传入
        #     raw_ids = []
        #     for row in states.tolist():
        #         if -100 in row:
        #             row = row[:row.index(-100)]
        #         raw_ids.append(row)
        #     states = torch.where(states < 0, self.base_model.dec_tok.k_start, states)
        keyword_out = self.base_model.decoder.generate(
            input_ids=states,
            max_new_tokens=5,
            pad_token_id=self.base_model.dec_tok.k_end,
            eos_token_id=self.base_model.dec_tok.k_end,
            num_beams=3,
            top_k=40,
            top_p=0.95,
            length_penalty=1.0,
            early_stopping=True
        )
        seq_len = states.shape[1] - 1
        key_pred = self.base_model.dec_tok.decode(keyword_out[0, seq_len:], skip_special_tokens=True).replace(' ', '')
        if key_pred not in self.base_model.find_path.graph_embed:
            key_pred = self.env.target
        words, embeds = list(self.env.sub_g_words.keys()), list(self.env.sub_g_words.values())
        faith_score = get_cos_similar_multi(self.base_model.find_path.graph_embed[key_pred], embeds).reshape(-1)
        words_list = [(w, round(cos, 4)) for w, cos in zip(words, faith_score)]
        words_list = sorted(words_list, key=lambda pair: pair[1], reverse=True)
        key_pred = words_list[0][0]
        if key_pred != self.env.target:
            del self.env.sub_g_words[key_pred]

        self.env.kws.append(key_pred)
        eos_token = self.base_model.dec_tok.eos_token
        keyword_input_ids = self.base_model.dec_tok.encode(key_pred) + raw_ids[0][raw_ids[0].index(50256):]
        # keyword_input_ids = raw_ids[0]
        keyword_input_ids += self.base_model.dec_tok.encode(key_pred + '<e_k>' + eos_token)
        response_out = self.base_model.decoder.generate(
            input_ids=torch.tensor([keyword_input_ids]).to(self.device),
            max_new_tokens=30,
            pad_token_id=50256,
            eos_token_id=50256,
            num_beams=3,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True
        )
        res_out = response_out[0, len(keyword_input_ids)-1:]
        res_pred = self.base_model.dec_tok.decode(res_out, skip_special_tokens=True)
        state_action_ids = [raw_ids[0] + self.base_model.dec_tok.encode(key_pred + '<e_k>' + eos_token + res_pred + eos_token)]
        # state_action_ids = [keyword_input_ids + self.base_model.dec_tok.encode(res_pred + eos_token)]
        state_action_ids = torch.tensor(state_action_ids).to(self.device)
        pi, actions = self.actor(state_action_ids)
        value = self.critic(actions)
        # value = self.critic(action_ids, attention_mask)  # input_ids or action_ids
        return pi, actions, value, res_pred, state_action_ids

    def discount_rewards(self, rewards: List[float], discount: float):
        """Calculate the discounted rewards of all rewards in list.

        Args:
            rewards: list of rewards/advantages

        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float):
        """Calculate the advantage given rewards, state values, and the last value of episode.

        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode

        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def generate_trajectory_samples(self):

        for step in range(self.steps_per_epoch):

            with torch.no_grad():
                pi, action, value, response, state_action_ids = self(self.state.to(self.device), self.state.tolist())
                log_prob = self.actor.get_log_prob(pi, action)

            next_state, reward, done, _ = self.env.step(response)

            self.episode_step += 1
            self.batch_states_action.append(state_action_ids[0])
            self.batch_states.append(self.state[0])  # tensor state
            self.batch_actions.append(action[0])
            self.batch_logp.append(log_prob[0])

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = torch.tensor(next_state).unsqueeze(0)

            epoch_end = step == (self.steps_per_epoch - 1)  # 一个 epoch 中可以玩好几轮游戏
            terminal = len(self.ep_rewards) == self.max_episode_len  # 最大游戏次数/对话轮数

            # if epoch_end:
            #     print('\n target ', self.env.target, self.env.context[:2])
            #     for c, r in zip(self.env.context[2:], self.ep_rewards):
            #         print(r, c)

            if epoch_end or done or terminal:
                # if random.randint(0, 50) == 1:
                # print('\n target ', self.env.target, self.env.context[:2])
                # for c, r in zip(self.env.context[2:], self.ep_rewards):
                #     print(r, c)
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    with torch.no_grad():
                        _, _, value, _, _ = self(self.state.to(self.device), self.state.tolist())
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(self.ep_rewards + [last_value], self.gamma)[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = torch.tensor(self.env.reset()).unsqueeze(0)

            if epoch_end:
                b_s = torch.nn.utils.rnn.pad_sequence(self.batch_states_action, True,  -100).to(self.device)
                train_data = zip(b_s, self.batch_actions, self.batch_logp, self.batch_qvals, self.batch_adv)

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                # print("-" * 30)
                # print(old_state['context'])
                # print("-" * 30)
                # print(old_state['history_word'])
                # print("-" * 30)
                # print(old_state['target'])
                # print("-" * 30)
                self.batch_states_action.clear()
                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (self.steps_per_epoch - steps_before_cutoff) / nb_episodes

                self.epoch_rewards.clear()

    def actor_loss(self, state, action, logp_old, qval, adv):
        mask = state.ne(-100).long()
        state = torch.where(state < 0, 50256, state)
        pi, _ = self.actor(state, mask)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, action, logp_old, qval, adv):
        mask = state.ne(-100).long()
        state = torch.where(state < 0, 50256, state)
        _, actions = self.actor(state, mask)
        value = self.critic(actions)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx, optimizer_idx):
        state, action, old_logp, qval, adv = batch
        # action = action.reshape(-1)
        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        self.log("avg_ep_len", float(self.avg_ep_len), prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_ep_reward", float(self.avg_ep_reward), prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_reward", float(self.avg_reward), prog_bar=True, on_step=False, on_epoch=True)

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, qval, adv)
            self.log("loss_actor", loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return loss_actor

        if optimizer_idx == 1:
            loss_critic = self.critic_loss(state, action, old_logp, qval, adv)
            self.log("loss_critic", loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_critic

    def configure_optimizers(self):
        """Initialize Adam optimizer."""
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic for each data
        sample."""
        for _ in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def collate_fn(self, batch):
        states = []
        actions = []
        logp_olds = []
        qvals = []
        advs = []
        for state, action, logp_old, qval, adv in batch:
            states.append(state)
            actions.append(action)
            logp_olds.append(logp_old)
            qvals.append(qval)
            advs.append(adv)
        return states, actions, logp_olds, torch.tensor(qvals), torch.tensor(advs)

    def _dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(dataset=dataset, shuffle=False, num_workers=0, batch_size=self.batch_size)
        # dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return dataloader

    def train_dataloader(self):
        """Get train loader."""
        return self._dataloader()

    @ staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("A2CPolicyNetwork")
        parser.add_argument("--base_path", type=str, default='logs_base/cv2/checkpoints/best.ckpt')  # cv1 dd1
        parser.add_argument("--disc_path", type=str, default='logs_discri/cv2')  # cv_cos_best cv_reasoning_best dd_reasoning_best  dd_cos_best
        parser.add_argument("--data_path", type=str, default='data/process/convai/test_good_case.json')  # daily_dialogue  convai
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--lam", type=float, default=0.95, help="advantage discount factor")
        parser.add_argument("--lr_actor", type=float, default=1e-5, help="learning rate of actor network")
        parser.add_argument("--lr_critic", type=float, default=1e-4, help="learning rate of critic network")
        parser.add_argument("--max_episode_len", type=int, default=16, help="capacity of the replay buffer")
        parser.add_argument("--batch_size", type=int, default=32, help="batch_size when training network")
        parser.add_argument(
            "--steps_per_epoch",
            type=int,
            default=128,
            help="how many action-state pairs to rollout for trajectory collection per epoch",
        )
        parser.add_argument(
            "--nb_optim_iters", type=int, default=4, help="how many steps of gradient descent to perform on each batch"
        )
        parser.add_argument(
            "--clip_ratio", type=float, default=0.2, help="hyperparameter for clipping in the policy objective"
        )

        return parent_parser


if __name__ == "__main__":
    pl.seed_everything(0, workers=True)
    tb_logger = pl.loggers.TensorBoardLogger('logs_rl/', name='')
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_last=True,
        verbose=True,
        filename='best',
        monitor='avg_ep_reward',
        mode='max'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=300)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parser = A2CPolicyNetwork.add_model_specific_args(parent_parser)
    args = parser.parse_args()
    model = A2CPolicyNetwork(**vars(args))
    # model = A2CPolicyNetwork(**vars(args)).load_from_checkpoint(
    #     'logs_rl/version_9/checkpoints/last.ckpt',
    # )
    trainer = pl.Trainer(
        gpus=1,
        # gpus=None,
        min_epochs=1,
        max_epochs=50,
        logger=tb_logger,
        detect_anomaly=True,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, bar_callback],
        log_every_n_steps=300,
    )
    trainer.fit(model)

# nohup python -u rl_a2c.py > rl_a2c_cv_cos.log 2>&1 &
# nohup python -u rl_a2c.py > rl_a2c_cv_rea.log 2>&1 &
# nohup python -u rl_a2c.py > rl_a2c_dd_cos.log 2>&1 &
# nohup python -u rl_a2c.py > rl_a2c_dd_rea.log 2>&1 &
