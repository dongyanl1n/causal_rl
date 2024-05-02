from stable_baselines.a2c.a2c import A2C
from stable_baselines.ppo2.ppo2 import PPO2
from envs.causal_env_v0 import CausalEnv_v0, ABconj, ACconj, BCconj, Adisj, Bdisj, Cdisj
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList, BaseCallback
import argparse
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from functools import partialmethod
import os
import numpy as np
import tensorflow as tf

class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional rewards to TensorBoard. Use for quiz-type where there's quiz reward and type reward.
    """
    def __init__(self, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)
        self.is_tb_set = False

    def _on_step(self) -> bool:
        # Accessing each sub-environment in a vectorized environment
        quiz_rewards = np.mean([env.get_quiz_reward() for env in self.model.env.envs])
        type_rewards = np.mean([env.get_type_reward() for env in self.model.env.envs])

        # Set up the TensorBoard summary logging only once
        if not self.is_tb_set:
            with self.model.graph.as_default():
                self.quiz_reward_ph = tf.placeholder(tf.float32, shape=None, name="quiz_reward")
                self.type_reward_ph = tf.placeholder(tf.float32, shape=None, name="type_reward")
                self.quiz_reward_summary = tf.summary.scalar('quiz_reward', self.quiz_reward_ph)
                self.type_reward_summary = tf.summary.scalar('type_reward', self.type_reward_ph)
                self.summary_op = tf.summary.merge([self.quiz_reward_summary, self.type_reward_summary])
            self.is_tb_set = True

        # Log the values using the placeholders and summaries
        feed_dict = {
            self.quiz_reward_ph: quiz_rewards,
            self.type_reward_ph: type_rewards
        }
        summary = self.model.sess.run(self.summary_op, feed_dict)
        self.locals['writer'].add_summary(summary, self.num_timesteps)

        return True



def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def _get_environments(holdout_strategy: str, quiz_disabled_steps: int = -1, reward_structure: str = 'quiz'):

    def make_env(rank, seed=0, qd=-1):
        def _init():
            if holdout_strategy == 'none':
                env = CausalEnv_v0({"reward_structure": reward_structure, "quiz_disabled_steps": qd})
            elif holdout_strategy == 'conjunctive_train':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        ABconj,
                        ACconj,
                        BCconj,
                    ]})
            elif holdout_strategy == 'disjunctive_train':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        Adisj,
                        Bdisj,
                        Cdisj,
                    ]})
            elif holdout_strategy == 'conjunctive_loo':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        ABconj,
                        ACconj,
                    ]
                })
            elif holdout_strategy == 'disjunctive_loo':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        Adisj,
                        Bdisj,
                    ]
                })
            elif holdout_strategy == 'both_loo':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        ABconj,
                        ACconj,
                        Adisj,
                        Bdisj,
                    ]
                })
            else:
                raise ValueError('Unsupported holdout strategy: {}'.format(holdout_strategy))
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init

    def vec_env(n=4, qd=-1):
        env = DummyVecEnv([make_env(i, qd=qd) for i in range(n)])
        return env

    env = vec_env(4, qd=quiz_disabled_steps)

    if holdout_strategy == 'none':
        eval_env = CausalEnv_v0({"reward_structure": reward_structure})
    elif holdout_strategy == 'conjunctive_train':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                Adisj,
                Bdisj,
                Cdisj,
            ]})
    elif holdout_strategy == 'disjunctive_train':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                ABconj,
                ACconj,
                BCconj,
            ]})
    elif holdout_strategy == 'conjunctive_loo':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                BCconj,
            ]
        })
    elif holdout_strategy == 'disjunctive_loo':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                Cdisj,
            ]
        })
    elif holdout_strategy == 'both_loo':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                Cdisj,
                BCconj,
            ]
        })
    else:
        raise ValueError('Unsupported holdout strategy: {}'.format(holdout_strategy))

    return env, eval_env


def main(args):
    env, eval_env = _get_environments(args.holdout_strategy, args.quiz_disabled_steps, args.reward_structure)

    # Stop training when the model reaches the reward threshold
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=3, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=None, eval_freq=5000, verbose=1, n_eval_episodes=args.num_eval_episodes)  # Using evaluate_policy_save with threshold=2.5
    
    if args.reward_structure == 'quiz-type':
        log_callback = CustomTensorboardCallback()  # Log quiz and type rewards to TensorBoard during quiz-type env
        eval_callback = CallbackList([eval_callback, log_callback])

    if args.policy == 'mlp':
        policy = MlpPolicy
        save_name = f'{args.alg}_{args.policy}'
    elif args.policy == 'mlp_lstm':
        policy = partialclass(MlpLstmPolicy, n_lstm=args.lstm_units)
        save_name = f'{args.alg}_{args.policy}_{args.lstm_units}'
    elif args.policy == 'mlp_lnlstm':
        policy = partialclass(MlpLnLstmPolicy, n_lstm=args.lstm_units)
        save_name = f'{args.alg}_{args.policy}_{args.lstm_units}'
    else:
        raise ValueError('Unsupported policy: {}'.format(args.policy))

    save_name += ('_qd=' + str(args.quiz_disabled_steps)) if args.quiz_disabled_steps > 0 else ''
    save_name += ('_rs=' + str(args.reward_structure))
    save_name += ('_hs=' + str(args.holdout_strategy))
    print(f'Saving model to {save_name}')
    if args.alg == 'a2c':
        model = A2C(policy, env, verbose=1, tensorboard_log="/network/scratch/l/lindongy/causal_overhypotheses/logs/{}".format(save_name))
    elif args.alg == 'ppo2':
        model = PPO2(policy, env, verbose=1, tensorboard_log="/network/scratch/l/lindongy/causal_overhypotheses/logs/{}".format(save_name))
    else:
        raise ValueError('Unsupported algorithm: {}'.format(args.alg))

    model.learn(
        total_timesteps=int(args.num_steps),
        callback=eval_callback
    )
    os.makedirs("/network/scratch/l/lindongy/causal_overhypotheses/model_output/{}".format(save_name), exist_ok=True)
    model.save("/network/scratch/l/lindongy/causal_overhypotheses/model_output/{}/model".format(save_name))

    slurm_tmpdir = os.environ.get('SLURM_TMPDIR', '/tmp')  # where eval trajectories are saved
    os.makedirs("/network/scratch/l/lindongy/causal_overhypotheses/model_output/{}/eval_traj/".format(save_name), exist_ok=True)
    os.system(f"mv {slurm_tmpdir}/* /network/scratch/l/lindongy/causal_overhypotheses/model_output/{save_name}/eval_traj/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--alg', type=str, default='a2c', help='Algorithm to use')
    parser.add_argument('--policy', type=str, default='mlp', help='Policy to use')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units')
    parser.add_argument('--num_steps', type=int, default=int(3000000), help='Number of training steps')
    parser.add_argument('--quiz_disabled_steps', type=int, default=-1, help='Number of quiz disabled steps (-1 for no forced exploration)')
    parser.add_argument('--holdout_strategy', type=str, default='none', help='Holdout strategy')
    parser.add_argument('--reward_structure', type=str, default='quiz', help='Reward structure')
    parser.add_argument('--num_eval_episodes', type=int, default=200, help='Number of evaluation episodes')
    args = parser.parse_args()
    argsdict = args.__dict__
    print(argsdict)
    main(args)