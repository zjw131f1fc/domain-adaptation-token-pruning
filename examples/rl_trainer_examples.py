"""BasicTianshouTrainer 使用示例

演示如何使用BasicTianshouTrainer进行强化学习训练。

示例1: 使用内置DQN算法（最简单）
示例2: 使用自定义Policy
示例3: 使用自定义环境和评估函数
"""

import torch
import gymnasium as gym
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from engine.configs.loader import load_config
from engine.trainers.loader import load_trainer
from engine.managers.loader import load_manager


# ==================== 示例1: 最简单的使用（内置算法+标准gym环境） ====================

def example1_simple_dqn():
    """最简单的DQN训练示例"""

    def preload_fn(config):
        """预加载（可以为空）"""
        return {}

    def run_fn(config, cache):
        """运行训练"""
        logger = config["logger"]
        logger.info("=" * 60)
        logger.info("示例1: 使用内置DQN算法")
        logger.info("=" * 60)

        # Step 1: 创建trainer
        trainer = load_trainer(config)

        # Step 2: 创建环境
        env_name = "CartPole-v1"
        train_envs = DummyVectorEnv([
            lambda: gym.make(env_name) for _ in range(10)
        ])
        test_envs = DummyVectorEnv([
            lambda: gym.make(env_name) for _ in range(1)
        ])

        # Step 3: 注册环境
        trainer.register_envs(train_envs, test_envs)

        # Step 4: 一键训练（policy会根据config自动创建）
        result = trainer.run()

        logger.info(f"训练完成! 最佳奖励: {result['best_reward']:.2f}")

        return result

    # 加载配置并启动
    config = load_config(override_file="configs/rl_dqn_example.yaml")

    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=None,
        result_handler_fn=None
    )

    manager.start()
    manager.wait()

    summary = manager.get_summary()
    print(f"训练结果: {summary}")


# ==================== 示例2: 使用自定义Policy ====================

def example2_custom_policy():
    """使用自定义Policy的示例"""

    def preload_fn(config):
        return {}

    def run_fn(config, cache):
        from tianshou.policy import DQNPolicy
        from tianshou.utils.net.common import Net

        logger = config["logger"]
        logger.info("=" * 60)
        logger.info("示例2: 使用自定义Policy")
        logger.info("=" * 60)

        # Step 1: 创建trainer
        trainer = load_trainer(config)

        # Step 2: 创建环境
        env_name = "CartPole-v1"
        train_envs = DummyVectorEnv([
            lambda: gym.make(env_name) for _ in range(10)
        ])
        test_envs = DummyVectorEnv([
            lambda: gym.make(env_name) for _ in range(1)
        ])
        trainer.register_envs(train_envs, test_envs)

        # Step 3: 创建自定义Policy
        obs_space = train_envs.observation_space
        act_space = train_envs.action_space

        # 自定义网络结构
        net = Net(
            state_shape=obs_space.shape,
            action_shape=act_space.n,
            hidden_sizes=[256, 256, 128],  # 更大的网络
            device=config["global_settings"]["device"]
        )

        optim = torch.optim.Adam(net.parameters(), lr=0.0005)

        # 自定义DQN超参数
        policy = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.95,
            estimation_step=5,
            target_update_freq=500
        )

        # Step 4: 注册自定义policy
        trainer.register_policy(policy)

        # Step 5: 训练
        result = trainer.run()

        logger.info(f"训练完成! 最佳奖励: {result['best_reward']:.2f}")

        return result

    # 注意：config中的algorithm应设为"custom"
    config = load_config(override_file="configs/rl_dqn_example.yaml")
    config["policy_settings"]["algorithm"] = "custom"  # 使用自定义policy

    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=None,
        result_handler_fn=None
    )

    manager.start()
    manager.wait()


# ==================== 示例3: 自定义环境 + 自定义评估 ====================

def example3_custom_env_and_eval():
    """使用自定义环境和评估函数的示例"""

    # 自定义环境（示例：带数据集的环境）
    class CustomEnv(gym.Env):
        """自定义环境示例"""

        def __init__(self, split="train", level=0):
            super().__init__()
            self.split = split
            self.level = level

            # 定义观察空间和动作空间
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
            self.action_space = gym.spaces.Discrete(2)

            self.state = None
            self.steps = 0
            self.max_steps = 200

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.state = self.observation_space.sample()
            self.steps = 0
            return self.state, {}

        def step(self, action):
            self.steps += 1

            # 简单的奖励逻辑
            reward = 1.0 if action == 1 else 0.5

            # 终止条件
            terminated = self.steps >= self.max_steps
            truncated = False

            # 下一个状态
            self.state = self.observation_space.sample()

            return self.state, reward, terminated, truncated, {}

    def preload_fn(config):
        return {}

    def run_fn(config, cache):
        logger = config["logger"]
        logger.info("=" * 60)
        logger.info("示例3: 自定义环境 + 自定义评估")
        logger.info("=" * 60)

        # Step 1: 创建trainer
        trainer = load_trainer(config)

        # Step 2: 创建自定义环境（train和test使用不同配置）
        train_envs = DummyVectorEnv([
            lambda i=i: CustomEnv(split="train", level=i)
            for i in range(10)
        ])

        test_envs = DummyVectorEnv([
            lambda: CustomEnv(split="test", level=0)
            for _ in range(1)
        ])

        trainer.register_envs(train_envs, test_envs)

        # Step 3: 注册自定义评估函数
        def custom_eval(collector_result, device, info):
            """自定义评估指标

            参数:
                collector_result: Tianshou collector的返回结果
                device: torch设备
                info: 包含config、policy等信息的字典

            返回:
                自定义指标字典
            """
            # 从collector_result提取数据
            rewards = collector_result.returns
            lengths = collector_result.lens

            # 计算自定义指标
            success_rate = float((rewards > 100).sum() / len(rewards))
            avg_reward = float(rewards.mean())
            max_reward = float(rewards.max())

            return {
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "max_reward": max_reward,
                "avg_episode_length": float(lengths.mean())
            }

        trainer.register_eval_step(custom_eval)

        # Step 4: 训练
        result = trainer.run()

        logger.info(f"训练完成!")
        logger.info(f"  - 最佳奖励: {result['best_reward']:.2f}")
        logger.info(f"  - 成功率: {result.get('success_rate', 0):.2%}")

        return result

    config = load_config(override_file="configs/rl_dqn_example.yaml")

    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=None,
        result_handler_fn=None
    )

    manager.start()
    manager.wait()


# ==================== 示例4: PPO算法 ====================

def example4_ppo():
    """使用内置PPO算法的示例"""

    def preload_fn(config):
        return {}

    def run_fn(config, cache):
        logger = config["logger"]
        logger.info("=" * 60)
        logger.info("示例4: 使用内置PPO算法")
        logger.info("=" * 60)

        trainer = load_trainer(config)

        env_name = "CartPole-v1"
        train_envs = DummyVectorEnv([
            lambda: gym.make(env_name) for _ in range(10)
        ])
        test_envs = DummyVectorEnv([
            lambda: gym.make(env_name) for _ in range(1)
        ])

        trainer.register_envs(train_envs, test_envs)

        result = trainer.run()

        logger.info(f"训练完成! 最佳奖励: {result['best_reward']:.2f}")

        return result

    # 使用PPO配置
    config = load_config(override_file="configs/rl_ppo_example.yaml")

    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=None,
        result_handler_fn=None
    )

    manager.start()
    manager.wait()


if __name__ == "__main__":
    import sys

    print("\n选择要运行的示例:")
    print("1. 简单DQN（内置算法）")
    print("2. 自定义Policy")
    print("3. 自定义环境和评估")
    print("4. PPO算法")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请输入选项 (1-4): ").strip()

    if choice == "1":
        example1_simple_dqn()
    elif choice == "2":
        example2_custom_policy()
    elif choice == "3":
        example3_custom_env_and_eval()
    elif choice == "4":
        example4_ppo()
    else:
        print("无效选项")
