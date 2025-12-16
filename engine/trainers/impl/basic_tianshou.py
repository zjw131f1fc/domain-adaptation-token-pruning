"""Basic Tianshou RL Trainer.

基于Tianshou库的通用强化学习训练器，支持：
- Onpolicy算法（PPO, A2C等）
- Offpolicy算法（DQN, SAC, TD3等）
- 自定义policy或内置算法
- 自定义评估指标

设计原则：
1. 解耦：训练引擎与具体算法分离
2. 配置驱动：大部分参数从config读取
3. 灵活：支持注册自定义policy和eval函数
"""

from typing import Any, Dict, Optional, Callable
import torch
import numpy as np
from tqdm import tqdm


class BasicTianshouTrainer:
    """基于Tianshou的RL训练器"""

    def __init__(self, config: Any):
        """初始化trainer

        参数:
            config: 配置字典或对象，需包含：
                - global_settings.device
                - trainer_settings.rl_settings
                - policy_settings (可选)
        """
        self.config = config
        self.logger = getattr(config, "logger", None) or config.get("logger")

        # 从config读取RL设置
        if isinstance(config, dict):
            rl_settings = config["trainer_settings"]["rl_settings"]
            device_str = config["global_settings"]["device"]
        else:
            ts = config.trainer_settings
            rl_settings = ts["rl_settings"] if isinstance(ts, dict) else ts.rl_settings
            gs = config.global_settings
            device_str = gs["device"] if isinstance(gs, dict) else gs.device

        # 训练类型
        self.trainer_type = rl_settings["trainer_type"]  # "onpolicy" or "offpolicy"

        # 训练参数
        self.max_epoch = int(rl_settings["max_epoch"])
        self.step_per_epoch = int(rl_settings["step_per_epoch"])
        self.batch_size = int(rl_settings["batch_size"])
        self.episode_per_test = int(rl_settings.get("episode_per_test", 10))
        self.test_in_train = rl_settings.get("test_in_train", True)

        # Offpolicy参数
        self.step_per_collect = int(rl_settings.get("step_per_collect", 10))
        self.update_per_step = float(rl_settings.get("update_per_step", 0.1))

        # Onpolicy参数
        self.episode_per_collect = int(rl_settings.get("episode_per_collect", 8))
        self.repeat_per_collect = int(rl_settings.get("repeat_per_collect", 4))

        # Buffer设置
        self.buffer_size = int(rl_settings.get("buffer_size", 100000))
        self.buffer_type = rl_settings.get("buffer_type", "replay")

        # 设备
        self.device = torch.device(
            device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
        )

        # 组件（通过注册接口设置）
        self.train_envs = None
        self.test_envs = None
        self.policy = None

        # Tianshou核心组件（延迟初始化）
        self.buffer = None
        self.train_collector = None
        self.test_collector = None
        self.tianshou_trainer = None

        # 自定义评估函数（可选）
        self.eval_fn = None

        # 自定义policy标志
        self._custom_policy = None

        if self.logger:
            self.logger.info("BasicTianshouTrainer 初始化完成")
            self.logger.info(f"  - Device: {self.device}")
            self.logger.info(f"  - Trainer Type: {self.trainer_type}")
            self.logger.info(f"  - Max Epoch: {self.max_epoch}")

    # ==================== 注册接口 ====================

    def register_envs(self, train_envs, test_envs):
        """注册训练和测试环境

        参数:
            train_envs: Tianshou VectorEnv实例（训练环境）
            test_envs: Tianshou VectorEnv实例（测试环境）
        """
        self.train_envs = train_envs
        self.test_envs = test_envs

        if self.logger:
            self.logger.info(f"注册环境完成")
            self.logger.info(f"  - 训练环境数量: {len(train_envs)}")
            self.logger.info(f"  - 测试环境数量: {len(test_envs)}")

    def register_policy(self, policy):
        """注册自定义policy

        如果调用此方法，trainer将使用提供的policy，
        否则会根据config自动创建内置算法的policy

        参数:
            policy: Tianshou BasePolicy实例
        """
        self._custom_policy = policy

        if self.logger:
            self.logger.info(f"注册自定义Policy: {type(policy).__name__}")

    def register_eval_step(self, fn: Callable):
        """注册自定义评估函数（可选）

        函数签名:
            def eval_fn(collector_result, device, info) -> Dict[str, float]:
                '''
                参数:
                    collector_result: test_collector.collect()的返回结果
                    device: torch设备
                    info: 包含config、epoch等信息的字典

                返回:
                    自定义评估指标字典
                '''
                return {"custom_metric": value, ...}

        参数:
            fn: 评估函数
        """
        self.eval_fn = fn

        if self.logger:
            self.logger.info("注册自定义评估回调完成")

    # ==================== 主训练接口 ====================

    def run(self) -> Dict[str, Any]:
        """一键训练：自动创建所有组件并开始训练

        流程:
        1. 验证环境已注册
        2. 创建或使用已注册的policy
        3. 创建buffer（如果需要）
        4. 创建collectors
        5. 创建Tianshou trainer
        6. 开始训练
        7. 返回训练结果

        返回:
            训练结果字典，包含最终评估指标
        """
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("开始RL训练流程")
            self.logger.info("=" * 60)

        # Step 1: 验证环境
        self._validate_envs()

        # Step 2: 创建policy
        self._setup_policy()

        # Step 3: 创建buffer
        self._setup_buffer()

        # Step 4: 创建collectors
        self._setup_collectors()

        # Step 5: 创建Tianshou trainer
        self._setup_tianshou_trainer()

        # Step 6: 训练
        result = self._train()

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("训练完成")
            self.logger.info("=" * 60)

        return result

    # ==================== 内部方法 ====================

    def _validate_envs(self):
        """验证环境已正确注册"""
        if self.train_envs is None or self.test_envs is None:
            raise ValueError(
                "必须先调用 register_envs(train_envs, test_envs) 注册环境"
            )

        if self.logger:
            self.logger.info("环境验证通过")

    def _setup_policy(self):
        """创建或使用已注册的policy"""
        # 如果用户已注册自定义policy，直接使用
        if self._custom_policy is not None:
            self.policy = self._custom_policy
            if self.logger:
                self.logger.info("使用自定义Policy")
            return

        # 否则根据config创建内置policy
        if isinstance(self.config, dict):
            policy_settings = self.config.get("policy_settings")
        else:
            policy_settings = getattr(self.config, "policy_settings", None)

        if policy_settings is None:
            raise ValueError(
                "未注册自定义policy，且config中缺少policy_settings，"
                "请调用 register_policy() 或在config中配置policy_settings"
            )

        if isinstance(policy_settings, dict):
            algorithm = policy_settings.get("algorithm")
        else:
            algorithm = getattr(policy_settings, "algorithm", None)

        if algorithm is None or algorithm == "custom":
            raise ValueError(
                "policy_settings.algorithm 未设置或为'custom'，"
                "请调用 register_policy() 注册自定义policy"
            )

        # 根据algorithm创建内置policy
        if algorithm == "dqn":
            self.policy = self._create_dqn_policy()
        elif algorithm == "ppo":
            self.policy = self._create_ppo_policy()
        elif algorithm == "sac":
            self.policy = self._create_sac_policy()
        else:
            raise ValueError(f"不支持的内置算法: {algorithm}")

        if self.logger:
            self.logger.info(f"创建内置Policy: {algorithm.upper()}")

    def _create_dqn_policy(self):
        """根据config创建DQN policy"""
        from tianshou.policy import DQNPolicy
        from tianshou.utils.net.common import Net

        if isinstance(self.config, dict):
            policy_settings = self.config["policy_settings"]
        else:
            policy_settings = self.config.policy_settings

        # 读取DQN配置
        if isinstance(policy_settings, dict):
            dqn_cfg = policy_settings.get("dqn", {})
            net_cfg = policy_settings.get("network", {})
        else:
            dqn_cfg = getattr(policy_settings, "dqn", {})
            net_cfg = getattr(policy_settings, "network", {})

        # 获取环境空间信息
        obs_space = self.train_envs.observation_space
        act_space = self.train_envs.action_space

        # 创建网络
        state_shape = obs_space.shape or obs_space.n
        action_shape = act_space.shape or act_space.n

        hidden_sizes = net_cfg.get("hidden_sizes", [128, 128, 64]) if isinstance(net_cfg, dict) else [128, 128, 64]

        net = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            device=self.device
        ).to(self.device)

        # 创建优化器
        lr = dqn_cfg.get("lr", 1e-3) if isinstance(dqn_cfg, dict) else 1e-3
        optim = torch.optim.Adam(net.parameters(), lr=lr)

        # 创建policy
        policy = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=dqn_cfg.get("discount_factor", 0.99) if isinstance(dqn_cfg, dict) else 0.99,
            estimation_step=dqn_cfg.get("estimation_step", 3) if isinstance(dqn_cfg, dict) else 3,
            target_update_freq=dqn_cfg.get("target_update_freq", 320) if isinstance(dqn_cfg, dict) else 320,
        )

        return policy

    def _create_ppo_policy(self):
        """根据config创建PPO policy"""
        from tianshou.policy import PPOPolicy
        from tianshou.utils.net.common import ActorCritic, Net
        from tianshou.utils.net.discrete import Actor, Critic

        if isinstance(self.config, dict):
            policy_settings = self.config["policy_settings"]
        else:
            policy_settings = self.config.policy_settings

        # 读取PPO配置
        if isinstance(policy_settings, dict):
            ppo_cfg = policy_settings.get("ppo", {})
            net_cfg = policy_settings.get("network", {})
        else:
            ppo_cfg = getattr(policy_settings, "ppo", {})
            net_cfg = getattr(policy_settings, "network", {})

        # 获取环境空间信息
        obs_space = self.train_envs.observation_space
        act_space = self.train_envs.action_space

        state_shape = obs_space.shape or obs_space.n
        action_shape = act_space.shape or act_space.n

        hidden_sizes = net_cfg.get("hidden_sizes", [128, 128, 64]) if isinstance(net_cfg, dict) else [128, 128, 64]

        # 创建actor网络
        actor_net = Net(state_shape, hidden_sizes=hidden_sizes, device=self.device)
        actor = Actor(actor_net, action_shape, device=self.device).to(self.device)

        # 创建critic网络
        critic_net = Net(state_shape, hidden_sizes=hidden_sizes, device=self.device)
        critic = Critic(critic_net, device=self.device).to(self.device)

        # 合并为ActorCritic
        actor_critic = ActorCritic(actor, critic)

        # 创建优化器
        lr = ppo_cfg.get("lr", 1e-3) if isinstance(ppo_cfg, dict) else 1e-3
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        # 创建policy
        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            discount_factor=ppo_cfg.get("discount_factor", 0.99) if isinstance(ppo_cfg, dict) else 0.99,
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95) if isinstance(ppo_cfg, dict) else 0.95,
            vf_coef=ppo_cfg.get("vf_coef", 0.5) if isinstance(ppo_cfg, dict) else 0.5,
            ent_coef=ppo_cfg.get("ent_coef", 0.01) if isinstance(ppo_cfg, dict) else 0.01,
            action_scaling=False,
        )

        return policy

    def _create_sac_policy(self):
        """根据config创建SAC policy（连续动作空间）"""
        # TODO: 实现SAC policy创建
        raise NotImplementedError("SAC policy创建尚未实现")

    def _setup_buffer(self):
        """根据config创建buffer（如果需要）"""
        # Onpolicy不需要buffer
        if self.trainer_type == "onpolicy":
            self.buffer = None
            if self.logger:
                self.logger.info("Onpolicy训练，无需buffer")
            return

        # Offpolicy创建buffer
        from tianshou.data import ReplayBuffer, PrioritizedReplayBuffer, VectorReplayBuffer

        if self.buffer_type == "replay":
            self.buffer = VectorReplayBuffer(
                total_size=self.buffer_size,
                buffer_num=len(self.train_envs)
            )
        elif self.buffer_type == "prioritized":
            self.buffer = PrioritizedReplayBuffer(
                size=self.buffer_size,
                alpha=0.6,
                beta=0.4
            )
        else:
            raise ValueError(f"不支持的buffer类型: {self.buffer_type}")

        if self.logger:
            self.logger.info(f"创建Buffer: {self.buffer_type}, size={self.buffer_size}")

    def _setup_collectors(self):
        """创建train/test collectors"""
        from tianshou.data import Collector

        # 训练collector
        self.train_collector = Collector(
            policy=self.policy,
            env=self.train_envs,
            buffer=self.buffer,
            exploration_noise=True
        )

        # 测试collector
        self.test_collector = Collector(
            policy=self.policy,
            env=self.test_envs,
            exploration_noise=False
        )

        if self.logger:
            self.logger.info("创建Collectors完成")

    def _setup_tianshou_trainer(self):
        """根据trainer_type创建对应的Tianshou trainer"""
        if self.trainer_type == "offpolicy":
            from tianshou.trainer import OffpolicyTrainer

            self.tianshou_trainer = OffpolicyTrainer(
                policy=self.policy,
                train_collector=self.train_collector,
                test_collector=self.test_collector,
                max_epoch=self.max_epoch,
                step_per_epoch=self.step_per_epoch,
                step_per_collect=self.step_per_collect,
                episode_per_test=self.episode_per_test,
                batch_size=self.batch_size,
                update_per_step=self.update_per_step,
                test_in_train=self.test_in_train,
                verbose=True
            )

        elif self.trainer_type == "onpolicy":
            from tianshou.trainer import OnpolicyTrainer

            self.tianshou_trainer = OnpolicyTrainer(
                policy=self.policy,
                train_collector=self.train_collector,
                test_collector=self.test_collector,
                max_epoch=self.max_epoch,
                step_per_epoch=self.step_per_epoch,
                repeat_per_collect=self.repeat_per_collect,
                episode_per_test=self.episode_per_test,
                batch_size=self.batch_size,
                episode_per_collect=self.episode_per_collect,
                test_in_train=self.test_in_train,
                verbose=True
            )

        else:
            raise ValueError(f"不支持的trainer类型: {self.trainer_type}")

        if self.logger:
            self.logger.info(f"创建Tianshou {self.trainer_type.capitalize()}Trainer完成")

    def _train(self) -> Dict[str, Any]:
        """执行训练主循环"""
        if self.logger:
            self.logger.info("开始训练...")

        # 执行Tianshou的训练
        result = self.tianshou_trainer.run()

        # 整理返回结果
        final_result = {
            "best_reward": float(result.get("best_reward", 0)),
            "best_reward_std": float(result.get("best_reward_std", 0)),
            "train_step": int(result.get("train_step", 0)),
            "train_episode": int(result.get("train_episode", 0)),
        }

        # 如果有自定义eval_fn，执行额外评估
        if self.eval_fn is not None:
            if self.logger:
                self.logger.info("执行自定义评估...")

            custom_metrics = self._run_custom_eval()
            final_result.update(custom_metrics)

        if self.logger:
            self.logger.info(f"最终结果: {final_result}")

        return final_result

    def _run_custom_eval(self) -> Dict[str, float]:
        """运行自定义评估函数"""
        # 在test环境上收集episodes
        collector_result = self.test_collector.collect(
            n_episode=self.episode_per_test,
            render=0
        )

        # 构建info
        info = {
            "config": self.config,
            "policy": self.policy,
            "train_envs": self.train_envs,
            "test_envs": self.test_envs,
        }

        # 调用自定义eval_fn
        custom_metrics = self.eval_fn(collector_result, self.device, info)

        if self.logger:
            self.logger.info(f"自定义评估结果: {custom_metrics}")

        return custom_metrics
