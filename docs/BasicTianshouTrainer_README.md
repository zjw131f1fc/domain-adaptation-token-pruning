# BasicTianshouTrainer 使用指南

## 概述

`BasicTianshouTrainer` 是一个基于 [Tianshou](https://github.com/thu-ml/tianshou) 库的通用强化学习训练器，与项目中的 `BasicPytorchTrainer` 设计理念一致，提供解耦、配置驱动、灵活扩展的RL训练框架。

### 核心特性

- **解耦设计**: 训练引擎与具体算法分离
- **配置驱动**: 大部分参数从YAML配置文件读取
- **统一接口**: 与 `BasicPytorchTrainer` 保持一致的注册模式
- **灵活扩展**: 支持内置算法和自定义Policy
- **双模式支持**: 同时支持 Onpolicy（PPO, A2C等）和 Offpolicy（DQN, SAC等）算法

---

## 快速开始

### 1. 最简单的使用（内置DQN算法）

```python
from engine.configs.loader import load_config
from engine.trainers.loader import load_trainer
import gymnasium as gym
from tianshou.env import DummyVectorEnv

# 加载配置
config = load_config(override_file="configs/rl_dqn_example.yaml")

# 创建trainer
trainer = load_trainer(config)

# 创建环境
train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])
test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(1)])

# 注册环境
trainer.register_envs(train_envs, test_envs)

# 一键训练（policy自动创建）
result = trainer.run()

print(f"最佳奖励: {result['best_reward']:.2f}")
```

### 2. 使用自定义Policy

```python
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net

# 创建自定义网络
net = Net(state_shape=(4,), action_shape=2, hidden_sizes=[256, 256])
optim = torch.optim.Adam(net.parameters(), lr=0.0005)

# 创建自定义policy
policy = DQNPolicy(model=net, optim=optim, discount_factor=0.95)

# 注册policy
trainer.register_policy(policy)

# 训练
result = trainer.run()
```

### 3. 使用自定义评估函数

```python
def custom_eval(collector_result, device, info):
    """自定义评估指标"""
    rewards = collector_result.returns
    success_rate = float((rewards > 100).sum() / len(rewards))

    return {
        "success_rate": success_rate,
        "avg_reward": float(rewards.mean())
    }

# 注册评估函数
trainer.register_eval_step(custom_eval)

# 训练（会自动调用自定义评估）
result = trainer.run()

print(f"成功率: {result['success_rate']:.2%}")
```

---

## 配置文件说明

### DQN配置示例（Offpolicy）

```yaml
# configs/rl_dqn_example.yaml

global_settings:
  device: "cuda"
  seed: 42

trainer_settings:
  name: "basic-tianshou"

  rl_settings:
    trainer_type: "offpolicy"  # Offpolicy算法

    max_epoch: 10
    step_per_epoch: 10000
    batch_size: 64

    # Offpolicy专用
    step_per_collect: 10
    update_per_step: 0.1

    # Buffer设置
    buffer_size: 20000
    buffer_type: "replay"

    # 评估
    episode_per_test: 10
    test_in_train: true

policy_settings:
  algorithm: "dqn"  # 使用内置DQN

  dqn:
    lr: 0.001
    discount_factor: 0.99
    estimation_step: 3
    target_update_freq: 320

  network:
    hidden_sizes: [128, 128, 64]
```

### PPO配置示例（Onpolicy）

```yaml
# configs/rl_ppo_example.yaml

trainer_settings:
  name: "basic-tianshou"

  rl_settings:
    trainer_type: "onpolicy"  # Onpolicy算法

    max_epoch: 10
    step_per_epoch: 50000
    batch_size: 64

    # Onpolicy专用
    episode_per_collect: 16
    repeat_per_collect: 2

    episode_per_test: 10

policy_settings:
  algorithm: "ppo"  # 使用内置PPO

  ppo:
    lr: 0.0003
    discount_factor: 0.99
    gae_lambda: 0.95
    vf_coef: 0.5
    ent_coef: 0.01

  network:
    hidden_sizes: [64, 64]
```

---

## API文档

### BasicTianshouTrainer

#### 初始化

```python
trainer = BasicTianshouTrainer(config)
```

**参数**:
- `config`: 配置字典或对象，需包含 `trainer_settings.rl_settings`

#### 注册接口

##### register_envs(train_envs, test_envs)

注册训练和测试环境。

```python
trainer.register_envs(train_envs, test_envs)
```

**参数**:
- `train_envs`: Tianshou VectorEnv实例（训练环境）
- `test_envs`: Tianshou VectorEnv实例（测试环境）

**说明**:
- 训练和测试环境可以使用不同的配置（如不同的数据集split）
- 环境数量从配置文件读取或由用户创建时决定

##### register_policy(policy)

注册自定义Policy（可选）。

```python
trainer.register_policy(policy)
```

**参数**:
- `policy`: Tianshou BasePolicy实例

**说明**:
- 如果不调用此方法，trainer会根据config自动创建内置算法的policy
- 如果调用，config中的 `policy_settings.algorithm` 应设为 `"custom"`

##### register_eval_step(fn)

注册自定义评估函数（可选）。

```python
def eval_fn(collector_result, device, info):
    return {"custom_metric": value}

trainer.register_eval_step(eval_fn)
```

**参数**:
- `fn`: 评估函数，签名为 `fn(collector_result, device, info) -> Dict[str, float]`

**说明**:
- Tianshou会自动计算标准指标（reward, length等）
- 自定义eval_fn用于计算额外的指标（如success_rate等）

#### 训练接口

##### run()

一键训练，返回训练结果。

```python
result = trainer.run()
```

**返回**:
```python
{
    "best_reward": float,        # 最佳平均奖励
    "best_reward_std": float,    # 最佳奖励标准差
    "train_step": int,           # 总训练步数
    "train_episode": int,        # 总训练episode数
    # 如果注册了custom eval_fn，会包含额外指标
}
```

---

## 支持的内置算法

### Offpolicy算法

- [x] **DQN** (Deep Q-Network)
- [ ] **SAC** (Soft Actor-Critic) - 待实现
- [ ] **TD3** (Twin Delayed DDPG) - 待实现

### Onpolicy算法

- [x] **PPO** (Proximal Policy Optimization)
- [ ] **A2C** (Advantage Actor-Critic) - 待实现

---

## 自定义环境示例

```python
import gymnasium as gym

class CustomEnv(gym.Env):
    """自定义环境"""

    def __init__(self, split="train", level=0):
        super().__init__()
        self.split = split  # train或test
        self.level = level

        # 定义空间
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 初始化状态
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        # 执行动作
        reward = ...
        terminated = ...
        next_state = ...

        return next_state, reward, terminated, False, {}

# 使用自定义环境
from tianshou.env import DummyVectorEnv

train_envs = DummyVectorEnv([
    lambda i=i: CustomEnv(split="train", level=i)
    for i in range(10)
])

test_envs = DummyVectorEnv([
    lambda: CustomEnv(split="test", level=0)
    for _ in range(1)
])

trainer.register_envs(train_envs, test_envs)
```

---

## 与 BasicPytorchTrainer 的对比

| 特性 | BasicPytorchTrainer | BasicTianshouTrainer |
|------|---------------------|---------------------|
| **数据来源** | Dataset (split["train"/"test"]) | VectorEnv (train_envs/test_envs) |
| **模型注册** | `register_model(name, model)` | `register_policy(policy)` |
| **参数组** | `add_param_group()` + `setup_optimizers()` | Policy内置optimizer |
| **训练回调** | `register_train_step(fn)` | 无（Tianshou内置） |
| **评估回调** | `register_eval_step(fn)` | `register_eval_step(fn)` |
| **训练循环** | 自定义 | Tianshou Trainer |

---

## 完整示例

参见 [`examples/rl_trainer_examples.py`](../examples/rl_trainer_examples.py)，包含：

1. 简单DQN训练
2. 自定义Policy
3. 自定义环境和评估
4. PPO算法训练

运行示例：

```bash
python examples/rl_trainer_examples.py 1  # 示例1
python examples/rl_trainer_examples.py 2  # 示例2
python examples/rl_trainer_examples.py 3  # 示例3
python examples/rl_trainer_examples.py 4  # 示例4
```

---

## 常见问题

### Q1: 如何支持多进程并行环境？

使用 `SubprocVectorEnv` 替代 `DummyVectorEnv`：

```python
from tianshou.env import SubprocVectorEnv

train_envs = SubprocVectorEnv([
    lambda: gym.make("CartPole-v1") for _ in range(10)
])
```

### Q2: 如何保存和加载训练好的模型？

```python
# 保存
torch.save(trainer.policy.state_dict(), "policy.pth")

# 加载
policy.load_state_dict(torch.load("policy.pth"))
```

### Q3: 如何支持连续动作空间？

使用SAC或TD3算法（待实现），或自定义Policy：

```python
from tianshou.policy import SACPolicy

# 创建连续动作空间的policy
policy = SACPolicy(...)
trainer.register_policy(policy)
```

### Q4: Buffer在哪里创建？

Trainer根据config自动创建：
- **Offpolicy**: 自动创建 `VectorReplayBuffer` 或 `PrioritizedReplayBuffer`
- **Onpolicy**: 不需要buffer

---

## 依赖

- `tianshou >= 1.0.0`
- `gymnasium >= 0.28.0`
- `torch >= 2.0.0`

安装：

```bash
pip install tianshou gymnasium torch
```

---

## 贡献

欢迎添加更多内置算法！参考 `_create_dqn_policy()` 和 `_create_ppo_policy()` 实现新算法。

---

## 参考

- [Tianshou 官方文档](https://tianshou.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)
