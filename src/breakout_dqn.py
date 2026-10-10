"""Partial DQN reproduction on ALE/Breakout-v5.
 
Refactored from the Group 42 executed Colab notebook. This script preserves the
notebook's main model, preprocessing, replay, training, and evaluation settings.
Run after installing requirements and ensuring the Atari Breakout ROM is available.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

# Configuration recorded in the executed notebook
SEED = 42
ENV_ID = "ALE/Breakout-v5"
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.00025
TRAIN_FRAMES = 50_000
LEARNING_STARTS = 1_000
TRAIN_FREQUENCY = 4
REPLAY_CAPACITY = 100_000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100_000
EVAL_EPISODES = 5
EVAL_EPSILON = 0.05


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB observation to a centered 84x84 grayscale frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
    top = (110 - 84) // 2
    cropped = resized[top:top + 84, :]
    return cropped.astype(np.uint8)


class FrameStack:
    """Stack four preprocessed frames as a (4, 84, 84) state."""

    def __init__(self, n_frames: int = 4):
        self.n_frames = n_frames
        self.frames: deque[np.ndarray] = deque(maxlen=n_frames)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(frame)
        self.frames.clear()
        for _ in range(self.n_frames):
            self.frames.append(processed)
        return np.stack(self.frames, axis=0)

    def append(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess_frame(frame))
        return np.stack(self.frames, axis=0)


class DQN(nn.Module):
    """Convolutional action-value network used in the notebook."""

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.features(x)
        x = self.fc(x)
        return self.head(x)


class ReplayBuffer:
    """Uniform replay buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, n_actions: int, device: str):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.online = DQN(4, n_actions).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.online.parameters(), lr=LR, alpha=0.95, eps=0.01
        )
        self.replay = ReplayBuffer(REPLAY_CAPACITY)
        self.total_steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.total_steps / EPS_DECAY)
        return EPS_START + fraction * (EPS_END - EPS_START)

    def act(self, state: np.ndarray, evaluate: bool = False) -> int:
        eps = EVAL_EPSILON if evaluate else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)
        x = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online(x).argmax(1).item())

    def update(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        current_q = self.online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.online(next_states_t).max(1).values
            target = rewards_t + GAMMA * (1.0 - dones_t) * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())


def train(agent: DQNAgent, env, device: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    checkpoint_path = out_dir / "checkpoint.pt"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "episode", "episode_reward", "epsilon", "loss"])
        frame = 0
        episode = 0

        while frame < TRAIN_FRAMES:
            obs, _ = env.reset(seed=SEED + episode)
            stacker = FrameStack(4)
            state = stacker.reset(obs)
            episode_reward = 0.0
            losses = []
            terminated = truncated = False

            while not (terminated or truncated) and frame < TRAIN_FRAMES:
                action = agent.act(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = stacker.append(next_obs)
                clipped_reward = 1.0 if reward > 0 else (-1.0 if reward < 0 else 0.0)
                agent.replay.add(
                    state, action, clipped_reward, next_state,
                    terminated or truncated
                )
                state = next_state
                episode_reward += reward
                frame += 1
                agent.total_steps += 1

                if frame >= LEARNING_STARTS and frame % TRAIN_FREQUENCY == 0:
                    loss = agent.update()
                    if loss is not None:
                        losses.append(loss)

            mean_loss = float(np.mean(losses)) if losses else np.nan
            writer.writerow([frame, episode, episode_reward, agent.epsilon(), mean_loss])
            f.flush()
            print(
                f"Episode {episode:4d} | frame {frame:7d}/{TRAIN_FRAMES} | "
                f"reward {episode_reward:7.2f} | epsilon {agent.epsilon():.3f}"
            )
            episode += 1

    torch.save({
        "model": agent.online.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "steps": agent.total_steps,
    }, checkpoint_path)
    print("\nTraining complete.")
    print("Training log:", log_path)
    print("Checkpoint:", checkpoint_path)
    return checkpoint_path


def evaluate(agent: DQNAgent, env, checkpoint_path: Path, device: str, out_dir: Path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.online.load_state_dict(checkpoint["model"])
    agent.online.eval()

    rewards = []
    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=1000 + ep)
        stacker = FrameStack(4)
        state = stacker.reset(obs)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(state, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = stacker.append(obs)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Evaluation episode {ep + 1}: reward = {total_reward:.2f}")

    eval_path = out_dir / "evaluation_results.csv"
    with eval_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, reward in enumerate(rewards, 1):
            writer.writerow([i, reward])
        writer.writerow([])
        writer.writerow(["mean_reward", np.mean(rewards)])
        writer.writerow(["std_reward", np.std(rewards)])

    print("\nMean evaluation reward:", np.mean(rewards))
    print("Evaluation results:", eval_path)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results", help="Where to save logs, checkpoint, and plots")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("PyTorch:", torch.__version__)
    print("Gymnasium:", gym.__version__)
    print("Device:", device)

    gym.register_envs(ale_py)
    env = gym.make(ENV_ID, frameskip=4, repeat_action_probability=0.0)
    try:
        obs, _ = env.reset(seed=SEED)
        print("Environment:", ENV_ID)
        print("Raw observation shape:", obs.shape)
        print("Action space:", env.action_space)
        print("Number of actions:", env.action_space.n)
        processed = preprocess_frame(obs)
        stacker = FrameStack(4)
        print("Processed frame shape:", processed.shape)
        print("Stacked state shape:", stacker.reset(obs).shape)

        agent = DQNAgent(env.action_space.n, device)
        print("Agent initialized on:", device)
        checkpoint_path = train(agent, env, device, out_dir)

        df = pd.read_csv(out_dir / "training_log.csv")
        plt.figure(figsize=(10, 5))
        plt.plot(df["frame"], df["episode_reward"], alpha=0.35, label="Episode reward")
        if len(df) >= 10:
            plt.plot(
                df["frame"], df["episode_reward"].rolling(10).mean(),
                label="10-episode moving average"
            )
        plt.xlabel("Training frame")
        plt.ylabel("Episode reward")
        plt.title("Breakout DQN — Our Partial Reproduction")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        reward_plot = out_dir / "reward_curve.png"
        plt.savefig(reward_plot, dpi=180)
        plt.close()
        print("Saved:", reward_plot)

        print("Number of recorded episodes:", len(df))
        print("Maximum observed episode reward:", df["episode_reward"].max())
        evaluation_rewards = evaluate(agent, env, checkpoint_path, device, out_dir)

        summary = pd.DataFrame({
            "Metric": [
                "Training frames", "Training episodes",
                "Maximum training episode reward", "Mean evaluation reward",
                "Evaluation episodes", "Device"
            ],
            "Our Partial Reproduction": [
                TRAIN_FRAMES, len(df), float(df["episode_reward"].max()),
                float(np.mean(evaluation_rewards)), EVAL_EPISODES, device
            ]
        })
        summary.to_csv(out_dir / "results_summary.csv", index=False)
        print(summary.to_string(index=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
