# Limitations and Open Questions

*DRL Assignment 1 – Group 42 | Breakout DQN Reproduction*

---

## Limitations highlighted by the authors

### 1. No Theoretical Convergence Guarantee

Although the method trained stably in all experiments, the authors explicitly state that it **does not have a theoretical guarantee of convergence**.

---

### 2. Experience Replay is Limited

The replay memory:

- stores only a **finite number of experiences**
- overwrites older experiences
- gives **equal sampling probability** to every stored transition

The authors suggest that a more sophisticated sampling strategy could focus on experiences from which the agent can learn more.

---

### 3. Reward Clipping Loses Information

During training, all positive rewards were changed to **+1** and all negative rewards to **−1**.

The authors acknowledge that this may affect performance because the agent **cannot distinguish between rewards of different magnitudes**.

---

## Open Questions highlighted by the authors

### Difficulty with Long-Term Strategies

DQN was far below human performance on:

- **Q\*bert**
- **Seaquest**
- **Space Invaders**

The authors explain that these games are more challenging because they require strategies that extend over **longer time scales**.

---

## Key Question

> **Can deep Q-learning reliably learn strategies that require longer-term planning, while providing stronger theoretical guarantees and making better use of past experiences?**

---

## Relevance to Our Reproduction

Our reproduction ran for only **50,000 frames** (vs. the paper's 10 million), which amplifies all three limitations above:

- The short training horizon means convergence is even less guaranteed.
- The small replay buffer (capacity 10,000) cycles through experiences quickly, making uniform sampling particularly wasteful.
- Reward clipping still applies; with low total rewards seen, its impact is hard to measure but the agent never distinguishes between a 1-point and a 7-point brick hit.

These constraints explain the **mean evaluation reward of 0.0** observed in our run and are the primary gap between our results and the paper's reported scores.
