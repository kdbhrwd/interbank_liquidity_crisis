# 🏦 Interbank Liquidity Crisis
### Advanced Game Theory — Mini Project

A **Multi-Agent Reinforcement Learning** simulation built with Python and Pygame where **you play as the Central Bank (Federal Reserve)** and 5 autonomous AI banks compete and cooperate using Independent Q-Learning.

---

## 🎮 Gameplay

You are the **Federal Reserve**. Your goal is to keep the interbank system stable across 2,000 episodes while 5 AI banks independently learn their liquidity strategies. Each step, banks decide how much capital to keep liquid vs. invest in illiquid assets — a classic **Prisoner's Dilemma** framing.

Random **liquidity shocks** hit banks at any time. If the system can't absorb them, it collapses — and your **Systemic Health** drops. Reach 0 and it's game over.

---

## 🧠 Tech Stack

| Library | Role |
|---|---|
| `numpy` | Vectorised Q-table operations |
| `pygame-ce` | Real-time dashboard & input |
| `pandas` | Episode logging to CSV |

---

## ⚙️ Installation

```bash
# Python 3.10+ recommended (3.14 works with pygame-ce)
pip install pygame-ce numpy pandas
python interbank_liquidity_crisis.py
```

---

## 🕹️ Controls

| Input | Action |
|---|---|
| `[1-5]` or Click | Select / deselect a bank |
| `[B]` | **Bailout** — restore selected bank ($50) |
| `[Q]` | **QE Inject** — add +25 capital to bank ($35) |
| `[E]` | **Mandate Liquidity** — force LIQUID strategy x8 steps ($20) |
| `[T]` | **Tax Hoarder** — penalise HOARD banks (earn +$15) |
| `[C]` | **Emergency Rate Cut** — boost liquid yields for 15 steps ($60) |
| `[R]` | **Toggle Fed Rate** — Low (1%) ↔ High (3%) |
| `[X]` | **Stress Test** — manually shock a bank |
| `[↑/↓]` | Speed up / slow down simulation |
| `[SPACE]` | Pause / Resume |

> **Tip:** When a shock hits, a **`BAIL [B]!`** prompt pulses on the bank. Press `B` fast enough to intercept it before the penalty is applied.

---

## 📐 MDP Definition

**State Space** `(Own_Tier, System_Tier, Fed_Rate)` — 3 × 3 × 2 = 18 states  
**Action Space** — 5 actions: hold 0% / 25% / 50% / 75% / 100% as liquid  
**Reward** — `(liquid × liq_rate) + (illiquid × illiq_rate)` minus shock penalties  
**Q-Table Shape** — `(3, 3, 2, 5)` per agent  

---

## 🔥 Contagion Engine

- **15% chance** per step of a liquidity shock (size: 40–90 units)
- If targeted bank can't cover it → borrows from **total system liquidity**
- If system can't cover → **Systemic Collapse**: −100 reward to all agents, −15 Health to you
- Cascade flows visualised as animated particles flowing between banks

---

## 📊 Game Theory Concepts Modelled

| Concept | Mechanism |
|---|---|
| **Prisoner's Dilemma** | Banks benefit individually from hoarding but harm the collective |
| **Tragedy of the Commons** | If all banks hoard, system collapses — shown as warning banner |
| **Nash Equilibrium** | Detected when all agents converge on same strategy |
| **Mechanism Design** | Player's mandates, taxes, and QE shift the payoff matrix |
| **Correlated Equilibrium** | Mandate tool removes individual choice — enforces coordination |

---

## 📁 Output

At the end of the simulation, `simulation_data.csv` is written with:

```
episode, total_liquid, fed_rate, health
```

---

## 🏆 Win/Lose Conditions

- **Game Over** → Systemic Health drops to 0
- **Win** → Survive all 2,000 episodes

Economy is graded **D → AAA** based on your final health score.
