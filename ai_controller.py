#!/usr/bin/env python3
"""
AI Traffic Light Controller  (MAPPO + Kinematic Safety + Emergency Preemption)
=============================================================================
Replaces heuristic TL sync in one SUMO instance to enable A/B comparison.

Architecture
------------
Phase 3 — PPO Actor-Critic   : state → action (0=maintain, 1=switch)
Phase 4 — Kinematic Safety   : T_yellow + T_clearance state machine, T_min/T_max
Phase 5 — Emergency Preempt  : ambulance ETA triggers green corridor

Usage
-----
  Training (headless, one-time):
      python ai_controller.py --train --episodes 500

  Inference (called by digital_twin.py via --use-ai):
      from ai_controller import AITrafficController
      ctrl = AITrafficController(traci_module, checkpoint="models/mappo_checkpoint.pt")
      ctrl.step()   # call every simulation step
"""

import os
import sys
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── NETWORK & INTERSECTION CONSTANTS ──────────────────────────────────────────
TL_ID             = "J25"
INCOMING_EDGES    = ["E17", "E19", "E18", "E16"]   # N, S, E, W
INCOMING_LANES    = ["E17_0", "E19_0", "E18_0", "E16_0"]

# TL phase indices in crossroads.net.xml
#   0: GGgrrrGGgrrr  (NS green)
#   1: yyyrrryyyrrr  (NS yellow)
#   2: rrrGGgrrrGGg  (EW green)
#   3: rrryyyrrryyy  (EW yellow)
PHASE_NS_GREEN    = 0
PHASE_EW_GREEN    = 2

# ── KINEMATIC SAFETY CONSTANTS ────────────────────────────────────────────────
REACTION_TIME     = 1.0       # s  — driver reaction time
DECEL_COMFORT     = 4.5       # m/s²  — comfortable deceleration
GRAVITY           = 9.81      # m/s²
GRADE             = 0.0       # road grade (flat)
INTERSECTION_W    = 14.5      # m  — intersection crossing width
VEHICLE_LENGTH    = 5.0       # m  — standard car length
DEFAULT_SPEED     = 13.89     # m/s  — 50 km/h speed limit

T_MIN             = 3.0       # s  — minimum green duration (prevent flickering)
T_MAX             = 45.0      # s  — maximum green duration (prevent starvation)
QUEUE_SWITCH_THRESHOLD = 2    # switch if opposite arm has this many more queued vehicles

# ── EMERGENCY PREEMPTION ──────────────────────────────────────────────────────
SAFETY_BUFFER     = 2.0       # s  — extra clearance before ambulance arrives
# Junction centre coordinates (from net.xml J25 position)
JUNCTION_X        = -10.89
JUNCTION_Y        = -25.86

# ── RL HYPERPARAMETERS ────────────────────────────────────────────────────────
STATE_DIM         = 10        # 4 queues + 4 speeds + 2 phase one-hot
ACTION_DIM        = 2         # 0=maintain, 1=switch
HIDDEN_DIM        = 64
ALPHA_QUEUE       = 0.5       # reward weight for queue penalty
BETA_DELAY        = 0.01      # reward weight for delay² penalty
GAMMA             = 0.99      # discount factor
CLIP_EPS          = 0.2       # PPO clipping epsilon
LR                = 3e-4
EPOCHS_PER_UPDATE = 4
BATCH_SIZE        = 64


# ═════════════════════════════════════════════════════════════════════════════
#  PPO ACTOR-CRITIC NETWORK
# ═════════════════════════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value  = self.critic(h)
        return logits, value

    def act(self, state_tensor):
        """Returns (action, log_prob, value)."""
        logits, value = self.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)

    def evaluate(self, states, actions):
        """Batch evaluation for PPO update."""
        logits, values = self.forward(states)
        probs    = torch.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# ═════════════════════════════════════════════════════════════════════════════
#  KINEMATIC SAFETY CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════
def compute_yellow_time(approach_speed: float) -> float:
    """T_yellow = reaction + v / (2·a + 2·g·grade)"""
    v = max(approach_speed, 1.0)
    denom = 2 * DECEL_COMFORT + 2 * GRAVITY * GRADE
    return REACTION_TIME + v / denom

def compute_clearance_time(approach_speed: float) -> float:
    """T_clearance = (W + L) / v"""
    v = max(approach_speed, 1.0)
    return (INTERSECTION_W + VEHICLE_LENGTH) / v

def compute_transition_time(approach_speed: float) -> tuple:
    """Returns (T_yellow, T_clearance, T_total)."""
    t_y = compute_yellow_time(approach_speed)
    t_c = compute_clearance_time(approach_speed)
    return t_y, t_c, t_y + t_c


# ═════════════════════════════════════════════════════════════════════════════
#  AI TRAFFIC CONTROLLER  (used at inference time inside digital_twin.py)
# ═════════════════════════════════════════════════════════════════════════════
class AITrafficController:
    """
    Drop-in replacement for heuristic TL sync.
    Call .step() every SUMO simulation step.

    State machine:
        GREEN → (AI says switch OR T_max exceeded OR ambulance preempt)
              → YELLOW (T_yellow seconds)
              → ALL_RED (T_clearance seconds)
              → next GREEN
    """

    # Internal states
    ST_GREEN   = "GREEN"
    ST_YELLOW  = "YELLOW"
    ST_ALL_RED = "ALL_RED"

    def __init__(self, traci_module, checkpoint: str = "", step_length: float = 1.0):
        self.traci = traci_module
        self.step_len = step_length   # SUMO step length in seconds

        # Load trained policy
        self.model = ActorCritic()
        if checkpoint and os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            print(f"  [AI] Loaded checkpoint: {checkpoint}")
        else:
            print("  [AI] No checkpoint — using random policy")
        self.model.eval()

        # State machine
        self._state       = self.ST_GREEN
        self._current_phase = PHASE_NS_GREEN   # which green is active now
        self._green_timer = 0.0                # seconds in current green
        self._trans_timer = 0.0                # countdown during yellow/all-red
        self._t_yellow    = 0.0
        self._t_clearance = 0.0
        self._preempt_arm = ""                 # edge that needs green for ambulance
        self._preempt_active = False

        # Counters for logging
        self._total_steps = 0
        self._switches    = 0

        # Immediately seize TL control from default program
        try:
            self.traci.trafficlight.setPhase(TL_ID, self._current_phase)
            self.traci.trafficlight.setPhaseDuration(TL_ID, 9999)
            print(f"  [AI] Seized TL control - starting with {'NS' if self._current_phase == PHASE_NS_GREEN else 'EW'} GREEN")
        except Exception:
            pass

    # ── State Extraction ──────────────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        """Extract 10-dim state: [queue×4, speed×4, phase_onehot×2]."""
        queues = []
        speeds = []
        for lane in INCOMING_LANES:
            try:
                queues.append(self.traci.lane.getLastStepHaltingNumber(lane))
                speeds.append(self.traci.lane.getLastStepMeanSpeed(lane))
            except Exception:
                queues.append(0)
                speeds.append(DEFAULT_SPEED)

        phase_oh = [1.0, 0.0] if self._current_phase == PHASE_NS_GREEN else [0.0, 1.0]
        return np.array(queues + speeds + phase_oh, dtype=np.float32)

    # ── Reward Calculation ────────────────────────────────────────────────────
    def _compute_reward(self) -> float:
        """R = -Σ(α·queue + β·delay²)"""
        reward = 0.0
        for lane in INCOMING_LANES:
            try:
                q = self.traci.lane.getLastStepHaltingNumber(lane)
                # delay = time vehicles spent waiting (proxy: halting count × step)
                w = self.traci.lane.getWaitingTime(lane)
                reward -= (ALPHA_QUEUE * q + BETA_DELAY * w * w)
            except Exception:
                pass
        return reward

    # ── 85th Percentile Approach Speed ────────────────────────────────────────
    def _get_approach_speed_p85(self) -> float:
        """Get 85th percentile speed of approach vehicles (for safety calcs)."""
        speeds = []
        for edge in INCOMING_EDGES:
            try:
                for vid in self.traci.edge.getLastStepVehicleIDs(edge):
                    s = self.traci.vehicle.getSpeed(vid)
                    if s > 0.5:
                        speeds.append(s)
            except Exception:
                pass
        if not speeds:
            return DEFAULT_SPEED
        speeds.sort()
        idx = int(len(speeds) * 0.85)
        return speeds[min(idx, len(speeds) - 1)]

    # ── Emergency Preemption Scan ─────────────────────────────────────────────
    def _scan_emergency(self) -> tuple:
        """
        Returns (should_preempt: bool, ambulance_edge: str).
        Triggers if ETA <= T_transition + SAFETY_BUFFER.
        """
        v85 = self._get_approach_speed_p85()
        _, _, t_trans = compute_transition_time(v85)

        try:
            all_vehicles = self.traci.vehicle.getIDList()
        except Exception:
            return False, ""

        for vid in all_vehicles:
            try:
                vtype = self.traci.vehicle.getTypeID(vid)
                if vtype != "ambulance":
                    continue

                # Euclidean distance to junction centre
                x, y = self.traci.vehicle.getPosition(vid)
                dist = math.sqrt((x - JUNCTION_X) ** 2 + (y - JUNCTION_Y) ** 2)
                speed = max(self.traci.vehicle.getSpeed(vid), 1.0)
                eta = dist / speed

                if eta <= (t_trans + SAFETY_BUFFER):
                    edge = self.traci.vehicle.getRoadID(vid)
                    print(f"  [AI-PREEMPT] Ambulance {vid} ETA={eta:.1f}s on {edge} — preempting!")
                    return True, edge
            except Exception:
                continue

        return False, ""

    def _edge_to_phase(self, edge: str) -> int:
        """Determine which green phase serves the given edge."""
        if edge in ("E17", "-E17", "E19", "-E19"):
            return PHASE_NS_GREEN
        return PHASE_EW_GREEN

    # ── Phase Transition Logic ────────────────────────────────────────────────
    def _start_transition(self, reason: str = "AI"):
        """Begin yellow → all-red → next green sequence."""
        v85 = self._get_approach_speed_p85()
        self._t_yellow, self._t_clearance, _ = compute_transition_time(v85)
        self._state = self.ST_YELLOW
        self._trans_timer = self._t_yellow

        # Set SUMO TL to yellow phase
        yellow_phase = 1 if self._current_phase == PHASE_NS_GREEN else 3
        try:
            self.traci.trafficlight.setPhase(TL_ID, yellow_phase)
            self.traci.trafficlight.setPhaseDuration(TL_ID, self._t_yellow + self._t_clearance + 5)
        except Exception:
            pass

        self._switches += 1
        print(f"  [AI-TL] {reason}: YELLOW {self._t_yellow:.1f}s + ALL_RED {self._t_clearance:.1f}s")

    # ── Main Step ─────────────────────────────────────────────────────────────
    def step(self):
        """Call once per SUMO simulation step."""
        self._total_steps += 1
        dt = self.step_len

        # ── Phase 5: Emergency preemption check (highest priority) ────────────
        if self._state == self.ST_GREEN and not self._preempt_active:
            should_preempt, amb_edge = self._scan_emergency()
            if should_preempt:
                needed_phase = self._edge_to_phase(amb_edge)
                if needed_phase != self._current_phase:
                    # Need to switch to give ambulance green
                    self._preempt_active = True
                    self._preempt_arm = amb_edge
                    self._start_transition(reason="AMBULANCE-PREEMPT")
                    return self._compute_reward()
                else:
                    # Already green for ambulance — extend green
                    self._green_timer = 0.0   # reset timer to keep it green
                    return self._compute_reward()

        # ── State machine ─────────────────────────────────────────────────────
        if self._state == self.ST_YELLOW:
            self._trans_timer -= dt
            if self._trans_timer <= 0:
                # Transition to ALL_RED
                self._state = self.ST_ALL_RED
                self._trans_timer = self._t_clearance
                # Stay in yellow phase — cross traffic is already red.
                # Do NOT call setRedYellowGreenState: it creates a 1-phase temp
                # program that breaks subsequent setPhase(index > 0) calls.
            return self._compute_reward()

        if self._state == self.ST_ALL_RED:
            self._trans_timer -= dt
            if self._trans_timer <= 0:
                # Snap to next green
                next_phase = PHASE_EW_GREEN if self._current_phase == PHASE_NS_GREEN else PHASE_NS_GREEN

                # If preempting for ambulance, go to the ambulance's needed phase
                if self._preempt_active:
                    next_phase = self._edge_to_phase(self._preempt_arm)
                    self._preempt_active = False
                    self._preempt_arm = ""

                self._current_phase = next_phase
                self._green_timer = 0.0
                self._state = self.ST_GREEN
                try:
                    self.traci.trafficlight.setPhase(TL_ID, next_phase)
                    self.traci.trafficlight.setPhaseDuration(TL_ID, 9999)
                except Exception:
                    pass
                print(f"  [AI-TL] GREEN: {'NS' if next_phase == PHASE_NS_GREEN else 'EW'}")
            return self._compute_reward()

        # ── ST_GREEN: check AI decision ───────────────────────────────────────
        self._green_timer += dt

        # Enforce T_max — force switch if green too long
        if self._green_timer >= T_MAX:
            self._start_transition(reason=f"T_MAX ({T_MAX:.0f}s)")
            return self._compute_reward()

        # Enforce T_min — don't allow switch before minimum green
        if self._green_timer < T_MIN:
            return self._compute_reward()

        # ── Queue-responsive adaptive logic ───────────────────────────────────
        # Get real-time queue lengths for each direction
        try:
            q_n = self.traci.lane.getLastStepHaltingNumber(INCOMING_LANES[0])  # E17 North
            q_s = self.traci.lane.getLastStepHaltingNumber(INCOMING_LANES[1])  # E19 South
            q_e = self.traci.lane.getLastStepHaltingNumber(INCOMING_LANES[2])  # E18 East
            q_w = self.traci.lane.getLastStepHaltingNumber(INCOMING_LANES[3])  # E16 West
        except Exception:
            q_n = q_s = q_e = q_w = 0

        if self._current_phase == PHASE_NS_GREEN:
            current_queue = q_n + q_s   # NS is green, these should be flowing
            opposite_queue = q_e + q_w  # EW is red, these are waiting
        else:
            current_queue = q_e + q_w
            opposite_queue = q_n + q_s

        # Rule-based: switch if opposite arm has significantly more demand
        queue_imbalance = opposite_queue - current_queue
        should_switch_rule = queue_imbalance >= QUEUE_SWITCH_THRESHOLD

        # Also ask the PPO policy
        state = self._get_state()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state_t)
            probs = torch.softmax(logits, dim=-1)
            ppo_action = torch.argmax(probs, dim=-1).item()

        # Combined decision: switch if EITHER the rule or PPO says so
        # (union of smart heuristic + learned policy)
        if should_switch_rule or ppo_action == 1:
            reason = f"QUEUE({opposite_queue}>{current_queue})" if should_switch_rule else "PPO"
            self._start_transition(reason=reason)

        return self._compute_reward()

    def get_stats(self) -> dict:
        return {
            "total_steps":   self._total_steps,
            "phase_switches": self._switches,
            "current_phase": "NS" if self._current_phase == PHASE_NS_GREEN else "EW",
            "state":         self._state,
            "green_timer":   round(self._green_timer, 1),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  PPO TRAINING LOOP  (headless SUMO, offline)
# ═════════════════════════════════════════════════════════════════════════════
class PPOTrainer:
    """Train the traffic light policy using headless SUMO episodes."""

    def __init__(self, sumo_cfg: str, port: int = 8820):
        self.sumo_cfg = sumo_cfg
        self.port     = port
        self.model    = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def _start_sumo(self):
        import traci
        import subprocess

        binary = "sumo"   # headless for training
        cmd = [
            binary, "-c", self.sumo_cfg,
            "--remote-port", str(self.port),
            "--start",
            "--quit-on-end", "false",
            "--no-step-log",
            "--time-to-teleport", "-1",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time; time.sleep(1.5)
        traci.init(self.port)
        return traci

    def _get_state(self, traci_mod) -> np.ndarray:
        queues, speeds = [], []
        for lane in INCOMING_LANES:
            try:
                queues.append(traci_mod.lane.getLastStepHaltingNumber(lane))
                speeds.append(traci_mod.lane.getLastStepMeanSpeed(lane))
            except Exception:
                queues.append(0)
                speeds.append(DEFAULT_SPEED)

        try:
            phase = traci_mod.trafficlight.getPhase(TL_ID)
        except Exception:
            phase = 0
        phase_oh = [1.0, 0.0] if phase in (0, 1) else [0.0, 1.0]
        return np.array(queues + speeds + phase_oh, dtype=np.float32)

    def _compute_reward(self, traci_mod) -> float:
        reward = 0.0
        for lane in INCOMING_LANES:
            try:
                q = traci_mod.lane.getLastStepHaltingNumber(lane)
                w = traci_mod.lane.getWaitingTime(lane)
                reward -= (ALPHA_QUEUE * q + BETA_DELAY * w * w)
            except Exception:
                pass
        return reward

    def train(self, episodes: int = 500, steps_per_ep: int = 300,
              save_path: str = "models/mappo_checkpoint.pt"):
        """Run PPO training across multiple headless SUMO episodes."""
        import traci as traci_mod

        best_reward = -float("inf")
        print(f"\n[AI-TRAIN] Starting PPO training: {episodes} episodes x {steps_per_ep} steps")
        print(f"[AI-TRAIN] Checkpoint will be saved to: {save_path}\n")

        for ep in range(episodes):
            # Start fresh SUMO episode
            try:
                traci_mod = self._start_sumo()
            except Exception as e:
                print(f"  [!] SUMO start failed ep {ep}: {e}")
                continue

            states, actions, log_probs, rewards, values = [], [], [], [], []
            current_phase = PHASE_NS_GREEN
            green_timer = 0.0

            for step in range(steps_per_ep):
                state = self._get_state(traci_mod)
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                self.model.train()
                action, log_prob, value = self.model.act(state_t)

                # Apply action with safety constraints
                green_timer += 1.0
                switched = False

                if action == 1 and green_timer >= T_MIN:
                    # Switch phase
                    next_phase = PHASE_EW_GREEN if current_phase == PHASE_NS_GREEN else PHASE_NS_GREEN
                    try:
                        yellow_ph = 1 if current_phase == PHASE_NS_GREEN else 3
                        traci_mod.trafficlight.setPhase(TL_ID, yellow_ph)
                        traci_mod.simulationStep()
                        traci_mod.simulationStep()
                        traci_mod.simulationStep()   # ~3s yellow
                        traci_mod.trafficlight.setPhase(TL_ID, next_phase)
                        traci_mod.trafficlight.setPhaseDuration(TL_ID, 9999)
                    except Exception:
                        pass
                    current_phase = next_phase
                    green_timer = 0.0
                    switched = True

                if green_timer >= T_MAX:
                    next_phase = PHASE_EW_GREEN if current_phase == PHASE_NS_GREEN else PHASE_NS_GREEN
                    try:
                        yellow_ph = 1 if current_phase == PHASE_NS_GREEN else 3
                        traci_mod.trafficlight.setPhase(TL_ID, yellow_ph)
                        traci_mod.simulationStep()
                        traci_mod.simulationStep()
                        traci_mod.simulationStep()
                        traci_mod.trafficlight.setPhase(TL_ID, next_phase)
                        traci_mod.trafficlight.setPhaseDuration(TL_ID, 9999)
                    except Exception:
                        pass
                    current_phase = next_phase
                    green_timer = 0.0

                r = self._compute_reward(traci_mod)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(r)
                values.append(value)

                try:
                    traci_mod.simulationStep()
                except Exception:
                    break

            # Close SUMO
            try:
                traci_mod.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass

            if len(states) < 10:
                continue

            # ── PPO Update ────────────────────────────────────────────────────
            ep_reward = sum(rewards)

            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)

            states_t   = torch.tensor(np.array(states), dtype=torch.float32)
            actions_t  = torch.tensor(actions, dtype=torch.long)
            old_lp_t   = torch.stack(log_probs).detach()
            returns_t  = torch.tensor(returns, dtype=torch.float32)
            values_t   = torch.stack([v.detach() for v in values])
            advantages = returns_t - values_t

            # Normalise advantages
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Mini-batch PPO updates
            for _ in range(EPOCHS_PER_UPDATE):
                new_lp, new_vals, entropy = self.model.evaluate(states_t, actions_t)
                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(new_vals, returns_t)
                entropy_bonus = -0.01 * entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            if ep_reward > best_reward:
                best_reward = ep_reward
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                torch.save(self.model.state_dict(), save_path)

            if (ep + 1) % 10 == 0:
                print(f"  [EP {ep+1:4d}/{episodes}]  reward={ep_reward:8.1f}  best={best_reward:8.1f}")

        print(f"\n[AI-TRAIN] Done. Best reward: {best_reward:.1f}")
        print(f"[AI-TRAIN] Checkpoint saved to: {save_path}")
        return save_path


# ═════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Traffic Light Controller")
    parser.add_argument("--train",    action="store_true", help="Run PPO training loop")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--steps",    type=int, default=300, help="Steps per episode")
    parser.add_argument("--cfg",      default="my_config.sumocfg", help="SUMO config file")
    parser.add_argument("--port",     type=int, default=8820, help="TraCI port for training")
    parser.add_argument("--output",   default="models/mappo_checkpoint.pt", help="Checkpoint path")
    args = parser.parse_args()

    if args.train:
        trainer = PPOTrainer(args.cfg, port=args.port)
        trainer.train(episodes=args.episodes, steps_per_ep=args.steps, save_path=args.output)
    else:
        print("Use --train to start training.")
        print("For inference, import AITrafficController from this module.")
