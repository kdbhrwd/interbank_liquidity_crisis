import numpy as np
import pygame
import pandas as pd
import random
import math

NUM_AGENTS   = 5
INIT_CAPITAL = 100
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.15
NUM_EPISODES = 2000
STEPS_PER_EP = 20
SHOCK_PROB   = 0.15

ACTION_PCT    = [0.0, 0.25, 0.50, 0.75, 1.00]
ACTION_LABELS = ["HOARD", "CAUTIOUS", "BALANCED", "LIQUID", "FLUSH"]
STRAT_COLORS  = [(208,96,104),(222,142,80),(210,188,80),(92,180,160),(64,175,120)]
BANK_NAMES    = ["FIRST NAT.", "PAC.TRUST", "CENTURY", "METRO CAP", "ATLAS RES."]
BASE_LIQ      = [0.010, 0.030]
BASE_ILLIQ    = [0.060, 0.060]

W, H    = 1280, 820
FPS     = 60
TOP_H   = 80
LP_X, LP_W = 4, 192
RP_X    = 979
RP_W    = W - RP_X - 4
BK_CX   = [291, 439, 587, 735, 883]
BK_W, BK_H, BK_TOP = 98, 264, TOP_H + 16

STEP_MS     = [3000, 1500, 750, 350, 120, 10]
SPEED_NAMES = ["SLOW", "NORM", "FAST", "2x", "4x", "MAX"]

BG      = (11,14,20)
PANEL   = (17,21,29)
PANEL2  = (24,29,39)
HAIR    = (40,48,62)
LIQ     = (64,175,120)
ILLIQ   = (224,96,100)
SHOKC   = (218,175,62)
BAILC   = (96,156,228)
TEXT    = (225,232,240)
DIM     = (120,132,148)
ACCENT  = (132,144,228)
ACC2    = (72,172,178)
GOLD    = (220,184,92)
WARN    = (228,164,92)
GOOD    = (92,188,132)
NETC    = (32,40,54)
ACTIVE  = (160,156,210)

TOOLS = [
    dict(key="b", label="BAILOUT",      desc="Restore target bank",  cost=50, earn=0,  sel=True,  clr=BAILC,            kcode=pygame.K_b),
    dict(key="q", label="QE INJECT",    desc="Add +25 capital",      cost=35, earn=0,  sel=True,  clr=ACC2,             kcode=pygame.K_q),
    dict(key="e", label="MANDATE LIQ",  desc="Force LIQUID x8 steps",cost=20, earn=0,  sel=True,  clr=BAILC,            kcode=pygame.K_e),
    dict(key="t", label="TAX HOARDER",  desc="Penalise HOARD bank",  cost=0,  earn=15, sel=True,  clr=WARN,             kcode=pygame.K_t),
    dict(key="c", label="RATE CUT",     desc="Boost liq rates x15",  cost=60, earn=0,  sel=False, clr=GOOD,             kcode=pygame.K_c),
    dict(key="x", label="STRESS TEST",  desc="Manual shock on bank", cost=0,  earn=0,  sel=True,  clr=SHOKC,            kcode=pygame.K_x),
    dict(key="r", label="RATE TOGGLE",  desc="Flip Fed rate",        cost=0,  earn=0,  sel=False, clr=ACCENT,           kcode=pygame.K_r),
]


class BankAgent:
    def __init__(self, idx):
        self.id = idx
        self.capital  = INIT_CAPITAL
        self.liquid   = INIT_CAPITAL * 0.5
        self.illiquid = INIT_CAPITAL * 0.5
        self.q = np.zeros((3,3,2,NUM_AGENTS))
        self.last_action  = 2
        self.coop_log     = []
        self.reward_log   = []
        self.mandate_turns   = 0
        self.mandated_action = 3
        self.qe_bonus = 0.0
        self.bankrupt = False

    def tier(self, pct):
        return 0 if pct < 0.25 else (1 if pct < 0.75 else 2)

    def eff_cap(self):
        return max(0, min(self.capital, INIT_CAPITAL)) + self.qe_bonus

    def get_state(self, agents, fed):
        own = self.liquid / max(self.eff_cap(), 1)
        tot = max(sum(a.eff_cap() for a in agents), 1)
        avg = sum(a.liquid for a in agents) / tot
        return (self.tier(own), self.tier(avg), fed)

    def act(self, s):
        if self.bankrupt: return 0
        if self.mandate_turns > 0:
            self.mandate_turns -= 1
            a = self.mandated_action
        elif random.random() < EPSILON:
            a = random.randint(0, NUM_AGENTS-1)
        else:
            a = int(np.argmax(self.q[s]))
        self.last_action = a
        return a

    def apply_action(self, a):
        if self.bankrupt:
            self.liquid = 0
            self.illiquid = 0
            self.capital = 0
            return
        pct = ACTION_PCT[a]
        ec  = self.eff_cap()
        self.liquid   = ec * pct
        self.illiquid = ec * (1.0 - pct)
        self.coop_log.append(pct)
        if len(self.coop_log) > 60: self.coop_log.pop(0)

    def learn(self, s, a, r, ns):
        self.q[s][a] += ALPHA * (r + GAMMA * float(np.max(self.q[ns])) - self.q[s][a])
        self.reward_log.append(r)
        if len(self.reward_log) > 60: self.reward_log.pop(0)

    @property
    def coop_score(self):
        if not self.coop_log: return 0.5
        return sum(1 for x in self.coop_log if x >= 0.25) / len(self.coop_log)

    @property
    def avg_reward(self):
        if not self.reward_log: return 0.0
        return sum(self.reward_log) / len(self.reward_log)


class Player:
    def __init__(self):
        self.budget = 500
        self.MAX_BUDGET = 500
        self.health = 100.0
        self.selected = -1
        self.collapses = 0
        self.interventions = 0
        self.rate_cut_turns = 0
        self.health_log = []
        self.fed_rate = 0

    def rating(self):
        h = self.health
        if h <= 0:  return "CRISIS", ILLIQ
        if h < 15:  return "D",      ILLIQ
        if h < 25:  return "C",      (230,80,80)
        if h < 35:  return "CCC",    WARN
        if h < 45:  return "CC",     WARN
        if h < 55:  return "B",      (220,190,60)
        if h < 65:  return "BB",     (210,200,70)
        if h < 75:  return "BBB",    (170,220,80)
        if h < 85:  return "A",      GOOD
        if h < 95:  return "AA",     LIQ
        return "AAA", (80,230,180)

    def regen(self):
        self.budget = min(self.MAX_BUDGET, self.budget + 3)

    def spend(self, n):
        self.budget = max(0, self.budget - n)
        self.interventions += 1

    def earn(self, n):
        self.budget = min(self.MAX_BUDGET, self.budget + n)

    def damage(self, n):
        self.health = max(0.0, self.health - n)
        self.health_log.append(self.health)
        if len(self.health_log) > 200: self.health_log.pop(0)

    def heal(self, n):
        self.health = min(100.0, self.health + n)
        self.health_log.append(self.health)
        if len(self.health_log) > 200: self.health_log.pop(0)

    @property
    def game_over(self):
        return self.health <= 0


class Ripple:
    def __init__(self, cx, cy, color, max_r=120, speed=4.0):
        self.cx, self.cy = cx, cy
        self.color, self.max_r, self.speed = color, max_r, speed
        self.r, self.done = 8.0, False

    def update(self):
        self.r += self.speed
        self.done = self.r >= self.max_r

    def draw(self, surf):
        if self.done: return
        a  = int(210 * (1 - self.r/self.max_r))
        sz = int(self.max_r)*2 + 10
        s  = pygame.Surface((sz, sz), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, a), (sz//2,sz//2), int(self.r), 2)
        surf.blit(s, (self.cx-sz//2, self.cy-sz//2))


class FlowDot:
    def __init__(self, sx, sy, ex, ey, color):
        self.sx,self.sy,self.ex,self.ey = sx,sy,ex,ey
        self.color = color
        self.t     = random.uniform(0, 0.28)
        self.speed = random.uniform(0.018, 0.034)
        self.done  = False

    def update(self):
        self.t += self.speed
        self.done = self.t >= 1.0

    def draw(self, surf):
        if self.done: return
        x = self.sx + (self.ex-self.sx)*self.t
        y = self.sy + (self.ey-self.sy)*self.t
        a = int(255 * max(0, 1-self.t))
        s = pygame.Surface((8,8), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, a), (4,4), 4)
        surf.blit(s, (int(x)-4, int(y)-4))


def lc(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i]+(b[i]-a[i])*t) for i in range(3))

def rr(surf, color, rect, r=5, w=0):
    x,y,ww,hh = int(rect[0]),int(rect[1]),int(rect[2]),int(rect[3])
    if len(color)==4:
        s = pygame.Surface((ww,hh), pygame.SRCALPHA)
        pygame.draw.rect(s, color, (0,0,ww,hh), border_radius=r, width=w)
        surf.blit(s, (x,y))
    else:
        pygame.draw.rect(surf, color, (x,y,ww,hh), border_radius=r, width=w)

def tx(surf, msg, font, color, cx, cy, anchor="center"):
    s = font.render(str(msg), True, color)
    r = s.get_rect()
    setattr(r, anchor, (int(cx), int(cy)))
    surf.blit(s, r)


def draw_top_hud(surf, player, ep, step, speed_idx, fonts):
    # Header bar - single flat surface, hairline border beneath
    rr(surf, PANEL, (0,0,W,TOP_H), 0)
    pygame.draw.line(surf, HAIR, (0,TOP_H-1), (W,TOP_H-1), 1)

    # ---- Title block --------------------------------------------------------
    tx(surf, "CENTRAL BANK",     fonts["sm"], TEXT,   LP_X+14, 28, anchor="midleft")
    tx(surf, "Federal Reserve",  fonts["xs"], DIM,    LP_X+14, 46, anchor="midleft")
    tx(surf, "Stabilize network", fonts["xs"], DIM,   LP_X+14, 60, anchor="midleft")
    pygame.draw.line(surf, HAIR, (LP_X+LP_W-2, 14), (LP_X+LP_W-2, TOP_H-14), 1)

    def _section(sx, sw, title):
        tx(surf, title, fonts["xs"], DIM, sx+10, 18, anchor="midleft")
        pygame.draw.line(surf, HAIR, (sx+sw-1, 14), (sx+sw-1, TOP_H-14), 1)

    # ---- Systemic Health ----------------------------------------------------
    hx, hw = LP_X+LP_W+8, 230
    _section(hx, hw, "SYSTEMIC HEALTH")
    rr(surf, PANEL2, (hx+10, 32, hw-20, 10), 3)
    hc = lc(ILLIQ, GOOD, player.health/100)
    rr(surf, hc, (hx+10, 32, int((hw-20)*player.health/100), 10), 3)
    tx(surf, f"{player.health:.0f} / 100", fonts["xs"], TEXT, hx+10, 56, anchor="midleft")
    rating, rc = player.rating()
    tx(surf, rating, fonts["md"], rc, hx+hw-14, 54, anchor="midright")

    # ---- Budget -------------------------------------------------------------
    bx, bw = hx+hw, 200
    _section(bx, bw, "POLICY BUDGET")
    rr(surf, PANEL2, (bx+10, 32, bw-20, 10), 3)
    bcol = lc(ILLIQ, ACC2, player.budget/player.MAX_BUDGET)
    rr(surf, bcol, (bx+10, 32, int((bw-20)*player.budget/player.MAX_BUDGET), 10), 3)
    tx(surf, f"${player.budget}", fonts["xs"], TEXT, bx+10, 56, anchor="midleft")
    tx(surf, f"/ ${player.MAX_BUDGET}", fonts["xs"], DIM, bx+bw-14, 56, anchor="midright")

    # ---- Fed / Speed --------------------------------------------------------
    sx2, sw2 = bx+bw, 180
    _section(sx2, sw2, "MONETARY POLICY")
    fed_c = WARN if player.fed_rate else GOOD
    tx(surf, f"Fed {'HIGH 3%' if player.fed_rate else 'LOW 1%'}", fonts["xs"], fed_c, sx2+10, 36, anchor="midleft")
    rc2 = WARN if player.rate_cut_turns>0 else DIM
    tx(surf, f"Rate Cut {'ON '+str(player.rate_cut_turns)+'t' if player.rate_cut_turns else 'OFF'}", fonts["xs"], rc2, sx2+10, 52, anchor="midleft")
    tx(surf, SPEED_NAMES[speed_idx], fonts["xs"], ACCENT, sx2+sw2-14, 52, anchor="midright")

    # ---- Episode progress ---------------------------------------------------
    ex, ew = sx2+sw2, 200
    _section(ex, ew, "PROGRESS")
    tx(surf, f"EP {ep:04d}/{NUM_EPISODES}", fonts["xs"], TEXT, ex+10, 36, anchor="midleft")
    tx(surf, f"Step {step:02d}/{STEPS_PER_EP}", fonts["xs"], DIM, ex+10, 52, anchor="midleft")
    col_c = ILLIQ if player.collapses else DIM
    tx(surf, f"Collapses {player.collapses}", fonts["xs"], col_c, ex+ew-14, 52, anchor="midright")

    # ---- Interventions ------------------------------------------------------
    ix = ex+ew
    if ix+140 < W-RP_W-10:
        _section(ix, W-RP_W-ix-4, "INTERVENTIONS")
        tx(surf, str(player.interventions), fonts["md"], BAILC, ix+12, 46, anchor="midleft")
        tx(surf, "[1-5] select bank", fonts["xs"], DIM, ix+60, 46, anchor="midleft")


def draw_left_panel(surf, player, agents, tool_rects, hovered_tool, fonts):
    rr(surf, PANEL, (LP_X, TOP_H, LP_W, H-TOP_H-4), 8)
    x, y, w = LP_X+8, TOP_H+12, LP_W-16

    tx(surf, "POLICY TOOLS", fonts["xs"], DIM, x, y, anchor="topleft")
    y += 16
    pygame.draw.line(surf, HAIR, (x, y), (x+w, y), 1)
    y += 10

    # --- Target card --------------------------------------------------------
    rr(surf, PANEL2, (x, y, w, 86), 5)
    if player.selected >= 0:
        ag = agents[player.selected]
        sc = STRAT_COLORS[ag.last_action]
        pygame.draw.rect(surf, sc, (x, y, 3, 86))
        tx(surf, BANK_NAMES[player.selected], fonts["sm"], TEXT, x+10, y+10, anchor="topleft")
        tx(surf, ACTION_LABELS[ag.last_action], fonts["xs"], sc, x+10, y+26, anchor="topleft")
        tx(surf, f"L {ag.liquid:.0f}", fonts["xs"], LIQ,   x+10, y+44, anchor="topleft")
        tx(surf, f"I {ag.illiquid:.0f}", fonts["xs"], ILLIQ, x+60, y+44, anchor="topleft")
        tx(surf, f"C {ag.capital:+.0f}", fonts["xs"], TEXT if ag.capital>0 else ILLIQ, x+10, y+58, anchor="topleft")
        cw = w-20
        coop = ag.coop_score
        rr(surf, BG, (x+10, y+72, cw, 5), 2)
        rr(surf, lc(ILLIQ,LIQ,coop), (x+10, y+72, int(cw*coop), 5), 2)
        if ag.mandate_turns > 0:
            tx(surf, f"MANDATE {ag.mandate_turns}t", fonts["xs"], BAILC, x+w-10, y+60, anchor="topright")
        else:
            tx(surf, f"{coop*100:.0f}% coop", fonts["xs"], DIM, x+w-10, y+60, anchor="topright")
    else:
        tx(surf, "NO TARGET", fonts["xs"], DIM, x+10, y+14, anchor="topleft")
        tx(surf, "Click a bank or", fonts["xs"], DIM, x+10, y+34, anchor="topleft")
        tx(surf, "press [1-5] to", fonts["xs"], DIM, x+10, y+48, anchor="topleft")
        tx(surf, "select a target", fonts["xs"], DIM, x+10, y+62, anchor="topleft")
    y += 94

    # --- Tool buttons -------------------------------------------------------
    tool_rects.clear()
    for ti, td in enumerate(TOOLS):
        can = not td["sel"] or player.selected >= 0
        aff = player.budget >= td["cost"]
        live = can and aff
        hov = (hovered_tool == ti) and live
        clr = td["clr"] if live else DIM

        rect = (x, y, w, 34)
        tool_rects.append((rect, td["key"]))
        # Base
        rr(surf, PANEL2 if live else (20,24,32), rect, 5)
        # Left accent strip (indicates tool color / enabled state)
        pygame.draw.rect(surf, clr if live else HAIR, (x, y, 3, 34))
        # Hover: subtle outline only
        if hov:
            pygame.draw.rect(surf, clr, (x,y,w,34), 1, border_radius=5)

        # Hotkey badge
        tx(surf, td["key"].upper(), fonts["sm"], clr if live else DIM, x+14, y+17, anchor="midleft")

        # Label + description
        lbl_c = TEXT if live else DIM
        dsc_c = DIM  if live else (65,75,88)
        tx(surf, td["label"], fonts["xs"], lbl_c, x+30, y+10, anchor="midleft")
        tx(surf, td["desc"],  fonts["xs"], dsc_c, x+30, y+24, anchor="midleft")

        # Cost / earn indicator
        if td["earn"]:
            cs, cc = f"+${td['earn']}", GOOD if live else DIM
        elif td["cost"]:
            cs, cc = f"-${td['cost']}", WARN if live else DIM
        else:
            cs, cc = "FREE", ACCENT if live else DIM
        tx(surf, cs, fonts["xs"], cc, x+w-6, y+17, anchor="midright")
        y += 38

    y += 6
    pygame.draw.line(surf, HAIR, (x, y), (x+w, y), 1)
    y += 10

    tx(surf, "CONTROLS", fonts["xs"], DIM, x, y, anchor="topleft")
    y += 16
    for line in ["↑ / ↓     speed", "space     pause", "1 - 5     select", "esc       quit"]:
        tx(surf, line, fonts["xs"], DIM, x, y, anchor="topleft")
        y += 14


def draw_bank_col(surf, agent, idx, disp_liq, selected, active_turn, shocked, bail_window, fonts):
    cx = BK_CX[idx]
    bx = cx - BK_W//2
    by = BK_TOP

    # Card backdrop
    rr(surf, PANEL, (bx-8, by-8, BK_W+16, BK_H+16), 6)

    if agent.bankrupt:
        rr(surf, (32,34,40), (bx, by, BK_W, BK_H), 4)
        tx(surf, "BANKRUPT", fonts["xs"], ILLIQ, cx, by+110)
        tx(surf, BANK_NAMES[idx], fonts["sm"], DIM, cx, by+BK_H+14)
        tx(surf, "L 0   I 0", fonts["xs"], DIM, cx, by+BK_H+30)
        tx(surf, f"C {agent.capital:+.0f}", fonts["xs"], ILLIQ, cx, by+BK_H+44)
        return

    # Liquid / Illiquid stack
    pct = max(0.0, min(1.0, disp_liq / max(agent.eff_cap(), 1)))
    lh  = int(BK_H * pct)
    ih  = BK_H - lh
    if ih > 0: rr(surf, ILLIQ, (bx, by, BK_W, ih),    4)
    if lh > 0: rr(surf, LIQ,   (bx, by+ih, BK_W, lh), 4)

    if agent.qe_bonus > 0:
        rr(surf, (*ACC2, 42), (bx, by, BK_W, BK_H), 4)

    # State indicators - no pulsing
    if shocked:
        pygame.draw.rect(surf, SHOKC, (bx-8,by-8,BK_W+16,BK_H+16), 2, border_radius=6)
    elif selected:
        pygame.draw.rect(surf, GOLD, (bx-8,by-8,BK_W+16,BK_H+16), 2, border_radius=6)
    elif agent.mandate_turns > 0:
        pygame.draw.rect(surf, BAILC, (bx-8,by-8,BK_W+16,BK_H+16), 1, border_radius=6)
    elif active_turn:
        pygame.draw.rect(surf, HAIR, (bx-8,by-8,BK_W+16,BK_H+16), 1, border_radius=6)
        pygame.draw.rect(surf, ACTIVE, (bx-8, by+BK_H+6, BK_W+16, 2))

    # Shock banner above card
    if shocked:
        if bail_window:
            rr(surf, BAILC, (bx-8, by-30, BK_W+16, 18), 3)
            tx(surf, "BAIL  [B]", fonts["xs"], BG, cx, by-21)
        else:
            tx(surf, "SHOCK", fonts["xs"], SHOKC, cx, by-20)

    # Strategy label band on top of stack
    sc = STRAT_COLORS[agent.last_action]
    rr(surf, sc, (bx+4, by+6, BK_W-8, 16), 4)
    tx(surf, ACTION_LABELS[agent.last_action], fonts["xs"], BG, cx, by+14)

    # Tier label above card (hidden during bail)
    tier_lbls = ["STRESSED","MODERATE","STABLE"]
    tier_clrs = [ILLIQ, WARN, LIQ]
    own_t     = agent.tier(agent.liquid / max(agent.eff_cap(),1))
    if not (shocked and bail_window):
        tx(surf, tier_lbls[own_t], fonts["xs"], tier_clrs[own_t], cx, by-20)

    # Bank name + stats below card
    tx(surf, BANK_NAMES[idx], fonts["sm"], GOLD if selected else TEXT, cx, by+BK_H+14)
    tx(surf, f"L {agent.liquid:.0f}   I {agent.illiquid:.0f}", fonts["xs"], DIM, cx, by+BK_H+30)
    cap_c = TEXT if agent.capital > 0 else ILLIQ
    tx(surf, f"C {agent.capital:+.0f}", fonts["xs"], cap_c, cx, by+BK_H+42)

    # Coop bar
    bw = BK_W - 12
    c2 = agent.coop_score
    rr(surf, BG, (bx+6, by+BK_H+56, bw, 4), 2)
    rr(surf, lc(ILLIQ,LIQ,c2), (bx+6, by+BK_H+56, int(bw*c2), 4), 2)
    tx(surf, f"{c2*100:.0f}% coop", fonts["xs"], DIM, cx, by+BK_H+70)

    # Status badges
    if agent.qe_bonus > 0:
        tx(surf, f"QE +{agent.qe_bonus:.0f}", fonts["xs"], ACC2, cx, by+BK_H+84)
    elif agent.mandate_turns > 0:
        tx(surf, f"MAND {agent.mandate_turns}t", fonts["xs"], BAILC, cx, by+BK_H+84)


def draw_right_panel(surf, agents, player, ep, step, event_log, liq_hist, fonts):
    x, w = RP_X, RP_W
    rr(surf, PANEL, (x, TOP_H, w, H-TOP_H-4), 8)
    y = TOP_H + 14

    def _heading(lbl, yy):
        tx(surf, lbl, fonts["xs"], DIM, x+12, yy, anchor="topleft")
        pygame.draw.line(surf, HAIR, (x+12, yy+14), (x+w-12, yy+14), 1)
        return yy + 22

    y = _heading("HEALTH TREND", y)
    if len(player.health_log) > 2:
        sh, bw = 34, w-24
        recent = player.health_log[-60:]
        if len(recent) > 1:
            mx2,mn2 = max(recent),min(recent)
            sp2 = max(mx2-mn2,1)
            pts = [(x+12+int(i/max(len(recent)-1,1)*(bw-4)),
                    y+sh-4-int((v-mn2)/sp2*(sh-8)))
                   for i,v in enumerate(recent)]
            if len(pts)>1: pygame.draw.lines(surf, lc(ILLIQ,GOOD,player.health/100), False, pts, 2)
        y += sh + 10
    else:
        y += 10

    y = _heading("STRATEGY MATRIX", y)
    for i, ag in enumerate(agents):
        ay = y + i*28
        sc = STRAT_COLORS[ag.last_action]
        # Left accent strip
        pygame.draw.rect(surf, sc, (x+12, ay+3, 3, 20))
        # Action tag
        tx(surf, ACTION_LABELS[ag.last_action][:4], fonts["xs"], sc, x+22, ay+13, anchor="midleft")
        # Name
        nm_c = GOLD if player.selected==i else TEXT
        tx(surf, BANK_NAMES[i][:8], fonts["xs"], nm_c, x+66, ay+13, anchor="midleft")
        # Coop bar
        bw2 = w-170
        coop = ag.coop_score
        rr(surf, BG, (x+128, ay+9, bw2, 8), 2)
        rr(surf, lc(ILLIQ,LIQ,coop), (x+128, ay+9, int(bw2*coop), 8), 2)
        # Reward
        rw = ag.avg_reward
        tx(surf, f"{rw:+.1f}", fonts["xs"], LIQ if rw>=0 else ILLIQ, x+w-12, ay+13, anchor="midright")
        if ag.mandate_turns > 0:
            pygame.draw.rect(surf, BAILC, (x+12, ay+2, 3, 22))
    y += NUM_AGENTS*28 + 8

    # System metrics
    y = _heading("SYSTEM METRICS", y)
    acts   = [a.last_action for a in agents]
    dom    = max(set(acts), key=acts.count)
    coop_s = sum(a.coop_score for a in agents)/NUM_AGENTS
    nash_d = sum(1 for a in acts if a!=dom)/NUM_AGENTS
    tot_l  = sum(a.liquid for a in agents)
    risk   = 1.0 - min(tot_l/(NUM_AGENTS*45), 1.0)
    pd_t   = (1-coop_s)*risk

    mets = [
        ("Dominant",       ACTION_LABELS[dom],      STRAT_COLORS[dom]),
        ("Cooperation",    f"{coop_s*100:.0f}%",    lc(ILLIQ,LIQ,coop_s)),
        ("Nash Deviation", f"{nash_d*100:.0f}%",    lc(LIQ,ILLIQ,nash_d)),
        ("Contagion Risk", f"{risk*100:.0f}%",      lc(LIQ,ILLIQ,risk)),
        ("PD Tension",     f"{pd_t*100:.0f}%",      lc(GOOD,ILLIQ,pd_t)),
    ]
    for label, val, vc in mets:
        tx(surf, label, fonts["xs"], DIM, x+14, y+10, anchor="midleft")
        tx(surf, val,   fonts["xs"], vc,  x+w-14, y+10, anchor="midright")
        y += 20
    y += 4

    # Regime banner - single line, no box
    banner = None
    if coop_s < 0.18:
        banner = ("TRAGEDY OF THE COMMONS", WARN)
    elif risk > 0.72:
        banner = ("SYSTEMIC CRISIS IMMINENT", ILLIQ)
    elif nash_d < 0.1 and coop_s > 0.55:
        banner = ("NASH EQ: COOPERATING", GOOD)
    if banner:
        rr(surf, (*banner[1], 30), (x+12, y, w-24, 22), 4)
        tx(surf, banner[0], fonts["xs"], banner[1], x+w//2, y+11)
        y += 28
    else:
        y += 4

    y = _heading("EVENT LOG", y)
    for ev in event_log[-5:]:
        tx(surf, ev["msg"][:38], fonts["xs"], ev.get("color",DIM), x+14, y+8, anchor="midleft")
        y += 18

    remaining = (H-8) - y - 24
    if len(liq_hist) > 3 and remaining > 52:
        y += 6
        sh2 = min(remaining-18, 56)
        tx(surf, "LIQUIDITY TREND", fonts["xs"], DIM, x+12, y, anchor="topleft")
        y += 14
        recent2 = liq_hist[-60:]
        if len(recent2)>1:
            mx3,mn3 = max(recent2),min(recent2)
            sp3 = max(mx3-mn3,1)
            bw3 = w-24
            pts2 = [(x+12+int(i/max(len(recent2)-1,1)*(bw3-4)),
                     y+sh2-4-int((v-mn3)/sp3*(sh2-8)))
                    for i,v in enumerate(recent2)]
            if len(pts2)>1: pygame.draw.lines(surf, ACC2, False, pts2, 1)


def draw_overlay(surf, title, subtitle, stats, hint, title_clr, border_clr, fonts):
    s = pygame.Surface((W,H), pygame.SRCALPHA)
    s.fill((0,0,0,170))
    surf.blit(s,(0,0))
    bx,by,bw,bh = W//2-240, H//2-170, 480, 340
    rr(surf, PANEL, (bx,by,bw,bh), 16)
    pygame.draw.rect(surf, border_clr, (bx,by,bw,bh), 2, border_radius=16)
    tx(surf, title,    fonts["xl"], title_clr, W//2, by+36)
    tx(surf, subtitle, fonts["sm"], TEXT,      W//2, by+68)
    for i,(label,val,clr) in enumerate(stats):
        tx(surf, f"{label}: {val}", fonts["sm"], clr, W//2, by+102+i*28)
    rr(surf,(20,30,20),(bx+40,by+bh-64,bw-80,34),8)
    pygame.draw.rect(surf,border_clr,(bx+40,by+bh-64,bw-80,34),1,border_radius=8)
    tx(surf,hint,fonts["xs"],TEXT,W//2,by+bh-47)


def apply_tool(key, player, agents, ripples, flow_dots, event_log, shock_target_ref):
    def log(msg, clr=DIM): event_log.append({"msg":msg,"color":clr})
    if agents[0].bankrupt:
        log("Bankrupt! You may only spectate.", ILLIQ)
        return
    sel = player.selected

    if key=="r":
        player.fed_rate = 1-player.fed_rate
        fc = (251,113,133) if player.fed_rate else (52,200,150)
        log(f"Fed Rate → {'HIGH' if player.fed_rate else 'LOW'}", fc)
        return

    if key=="c":
        if not player.budget>=60: return
        player.spend(60)
        player.rate_cut_turns=15
        log("Emergency Rate Cut active (15 steps)", GOOD)
        return

    if sel<0:
        log("Select a bank first!", DIM); return

    ag = agents[sel]
    cx,cy = BK_CX[sel], BK_TOP+BK_H//2

    if key=="b":
        if not player.budget>=50: log("Insufficient budget",ILLIQ); return
        if shock_target_ref[0]==sel:
            shock_target_ref[0]=-1
        player.spend(50)
        ag.liquid   = ag.eff_cap()*0.55
        ag.illiquid = ag.eff_cap()*0.45
        ripples += [Ripple(cx,cy,BAILC,max_r=110), Ripple(cx,cy,ACC2,speed=2.6,max_r=85)]
        log(f"BAILOUT: {BANK_NAMES[sel]} rescued", BAILC)

    elif key=="q":
        if not player.budget>=35: log("Insufficient budget",ILLIQ); return
        player.spend(35)
        ag.qe_bonus += 25
        ag.apply_action(ag.last_action)
        ripples.append(Ripple(cx,cy,ACC2,max_r=90))
        log(f"QE +25 capital: {BANK_NAMES[sel]}", ACC2)

    elif key=="e":
        if not player.budget>=20: log("Insufficient budget",ILLIQ); return
        player.spend(20)
        ag.mandate_turns=8; ag.mandated_action=3
        log(f"MANDATE LIQUID: {BANK_NAMES[sel]} x8 steps", (100,180,255))

    elif key=="t":
        if ag.last_action>1: log("Bank is not hoarding",DIM); return
        ag.capital -= 20
        ag.apply_action(ag.last_action)
        player.earn(15)
        log(f"TAXED HOARDER: {BANK_NAMES[sel]} -20 cap", WARN)

    elif key=="x":
        amt=random.randint(40,90)
        ag.capital -= amt
        ag.liquid=max(0,ag.liquid-amt)
        ripples+=[Ripple(cx,cy,SHOKC),Ripple(cx,cy,(255,180,50),speed=2,max_r=80)]
        log(f"STRESS TEST: {BANK_NAMES[sel]} -{amt}", SHOKC)


def classify_regime(coop, nash_dev, risk):
    if risk > 0.72 and coop < 0.35:
        return "CRISIS"
    if coop < 0.18:
        return "TRAGEDY"
    if coop > 0.55 and nash_dev < 0.10:
        return "NASH-COOP"
    return "MIXED"


REGIME_CLR = {
    "NASH-COOP": GOOD,
    "TRAGEDY":   ILLIQ,
    "CRISIS":    ILLIQ,
    "MIXED":     WARN,
}

REGIME_EXPL = {
    "NASH-COOP": "Cooperative Nash eq reached: banks sustain liquidity (folk-theorem outcome).",
    "TRAGEDY":   "Tragedy of the commons: defection (HOARD) dominates, system fragile.",
    "CRISIS":    "Systemic crisis: low liquidity + high contagion risk. Intervene now.",
    "MIXED":     "No stable equilibrium yet - agents still exploring strategies.",
}


def draw_nash_chart(surf, nash_hist, fonts, x, y, w, h):
    rr(surf, PANEL, (x, y, w, h), 8)
    tx(surf, "NASH EQUILIBRIUM TRAJECTORY", fonts["sm"], TEXT, x+14, y+14, anchor="midleft")
    tx(surf, "Applied Game Theory  ::  Repeated N-Player Prisoner's Dilemma", fonts["xs"], DIM, x+w-14, y+14, anchor="midright")
    pygame.draw.line(surf, HAIR, (x+10, y+28), (x+w-10, y+28), 1)

    pad_l, pad_r, pad_t, pad_b = 44, 170, 36, 32
    cx0 = x + pad_l
    cy0 = y + pad_t
    cw  = w - pad_l - pad_r
    ch  = h - pad_t - pad_b

    rr(surf, PANEL2, (cx0, cy0, cw, ch), 4)

    # Green tint: coop >= 0.55 (cooperative Nash eq zone)
    zt = cy0 + ch - int(0.55 * ch)
    s1 = pygame.Surface((cw, zt - cy0), pygame.SRCALPHA)
    s1.fill((50, 200, 120, 22))
    surf.blit(s1, (cx0, cy0))
    # Red tint: coop <= 0.18 (tragedy zone)
    tt = cy0 + ch - int(0.18 * ch)
    s2 = pygame.Surface((cw, (cy0+ch) - tt), pygame.SRCALPHA)
    s2.fill((220, 60, 90, 22))
    surf.blit(s2, (cx0, tt))

    # Threshold dashed lines
    for frac, clr in [(0.55, GOOD), (0.18, ILLIQ), (0.10, ACC2)]:
        gy = cy0 + ch - int(frac * ch)
        for dx in range(0, cw, 8):
            pygame.draw.line(surf, clr, (cx0+dx, gy), (cx0+dx+4, gy), 1)

    for frac, lbl in [(0.0,"0%"),(0.25,"25%"),(0.5,"50%"),(0.75,"75%"),(1.0,"100%")]:
        gy = cy0 + ch - int(frac * ch)
        pygame.draw.line(surf, (30,42,74), (cx0, gy), (cx0+cw, gy), 1)
        tx(surf, lbl, fonts["xs"], DIM, cx0-6, gy, anchor="midright")

    if len(nash_hist) >= 2:
        recent = nash_hist[-160:]
        n = len(recent)
        pts_coop, pts_dev = [], []
        for i, m in enumerate(recent):
            px = cx0 + int(i/(n-1) * cw)
            pts_coop.append((px, cy0 + ch - int(max(0,min(1,m["coop"])) * ch)))
            pts_dev.append( (px, cy0 + ch - int(max(0,min(1,m["nash_dev"])) * ch)))
        pygame.draw.lines(surf, LIQ,   False, pts_coop, 2)
        pygame.draw.lines(surf, ILLIQ, False, pts_dev,  2)
        pygame.draw.circle(surf, GOLD, pts_coop[-1], 4)
        pygame.draw.circle(surf, GOLD, pts_dev[-1],  4, 1)
    else:
        tx(surf, "collecting data...", fonts["xs"], DIM, cx0+cw//2, cy0+ch//2)

    tx(surf, f"last {min(len(nash_hist),160)} steps  --->", fonts["xs"], DIM, cx0+cw//2, cy0+ch+12)

    lx = cx0 + cw + 14
    ly = cy0

    pygame.draw.line(surf, LIQ, (lx, ly+6), (lx+22, ly+6), 2)
    tx(surf, "Cooperation",    fonts["xs"], LIQ,   lx+28, ly+6, anchor="midleft")
    ly += 16
    pygame.draw.line(surf, ILLIQ, (lx, ly+6), (lx+22, ly+6), 2)
    tx(surf, "Nash Deviation", fonts["xs"], ILLIQ, lx+28, ly+6, anchor="midleft")
    ly += 20

    pygame.draw.rect(surf, (50,200,120), (lx, ly, 14, 10))
    tx(surf, "Coop Nash Zone", fonts["xs"], GOOD,  lx+18, ly+5, anchor="midleft")
    ly += 14
    pygame.draw.rect(surf, (220,60,90),  (lx, ly, 14, 10))
    tx(surf, "Tragedy Zone",   fonts["xs"], ILLIQ, lx+18, ly+5, anchor="midleft")
    ly += 18

    cur = nash_hist[-1] if nash_hist else None
    regime = cur["regime"] if cur else "MIXED"
    rc = REGIME_CLR.get(regime, DIM)
    tx(surf, "CURRENT REGIME", fonts["xs"], DIM, lx, ly, anchor="topleft")
    ly += 14
    rr(surf, PANEL2, (lx, ly, 148, 24), 5)
    pygame.draw.rect(surf, rc, (lx, ly, 148, 24), 1, border_radius=5)
    tx(surf, regime, fonts["sm"], rc, lx+74, ly+12)
    ly += 30
    if cur:
        tx(surf, f"Coop:     {cur['coop']*100:5.0f}%",     fonts["xs"], LIQ,   lx, ly, anchor="topleft"); ly += 13
        tx(surf, f"Dev:      {cur['nash_dev']*100:5.0f}%", fonts["xs"], ILLIQ, lx, ly, anchor="topleft"); ly += 13
        tx(surf, f"Risk:     {cur['risk']*100:5.0f}%",     fonts["xs"], WARN,  lx, ly, anchor="topleft")

    msg = REGIME_EXPL.get(regime, "")
    tx(surf, msg, fonts["xs"], TEXT, x+w//2, y+h-14)


def run():
    pygame.init()
    info = pygame.display.Info()
    avail_w = max(640, info.current_w - 80)
    avail_h = max(480, info.current_h - 120)
    scale = min(1.0, avail_w / W, avail_h / H)
    win_w, win_h = int(W * scale), int(H * scale)
    window = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Interbank Liquidity Crisis - Central Bank Command")
    screen = pygame.Surface((W, H)) if scale < 1.0 else window
    clock = pygame.time.Clock()

    fonts = {
        "xl": pygame.font.SysFont("Consolas",24,bold=True),
        "md": pygame.font.SysFont("Consolas",16,bold=True),
        "sm": pygame.font.SysFont("Consolas",13,bold=True),
        "xs": pygame.font.SysFont("Consolas",11),
    }

    def new_game():
        a=[BankAgent(i) for i in range(NUM_AGENTS)]
        p=Player()
        return a,p

    agents, player = new_game()
    speed_idx   = 1
    paused      = False
    episode     = 0
    step        = 0
    ep_history  = []
    event_log   = []
    liq_hist    = []
    nash_hist   = []
    ripples     = []
    flow_dots   = []
    disp_liq    = [a.liquid for a in agents]
    active_agt  = -1
    shock_ref   = [-1]
    shock_flash = 0
    phase       = "AGENT_TURN"
    agent_idx   = 0
    phase_start = pygame.time.get_ticks()
    step_data   = None
    game_state  = "PLAYING"
    tool_rects  = []
    hov_tool    = -1
    
    win_reason = ""
    lose_reason = ""

    pending_shock     = None
    pending_states    = None
    pending_actions   = None
    pending_rewards   = None

    def log(msg,clr=DIM): 
        event_log.append({"msg":msg,"color":clr})
        if len(event_log)>40: event_log.pop(0)

    def prepare_step():
        fed = player.fed_rate
        states  = [a.get_state(agents,fed) if not a.bankrupt else None for a in agents]
        actions = [a.act(s) for a,s in zip(agents,states)]
        for a,ac in zip(agents,actions): a.apply_action(ac)
        elr = BASE_LIQ[fed] + (0.02 if player.rate_cut_turns>0 else 0)
        eir = BASE_ILLIQ[fed]
        rewards = [a.liquid*elr + a.illiquid*eir if not a.bankrupt else 0 for a in agents]
        for i, a in enumerate(agents):
            if not a.bankrupt: a.capital += rewards[i]
        shock_info = None
        active_agents = [i for i, a in enumerate(agents) if not a.bankrupt]
        if active_agents and random.random() < SHOCK_PROB:
            si=random.choice(active_agents)
            sa=random.randint(40,90)
            shock_info={"idx":si,"amt":sa}
        return states,actions,rewards,shock_info

    def finalize_step(states,actions,rewards,shock_info,bailed):
        fed = player.fed_rate
        if shock_info and not bailed:
            si,sa = shock_info["idx"],shock_info["amt"]
            tgt   = agents[si]
            tgt.capital -= sa
            if tgt.liquid < sa:
                shortfall = sa - tgt.liquid
                sys_liq   = sum(a.liquid for a in agents if not a.bankrupt) - tgt.liquid
                if sys_liq < shortfall:
                    for j in range(NUM_AGENTS):
                        if not agents[j].bankrupt: rewards[j]-=100
                    player.damage(15)
                    player.collapses+=1
                    log("SYSTEMIC COLLAPSE! -15 health",ILLIQ)
                else:
                    rewards[si]-=20
                    td2=max(sys_liq,1)
                    for j,a in enumerate(agents):
                        if j!=si and not a.bankrupt:
                            impact = a.liquid/td2*shortfall
                            a.liquid=max(0,a.liquid-impact)
                            a.capital -= impact
                    log(f"{BANK_NAMES[si]}: cascade shock -{sa}",SHOKC)
                    player.damage(3)
            else:
                rewards[si]-=20; agents[si].liquid-=sa
                log(f"{BANK_NAMES[si]}: shock absorbed -{sa}",WARN)
                player.damage(1)
        elif shock_info and bailed:
            log(f"Bailout blocked shock on {BANK_NAMES[shock_info['idx']]}",BAILC)

        if not shock_info: player.heal(0.35)
        if player.rate_cut_turns>0: player.rate_cut_turns-=1
        player.regen()

        for i, a in enumerate(agents):
            if not a.bankrupt and a.capital <= 0:
                a.bankrupt = True
                a.capital = 0
                a.liquid = 0
                a.illiquid = 0
                log(f"BANKRUPT: {BANK_NAMES[i]} IS DEAD!", ILLIQ)

        ns=[a.get_state(agents,fed) if not a.bankrupt else None for a in agents]
        for i,a in enumerate(agents):
            if not a.bankrupt and states[i] is not None and ns[i] is not None:
                a.learn(states[i],actions[i],rewards[i],ns[i])

    running=True
    while running:
        clock.tick(FPS)
        now=pygame.time.get_ticks()
        ms =STEP_MS[speed_idx]
        _rmx,_rmy=pygame.mouse.get_pos()
        mx2,my2=(int(_rmx/scale), int(_rmy/scale)) if scale<1.0 else (_rmx,_rmy)

        hov_tool=-1
        for ti,(rect,_) in enumerate(tool_rects):
            rx,ry,rw,rh=rect
            if rx<=mx2<=rx+rw and ry<=my2<=ry+rh:
                hov_tool=ti; break

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False

            if game_state in ("GAME_OVER","WIN"):
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_RETURN:
                        agents,player=new_game()
                        episode=step=0; ep_history.clear(); event_log.clear()
                        liq_hist.clear(); nash_hist.clear(); ripples.clear(); flow_dots.clear()
                        disp_liq=[a.liquid for a in agents]; active_agt=-1
                        shock_ref=[-1]; shock_flash=0
                        phase="AGENT_TURN"; agent_idx=0; phase_start=now
                        pending_shock=pending_states=pending_actions=pending_rewards=None
                        game_state="PLAYING"
                    elif ev.key==pygame.K_ESCAPE: running=False
                continue

            if ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE: running=False
                if ev.key==pygame.K_SPACE:  paused=not paused
                if ev.key==pygame.K_UP:     speed_idx=min(speed_idx+1,5)
                if ev.key==pygame.K_DOWN:   speed_idx=max(speed_idx-1,0)
                for i,k in enumerate([pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5]):
                    if ev.key==k:
                        player.selected= i if player.selected!=i else -1
                for td in TOOLS:
                    if ev.key==td["kcode"]:
                        apply_tool(td["key"],player,agents,ripples,flow_dots,event_log,shock_ref)
                        break

            if ev.type==pygame.MOUSEBUTTONDOWN and game_state=="PLAYING":
                for i in range(NUM_AGENTS):
                    bx=BK_CX[i]-BK_W//2-10; by2=BK_TOP-10
                    if bx<=mx2<=bx+BK_W+20 and by2<=my2<=by2+BK_H+20:
                        player.selected= i if player.selected!=i else -1; break
                for (rect,key) in tool_rects:
                    rx,ry,rw,rh=rect
                    if rx<=mx2<=rx+rw and ry<=my2<=ry+rh:
                        apply_tool(key,player,agents,ripples,flow_dots,event_log,shock_ref); break

        if game_state=="PLAYING" and not paused:
            agent_ms=int(ms*0.13)

            if phase=="AGENT_TURN":
                active_agt=agent_idx
                if now-phase_start>=agent_ms:
                    agent_idx+=1; phase_start=now
                    if agent_idx>=NUM_AGENTS:
                        agent_idx=0; active_agt=-1
                        ps,pa,pr,si=prepare_step()
                        pending_states,pending_actions,pending_rewards,pending_shock=ps,pa,pr,si
                        step+=1
                        for i,a in enumerate(agents): disp_liq[i]=a.liquid
                        if si:
                            shock_ref[0]=si["idx"]; shock_flash=55
                            scx=BK_CX[si["idx"]]; scy=BK_TOP+BK_H//2
                            ripples+=[Ripple(scx,scy,SHOKC),Ripple(scx,scy,(255,180,50),speed=2.3,max_r=90)]
                            for j in range(NUM_AGENTS):
                                if j!=si["idx"]: flow_dots+=[FlowDot(BK_CX[j],BK_TOP+BK_H//2,scx,scy,LIQ) for _ in range(5)]
                            phase="SHOCK_HOLD"
                        else:
                            shock_ref[0]=-1
                            finalize_step(pending_states,pending_actions,pending_rewards,None,False)
                            pending_shock=None
                            phase="STEP_END"
                        phase_start=now

            elif phase=="SHOCK_HOLD":
                if now-phase_start>=int(ms*0.44):
                    bailed=(shock_ref[0]==-1)
                    if not bailed: shock_ref[0]=-1
                    finalize_step(pending_states,pending_actions,pending_rewards,pending_shock,bailed)
                    pending_shock=None; phase="STEP_END"; phase_start=now

            elif phase=="STEP_END":
                if now-phase_start>=int(ms*0.15):
                    liq_hist.append(sum(a.liquid for a in agents))
                    if len(liq_hist)>200: liq_hist.pop(0)
                    _acts = [a.last_action for a in agents]
                    _dom  = max(set(_acts), key=_acts.count)
                    _coop = sum(a.coop_score for a in agents)/NUM_AGENTS
                    _ndev = sum(1 for a in _acts if a!=_dom)/NUM_AGENTS
                    _totl = sum(a.liquid for a in agents)
                    _risk = 1.0 - min(_totl/(NUM_AGENTS*45), 1.0)
                    nash_hist.append({
                        "coop": _coop, "nash_dev": _ndev, "risk": _risk,
                        "regime": classify_regime(_coop, _ndev, _risk),
                    })
                    if len(nash_hist)>400: nash_hist.pop(0)
                    if step>=STEPS_PER_EP:
                        ep_history.append({"episode":episode,"total_liquid":sum(a.liquid for a in agents),
                                           "fed_rate":player.fed_rate,"health":player.health})
                        episode+=1; step=0
                        for a in agents:
                            a.qe_bonus=0.0
                            if not a.bankrupt:
                                a.liquid=a.eff_cap()*ACTION_PCT[a.last_action]
                                a.illiquid=a.eff_cap()-a.liquid
                            disp_liq[a.id]=a.liquid

                    active_ai = sum(1 for a in agents[1:] if not a.bankrupt)

                    # Win/Loss Evaluation
                    if player.game_over:
                        game_state="GAME_OVER"
                        lose_reason="The interbank network has failed under your watch."
                    elif not agents[0].bankrupt and active_ai == 0:
                        game_state="WIN"
                        win_reason="SURVIVAL: Outlasted all AI banks!"
                    elif episode>=NUM_EPISODES:
                        if not agents[0].bankrupt and all(agents[0].capital >= a.capital for a in agents[1:] if not a.bankrupt):
                            game_state="WIN"
                            win_reason="APEX CAPITALIST: Highest cumulative capital ($C_i$)!"
                        else:
                            game_state="GAME_OVER"
                            if agents[0].bankrupt:
                                lose_reason="BANKRUPT: Player $C_0$ <= 0."
                            else:
                                lose_reason="AI achieved higher capital by Turn 2000."
                    else:
                        phase="AGENT_TURN"; phase_start=now

        if shock_flash>0: shock_flash-=1
        for r in ripples:   r.update()
        for d in flow_dots: d.update()
        ripples   =[r for r in ripples   if not r.done]
        flow_dots =[d for d in flow_dots if not d.done]
        for i in range(NUM_AGENTS): disp_liq[i]+=(agents[i].liquid-disp_liq[i])*0.13

        bg=lc(BG,(55,12,12),(shock_flash-28)/24) if shock_flash>28 else BG
        screen.fill(bg)

        for i in range(NUM_AGENTS):
            for j in range(i+1,NUM_AGENTS):
                c=lc(NETC,(50,80,145),min((agents[i].liquid+agents[j].liquid)/120,1.0))
                pygame.draw.line(screen,c,(BK_CX[i],BK_TOP+BK_H//2),(BK_CX[j],BK_TOP+BK_H//2),1)

        for r in ripples:   r.draw(screen)
        for d in flow_dots: d.draw(screen)

        bail_window = (phase=="SHOCK_HOLD")
        for i,ag in enumerate(agents):
            draw_bank_col(screen,ag,i,disp_liq[i],
                          selected    =(player.selected==i),
                          active_turn =(active_agt==i),
                          shocked     =(shock_ref[0]==i),
                          bail_window = bail_window,
                          fonts=fonts)

        # Legend strip along the bottom
        lx = 210
        for clr,lbl in [(LIQ,"Liquid"),(ILLIQ,"Illiquid"),(GOLD,"Selected"),(SHOKC,"Shock"),(BAILC,"Bailout")]:
            pygame.draw.rect(screen, clr, (lx, H-16, 8, 8), border_radius=2)
            tx(screen, lbl, fonts["xs"], DIM, lx+14, H-12, anchor="midleft"); lx += 76

        nc_x, nc_y, nc_w, nc_h = 207, BK_TOP+BK_H+98, 720, 290
        draw_nash_chart(screen, nash_hist, fonts, nc_x, nc_y, nc_w, nc_h)

        draw_top_hud(screen,player,episode,step,speed_idx,fonts)
        draw_left_panel(screen,player,agents,tool_rects,hov_tool,fonts)
        draw_right_panel(screen,agents,player,episode,step,event_log,liq_hist,fonts)

        if paused:
            ov=pygame.Surface((W,H),pygame.SRCALPHA); ov.fill((0,0,0,110)); screen.blit(ov,(0,0))
            tx(screen,"[ PAUSED ]",fonts["xl"],WARN,W//2,H//2)

        if game_state=="GAME_OVER":
            rt,rc2=player.rating()
            draw_overlay(screen,
                "SYSTEM COLLAPSED", lose_reason,
                [(f"Survived",f"{episode} of {NUM_EPISODES} episodes",DIM),
                 ("Collapses",str(player.collapses),ILLIQ),
                 ("Interventions",str(player.interventions),BAILC),
                 ("Final Rating",rt,rc2)],
                "Press ENTER to try again  |  ESC to quit",
                ILLIQ,ILLIQ,fonts)
        elif game_state=="WIN":
            rt,rc2=player.rating()
            draw_overlay(screen,
                "MISSION COMPLETE", win_reason,
                [(f"Final Health",f"{player.health:.0f}/100",lc(ILLIQ,LIQ,player.health/100)),
                 ("Collapses",str(player.collapses),LIQ if player.collapses==0 else ILLIQ),
                 ("Interventions",str(player.interventions),BAILC),
                 ("Final Rating",rt,rc2)],
                "Press ENTER to play again  |  ESC to quit",
                GOOD,GOOD,fonts)

        if scale < 1.0:
            pygame.transform.smoothscale(screen, (win_w, win_h), window)
        pygame.display.flip()

    pygame.quit()
    if ep_history:
        pd.DataFrame(ep_history).to_csv("simulation_data.csv",index=False)
        print(f"Saved {len(ep_history)} episodes to simulation_data.csv")

if __name__=="__main__":
    run()
