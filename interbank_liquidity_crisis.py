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
STRAT_COLORS  = [(220,60,90),(240,130,60),(220,200,55),(80,210,160),(34,211,140)]
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

BG     = (8,10,18)
PANEL  = (13,18,35)
PANEL2 = (19,26,50)
LIQ    = (34,211,140)
ILLIQ  = (220,60,90)
SHOKC  = (255,215,50)
BAILC  = (60,120,255)
TEXT   = (220,228,245)
DIM    = (75,98,128)
ACCENT = (120,100,255)
ACC2   = (0,200,180)
GOLD   = (255,200,55)
WARN   = (255,160,50)
GOOD   = (50,200,120)
NETC   = (22,38,72)
ACTIVE = (185,164,255)

TOOLS = [
    dict(key="b", label="BAILOUT",      desc="Restore target bank",  cost=50, earn=0,  sel=True,  clr=BAILC,            kcode=pygame.K_b),
    dict(key="q", label="QE INJECT",    desc="Add +25 capital",      cost=35, earn=0,  sel=True,  clr=ACC2,             kcode=pygame.K_q),
    dict(key="e", label="MANDATE LIQ",  desc="Force LIQUID x8 steps",cost=20, earn=0,  sel=True,  clr=(100,180,255),    kcode=pygame.K_e),
    dict(key="t", label="TAX HOARDER",  desc="Penalise HOARD bank",  cost=0,  earn=15, sel=True,  clr=WARN,             kcode=pygame.K_t),
    dict(key="c", label="RATE CUT",     desc="Boost liq rates x15",  cost=60, earn=0,  sel=False, clr=GOOD,             kcode=pygame.K_c),
    dict(key="x", label="STRESS TEST",  desc="Manual shock on bank", cost=0,  earn=0,  sel=True,  clr=SHOKC,            kcode=pygame.K_x),
    dict(key="r", label="RATE TOGGLE",  desc="Flip Fed rate",        cost=0,  earn=0,  sel=False, clr=(180,140,255),    kcode=pygame.K_r),
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

    def tier(self, pct):
        return 0 if pct < 0.25 else (1 if pct < 0.75 else 2)

    def eff_cap(self):
        return self.capital + self.qe_bonus

    def get_state(self, agents, fed):
        own = self.liquid / max(self.eff_cap(), 1)
        tot = max(sum(a.eff_cap() for a in agents), 1)
        avg = sum(a.liquid for a in agents) / tot
        return (self.tier(own), self.tier(avg), fed)

    def act(self, s):
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

def rr(surf, color, rect, r=8, w=0):
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
    rr(surf, PANEL, (0,0,W,TOP_H), 0)
    pygame.draw.line(surf, (28,38,70), (0,TOP_H-1), (W,TOP_H-1), 1)

    rr(surf, PANEL2, (LP_X+4, 8, LP_W-8, TOP_H-16), 8)
    tx(surf, "CENTRAL BANK CMD", fonts["sm"], ACCENT, LP_X+LP_W//2, 26)
    tx(surf, "You are the Federal Reserve", fonts["xs"], DIM, LP_X+LP_W//2, 46)
    tx(surf, "Stabilize the network", fonts["xs"], DIM, LP_X+LP_W//2, 60)

    hx = LP_X+LP_W+20
    rr(surf, PANEL2, (hx, 8, 220, TOP_H-16), 8)
    tx(surf, "SYSTEMIC HEALTH", fonts["xs"], DIM, hx+110, 20)
    rr(surf, (22,32,58), (hx+8, 28, 204, 14), 7)
    hc = lc(ILLIQ, GOOD, player.health/100)
    rr(surf, hc, (hx+8, 28, int(204*player.health/100), 14), 7)
    tx(surf, f"{player.health:.0f} / 100", fonts["xs"], hc, hx+110, 50)

    pulse = math.sin(pygame.time.get_ticks()*0.003)*8
    if player.health < 30:
        rr(surf, (*ILLIQ, 60), (hx, 8, 220, TOP_H-16), 8)

    rating, rc = player.rating()
    rr(surf, PANEL2, (hx+230, 8, 80, TOP_H-16), 8)
    tx(surf, "RATING", fonts["xs"], DIM,  hx+270, 20)
    tx(surf, rating,   fonts["md"], rc,   hx+270, 51)

    bx = hx+320
    rr(surf, PANEL2, (bx, 8, 190, TOP_H-16), 8)
    tx(surf, "POLICY BUDGET", fonts["xs"], DIM, bx+95, 20)
    rr(surf, (22,32,58), (bx+8, 30, 174, 10), 5)
    bc = lc(ILLIQ, ACC2, player.budget/player.MAX_BUDGET)
    rr(surf, bc, (bx+8, 30, int(174*player.budget/player.MAX_BUDGET), 10), 5)
    tx(surf, f"${player.budget} / ${player.MAX_BUDGET}", fonts["xs"], bc, bx+95, 52)

    sx2 = bx+200
    rr(surf, PANEL2, (sx2, 8, 170, TOP_H-16), 8)
    fed_c = (251,113,133) if player.fed_rate else (52,200,150)
    tx(surf, "FED: "+("HIGH(3%)" if player.fed_rate else "LOW(1%)"),  fonts["xs"], fed_c,   sx2+85, 20)
    rc2 = WARN if player.rate_cut_turns>0 else DIM
    tx(surf, f"Rate Cut: {'ON '+str(player.rate_cut_turns)+'t' if player.rate_cut_turns else 'off'}", fonts["xs"], rc2, sx2+85, 37)
    tx(surf, f"Speed: {SPEED_NAMES[speed_idx]}", fonts["xs"], ACCENT, sx2+85, 54)

    ex = sx2+180
    rr(surf, PANEL2, (ex, 8, 180, TOP_H-16), 8)
    tx(surf, f"Episode {ep:04d} / {NUM_EPISODES}", fonts["xs"], TEXT,  ex+90, 20)
    tx(surf, f"Step    {step:02d} / {STEPS_PER_EP}",  fonts["xs"], DIM,   ex+90, 37)
    tx(surf, f"Collapses: {player.collapses}",         fonts["xs"], ILLIQ if player.collapses else DIM, ex+90, 54)

    ix = ex+190
    if ix+170 < W-RP_W-10:
        rr(surf, PANEL2, (ix, 8, 155, TOP_H-16), 8)
        tx(surf, f"Interventions: {player.interventions}", fonts["xs"], BAILC, ix+77, 38)
        tx(surf, "[1-5] Select Bank", fonts["xs"], DIM, ix+77, 56)


def draw_left_panel(surf, player, agents, tool_rects, hovered_tool, fonts):
    rr(surf, PANEL, (LP_X, TOP_H, LP_W, H-TOP_H-4), 10)
    x, y, w = LP_X+6, TOP_H+10, LP_W-12

    tx(surf, "POLICY TOOLS", fonts["sm"], ACCENT, x+w//2, y+10)
    pygame.draw.line(surf, (30,42,74), (x,y+22), (x+w,y+22), 1)
    y += 30

    rr(surf, PANEL2, (x, y, w, 88), 6)
    if player.selected >= 0:
        ag = agents[player.selected]
        sc = STRAT_COLORS[ag.last_action]
        tx(surf, BANK_NAMES[player.selected], fonts["sm"], GOLD, x+w//2, y+12)
        rr(surf, sc, (x+5, y+26, w-10, 17), 8)
        tx(surf, ACTION_LABELS[ag.last_action], fonts["xs"], BG, x+w//2, y+34)
        tx(surf, f"L:{ag.liquid:.0f}  I:{ag.illiquid:.0f}", fonts["xs"], TEXT, x+w//2, y+50)
        cw = w-16
        coop = ag.coop_score
        rr(surf, (20,32,58), (x+8, y+64, cw, 7), 3)
        rr(surf, lc(ILLIQ,LIQ,coop), (x+8, y+64, int(cw*coop), 7), 3)
        if ag.mandate_turns > 0:
            tx(surf, f"MANDATED {ag.mandate_turns}t", fonts["xs"], (100,180,255), x+w//2, y+78)
        else:
            tx(surf, f"Coop {coop*100:.0f}%", fonts["xs"], lc(ILLIQ,LIQ,coop), x+w//2, y+78)
    else:
        tx(surf, "NO TARGET", fonts["xs"], DIM, x+w//2, y+18)
        tx(surf, "Click a bank or", fonts["xs"], DIM, x+w//2, y+36)
        tx(surf, "press [1-5]", fonts["xs"], DIM, x+w//2, y+52)
        tx(surf, "to select", fonts["xs"], DIM, x+w//2, y+68)
    y += 96

    tool_rects.clear()
    for ti, td in enumerate(TOOLS):
        can = not td["sel"] or player.selected >= 0
        aff = player.budget >= td["cost"]
        clr = td["clr"] if (can and aff) else DIM
        bg  = PANEL2 if can else (15,22,40)
        hov = (hovered_tool == ti)

        rect = (x, y, w, 36)
        tool_rects.append((rect, td["key"]))
        rr(surf, bg, rect, 6)
        if can and aff:
            rr(surf, (*clr, 40 if hov else 15), rect, 6)
            pygame.draw.rect(surf, clr, (x,y,w,36), 1, border_radius=6)

        kw = 20
        rr(surf, clr if (can and aff) else (28,36,56), (x+3, y+8, kw, 20), 4)
        tx(surf, td["key"].upper(), fonts["xs"], BG if (can and aff) else (50,65,90), x+3+kw//2, y+18)

        tx(surf, td["label"], fonts["sm" if hov else "xs"], TEXT if (can and aff) else (50,65,90), x+28, y+10, anchor="midleft")
        tx(surf, td["desc"],  fonts["xs"], lc(DIM,TEXT,0.5) if (can and aff) else (40,55,70), x+28, y+26, anchor="midleft")

        if td["earn"]:
            cs, cc = f"+${td['earn']}", GOOD
        elif td["cost"]:
            cs, cc = f"-${td['cost']}", lc(ILLIQ,(220,160,80),0.5) if (can and aff) else DIM
        else:
            cs, cc = "FREE", ACCENT if (can and aff) else DIM
        tx(surf, cs, fonts["xs"], cc, x+w-4, y+18, anchor="midright")
        y += 40

    y += 4
    pygame.draw.line(surf, (30,42,74), (x,y), (x+w,y), 1)
    y += 8

    tx(surf, "CONTROLS", fonts["xs"], DIM, x+w//2, y+9)
    y += 20
    for line in ["[↑/↓]  Speed", "[SPACE]  Pause/Resume", "[1-5]  Select Bank", "[ESC]   Quit"]:
        tx(surf, line, fonts["xs"], (90,110,140), x+w//2, y)
        y += 16


def draw_bank_col(surf, agent, idx, disp_liq, selected, active_turn, shocked, bail_window, fonts):
    cx = BK_CX[idx]
    bx = cx - BK_W//2
    by = BK_TOP

    rr(surf, PANEL, (bx-10, by-10, BK_W+20, BK_H+20), 10)

    pct  = max(0.0, min(1.0, disp_liq / max(agent.eff_cap(), 1)))
    lh   = int(BK_H * pct)
    ih   = BK_H - lh
    if ih > 0: rr(surf, ILLIQ, (bx, by, BK_W, ih),    6)
    if lh > 0: rr(surf, LIQ,   (bx, by+ih, BK_W, lh), 6)

    if agent.qe_bonus > 0:
        rr(surf, (*ACC2, 55), (bx, by, BK_W, BK_H), 6)

    if agent.mandate_turns > 0:
        pygame.draw.rect(surf, (100,180,255), (bx-10,by-10,BK_W+20,BK_H+20), 2, border_radius=10)

    if selected:
        t = (pygame.time.get_ticks() % 900) / 900
        a = int(140+110*math.sin(t*2*math.pi))
        rr(surf, (*GOLD, a), (bx-10,by-10,BK_W+20,BK_H+20), 10, w=3)

    if active_turn and not selected:
        rr(surf, (*(ACTIVE),35), (bx-10,by-10,BK_W+20,BK_H+20), 10)
        pygame.draw.rect(surf, ACTIVE, (bx-10,by-10,BK_W+20,BK_H+20), 2, border_radius=10)

    if shocked:
        pygame.draw.rect(surf, SHOKC, (bx-10,by-10,BK_W+20,BK_H+20), 3, border_radius=10)
        if bail_window:
            t2 = (pygame.time.get_ticks()%600)/600
            a2 = int(150+100*math.sin(t2*math.pi*2))
            rr(surf, (*BAILC, a2), (bx-10,by-30,BK_W+20,20), 6)
            tx(surf, "BAIL [B]!", fonts["xs"], TEXT, cx, by-20)
        else:
            tx(surf, "SHOCK!", fonts["xs"], SHOKC, cx, by-20)

    sc = STRAT_COLORS[agent.last_action]
    rr(surf, sc, (bx+5, by+8, BK_W-10, 17), 8)
    tx(surf, ACTION_LABELS[agent.last_action], fonts["xs"], BG, cx, by+16)

    tier_lbls   = ["STRESSED","MODERATE","STABLE"]
    tier_clrs   = [ILLIQ, WARN, LIQ]
    own_t       = agent.tier(agent.liquid / max(agent.eff_cap(),1))
    lbl_y       = by - 20 if not shocked else by - 38
    if not shocked or not bail_window:
        tx(surf, tier_lbls[own_t], fonts["xs"], tier_clrs[own_t], cx, lbl_y)

    tx(surf, BANK_NAMES[idx], fonts["sm"], GOLD if selected else TEXT, cx, by+BK_H+14)
    tx(surf, f"L:{agent.liquid:.0f}  I:{agent.illiquid:.0f}", fonts["xs"], DIM, cx, by+BK_H+30)

    bw = BK_W - 12
    c2 = agent.coop_score
    rr(surf, PANEL2, (bx+6, by+BK_H+43, bw, 7),          3)
    rr(surf, lc(ILLIQ,LIQ,c2), (bx+6, by+BK_H+43, int(bw*c2), 7), 3)
    tx(surf, f"COOP {c2*100:.0f}%", fonts["xs"], lc(ILLIQ,LIQ,c2), cx, by+BK_H+59)

    if agent.qe_bonus > 0:
        tx(surf, f"QE+{agent.qe_bonus:.0f}", fonts["xs"], ACC2, cx, by+BK_H+73)
    if agent.mandate_turns > 0:
        tx(surf, f"MAND {agent.mandate_turns}t", fonts["xs"], (100,180,255), cx, by+BK_H+73)


def draw_right_panel(surf, agents, player, ep, step, event_log, liq_hist, fonts):
    x, w = RP_X, RP_W
    rr(surf, PANEL, (x, TOP_H, w, H-TOP_H-4), 10)
    y = TOP_H + 10

    tx(surf, "GAME THEORY ENGINE", fonts["md"], ACCENT, x+w//2, y+12)
    pygame.draw.line(surf, (28,38,68), (x+8,y+26), (x+w-8,y+26), 1)
    y += 32

    if len(player.health_log) > 2:
        sh, bw = 36, w-20
        rr(surf, PANEL2, (x+8, y, bw, sh), 6)
        recent = player.health_log[-60:]
        if len(recent) > 1:
            mx2,mn2 = max(recent),min(recent)
            sp2 = max(mx2-mn2,1)
            pts = [(x+10+int(i/max(len(recent)-1,1)*(bw-4)),
                    y+sh-4-int((v-mn2)/sp2*(sh-8)))
                   for i,v in enumerate(recent)]
            if len(pts)>1: pygame.draw.lines(surf, lc(ILLIQ,GOOD,player.health/100), False, pts, 2)
        tx(surf, "Health Trend", fonts["xs"], DIM, x+w//2, y+sh+8)
        y += sh+16

    tx(surf, "STRATEGY MATRIX", fonts["sm"], TEXT, x+w//2, y+8)
    pygame.draw.line(surf, (28,38,68), (x+8,y+20), (x+w-8,y+20), 1)
    y += 26

    for i, ag in enumerate(agents):
        ay = y + i*37
        rr(surf, PANEL2, (x+8, ay, w-16, 29), 5)
        sc = STRAT_COLORS[ag.last_action]
        rr(surf, sc, (x+12, ay+5, 50, 19), 9)
        tx(surf, ACTION_LABELS[ag.last_action][:3], fonts["xs"], BG, x+12+25, ay+14)
        nm_c = GOLD if player.selected==i else TEXT
        tx(surf, BANK_NAMES[i][:6], fonts["xs"], nm_c, x+68, ay+14, anchor="midleft")
        bw2 = w-160-16
        coop = ag.coop_score
        rr(surf, (20,32,58), (x+112, ay+9, bw2,12), 5)
        rr(surf, lc(ILLIQ,LIQ,coop), (x+112, ay+9, int(bw2*coop),12), 5)
        rw = ag.avg_reward
        tx(surf, f"{rw:+.1f}", fonts["xs"], LIQ if rw>=0 else ILLIQ, x+w-8, ay+14, anchor="midright")
        if ag.mandate_turns > 0:
            pygame.draw.rect(surf,(100,180,255),(x+8,ay,w-16,29),1,border_radius=5)
    y += NUM_AGENTS*37 + 6

    tx(surf, "SYSTEM METRICS", fonts["sm"], TEXT, x+w//2, y+8)
    pygame.draw.line(surf, (28,38,68), (x+8,y+20), (x+w-8,y+20), 1)
    y += 28

    acts   = [a.last_action for a in agents]
    dom    = max(set(acts), key=acts.count)
    coop_s = sum(a.coop_score for a in agents)/NUM_AGENTS
    nash_d = sum(1 for a in acts if a!=dom)/NUM_AGENTS
    tot_l  = sum(a.liquid for a in agents)
    risk   = 1.0 - min(tot_l/(NUM_AGENTS*45), 1.0)
    pd_t   = (1-coop_s)*risk

    mets = [
        ("Dominant Strat",  ACTION_LABELS[dom],          STRAT_COLORS[dom]),
        ("Sys Cooperation", f"{coop_s*100:.0f}%",        lc(ILLIQ,LIQ,coop_s)),
        ("Nash Deviation",  f"{nash_d*100:.0f}%",        lc(LIQ,ILLIQ,nash_d)),
        ("Contagion Risk",  f"{risk*100:.0f}%",          lc(LIQ,ILLIQ,risk)),
        ("PD Tension",      f"{pd_t*100:.0f}%",          lc(GOOD,ILLIQ,pd_t)),
    ]
    for label, val, vc in mets:
        rr(surf, PANEL2, (x+8, y, w-16, 24), 4)
        tx(surf, label, fonts["xs"], DIM,  x+14, y+12, anchor="midleft")
        tx(surf, val,   fonts["xs"], vc,   x+w-12, y+12, anchor="midright")
        y += 28

    if coop_s < 0.18:
        rr(surf,(50,28,8),(x+8,y,w-16,22),5)
        pygame.draw.rect(surf,WARN,(x+8,y,w-16,22),1,border_radius=5)
        tx(surf,"TRAGEDY OF THE COMMONS",fonts["xs"],WARN,x+w//2,y+11)
        y += 26
    elif risk > 0.72:
        rr(surf,(50,12,12),(x+8,y,w-16,22),5)
        pygame.draw.rect(surf,ILLIQ,(x+8,y,w-16,22),1,border_radius=5)
        tx(surf,"SYSTEMIC CRISIS IMMINENT",fonts["xs"],ILLIQ,x+w//2,y+11)
        y += 26
    elif nash_d < 0.1 and coop_s > 0.55:
        rr(surf,(12,42,22),(x+8,y,w-16,22),5)
        pygame.draw.rect(surf,GOOD,(x+8,y,w-16,22),1,border_radius=5)
        tx(surf,"NASH EQ: COOPERATING",fonts["xs"],GOOD,x+w//2,y+11)
        y += 26

    y += 4
    tx(surf,"EVENT LOG",fonts["sm"],TEXT,x+w//2,y+8)
    pygame.draw.line(surf,(28,38,68),(x+8,y+20),(x+w-8,y+20),1)
    y += 26

    for ev in event_log[-5:]:
        rr(surf,PANEL2,(x+8,y,w-16,20),3)
        tx(surf,ev["msg"][:34],fonts["xs"],ev.get("color",DIM),x+14,y+10,anchor="midleft")
        y += 23

    remaining = (H-8) - y - 24
    if len(liq_hist) > 3 and remaining > 52:
        sh2 = min(remaining-18, 58)
        rr(surf,PANEL2,(x+8,y+6,w-16,sh2),6)
        recent2 = liq_hist[-60:]
        if len(recent2)>1:
            mx3,mn3 = max(recent2),min(recent2)
            sp3 = max(mx3-mn3,1)
            bw3 = w-24
            pts2 = [(x+12+int(i/max(len(recent2)-1,1)*(bw3-4)),
                     y+6+sh2-4-int((v-mn3)/sp3*(sh2-8)))
                    for i,v in enumerate(recent2)]
            if len(pts2)>1: pygame.draw.lines(surf,ACC2,False,pts2,1)
        tx(surf,"Liq Trend",fonts["xs"],DIM,x+w//2,y+sh2+14)


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
        ag.capital=max(10,ag.capital-20)
        ag.apply_action(ag.last_action)
        player.earn(15)
        log(f"TAXED HOARDER: {BANK_NAMES[sel]} -20 cap", WARN)

    elif key=="x":
        amt=random.randint(40,90)
        ag.liquid=max(0,ag.liquid-amt)
        ripples+=[Ripple(cx,cy,SHOKC),Ripple(cx,cy,(255,180,50),speed=2,max_r=80)]
        log(f"STRESS TEST: {BANK_NAMES[sel]} -{amt}", SHOKC)


def run():
    pygame.init()
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Interbank Liquidity Crisis — Central Bank Command")
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

    pending_shock     = None
    pending_states    = None
    pending_actions   = None
    pending_rewards   = None

    def log(msg,clr=DIM): 
        event_log.append({"msg":msg,"color":clr})
        if len(event_log)>40: event_log.pop(0)

    def prepare_step():
        fed = player.fed_rate
        states  = [a.get_state(agents,fed) for a in agents]
        actions = [a.act(s) for a,s in zip(agents,states)]
        for a,ac in zip(agents,actions): a.apply_action(ac)
        elr = BASE_LIQ[fed] + (0.02 if player.rate_cut_turns>0 else 0)
        eir = BASE_ILLIQ[fed]
        rewards = [a.liquid*elr + a.illiquid*eir for a in agents]
        shock_info = None
        if random.random() < SHOCK_PROB:
            si=random.randint(0,NUM_AGENTS-1)
            sa=random.randint(40,90)
            shock_info={"idx":si,"amt":sa}
        return states,actions,rewards,shock_info

    def finalize_step(states,actions,rewards,shock_info,bailed):
        fed = player.fed_rate
        if shock_info and not bailed:
            si,sa = shock_info["idx"],shock_info["amt"]
            tgt   = agents[si]
            if tgt.liquid < sa:
                shortfall = sa - tgt.liquid
                sys_liq   = sum(a.liquid for a in agents) - tgt.liquid
                if sys_liq < shortfall:
                    for j in range(NUM_AGENTS): rewards[j]-=100
                    player.damage(15)
                    player.collapses+=1
                    log("SYSTEMIC COLLAPSE! -15 health",ILLIQ)
                else:
                    rewards[si]-=20
                    td2=max(sys_liq,1)
                    for j,a in enumerate(agents):
                        if j!=si: a.liquid=max(0,a.liquid-a.liquid/td2*shortfall)
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

        ns=[a.get_state(agents,fed) for a in agents]
        for i,a in enumerate(agents): a.learn(states[i],actions[i],rewards[i],ns[i])

    running=True
    while running:
        clock.tick(FPS)
        now=pygame.time.get_ticks()
        ms =STEP_MS[speed_idx]
        mx2,my2=pygame.mouse.get_pos()

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
                        liq_hist.clear(); ripples.clear(); flow_dots.clear()
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
                    if step>=STEPS_PER_EP:
                        ep_history.append({"episode":episode,"total_liquid":sum(a.liquid for a in agents),
                                           "fed_rate":player.fed_rate,"health":player.health})
                        episode+=1; step=0
                        for a in agents:
                            a.capital=INIT_CAPITAL; a.qe_bonus=0.0
                            a.liquid=a.capital*ACTION_PCT[a.last_action]
                            a.illiquid=a.capital-a.liquid
                            disp_liq[a.id]=a.liquid
                    if player.game_over:       game_state="GAME_OVER"
                    elif episode>=NUM_EPISODES: game_state="WIN"
                    else: phase="AGENT_TURN"; phase_start=now

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

        if len(liq_hist)>3:
            sx,sy,sw,sh2=207,BK_TOP+BK_H+82,720,40
            rr(screen,PANEL,(sx-6,sy-14,sw+12,sh2+20),8)
            tx(screen,"SYSTEM LIQUIDITY TREND",fonts["xs"],DIM,sx+sw//2,sy-6)
            recent3=liq_hist[-80:]
            if len(recent3)>1:
                m3,n3=max(recent3),min(recent3); s3=max(m3-n3,1)
                pts3=[(sx+int(i/max(len(recent3)-1,1)*sw),sy+sh2-int((v-n3)/s3*sh2)) for i,v in enumerate(recent3)]
                pygame.draw.lines(screen,ACC2,False,pts3,1)

        lx=210
        for clr,lbl in [(LIQ,"Liquid"),(ILLIQ,"Illiquid"),(GOLD,"Selected"),(SHOKC,"Shock"),(BAILC,"Bailout")]:
            pygame.draw.rect(screen,clr,(lx,H-18,10,10),border_radius=2)
            tx(screen,lbl,fonts["xs"],DIM,lx+12,H-13,anchor="midleft"); lx+=80

        draw_top_hud(screen,player,episode,step,speed_idx,fonts)
        draw_left_panel(screen,player,agents,tool_rects,hov_tool,fonts)
        draw_right_panel(screen,agents,player,episode,step,event_log,liq_hist,fonts)

        if paused:
            ov=pygame.Surface((W,H),pygame.SRCALPHA); ov.fill((0,0,0,110)); screen.blit(ov,(0,0))
            tx(screen,"[ PAUSED ]",fonts["xl"],WARN,W//2,H//2)

        if game_state=="GAME_OVER":
            rt,rc2=player.rating()
            draw_overlay(screen,
                "SYSTEM COLLAPSED","The interbank network has failed under your watch.",
                [(f"Survived",f"{episode} of {NUM_EPISODES} episodes",DIM),
                 ("Collapses",str(player.collapses),ILLIQ),
                 ("Interventions",str(player.interventions),BAILC),
                 ("Final Rating",rt,rc2)],
                "Press ENTER to try again  |  ESC to quit",
                ILLIQ,ILLIQ,fonts)
        elif game_state=="WIN":
            rt,rc2=player.rating()
            draw_overlay(screen,
                "MISSION COMPLETE","You stabilised the global banking system!",
                [(f"Final Health",f"{player.health:.0f}/100",lc(ILLIQ,LIQ,player.health/100)),
                 ("Collapses",str(player.collapses),LIQ if player.collapses==0 else ILLIQ),
                 ("Interventions",str(player.interventions),BAILC),
                 ("Final Rating",rt,rc2)],
                "Press ENTER to play again  |  ESC to quit",
                GOOD,GOOD,fonts)

        pygame.display.flip()

    pygame.quit()
    if ep_history:
        pd.DataFrame(ep_history).to_csv("simulation_data.csv",index=False)
        print(f"Saved {len(ep_history)} episodes to simulation_data.csv")

if __name__=="__main__":
    run()
