import numpy as np
import matplotlib.pyplot as plt

from environment import PaperRaceEnv
from replay_buffer import ReplayBuffer

trk_col = np.array([99, 99, 99]) # pálya színe (szürke)

sections = np.array([[273, 125, 273, 64],  # [333, 125, 333, 64],[394, 157, 440, 102],
                     [370, 195, 440, 270]])

env = PaperRaceEnv('PALYA4.bmp', trk_col, 'GG1.bmp', sections, random_init=False) # paperrace környezet létrehozása
random_seed = 123

s_dim = 4 #állapottér dimenziója
a_dim = 1 #action space dimenzója

draw = True

ref_actions = np.array([0, 0, -60, -90, -130, -130, -110, -90])

env.reset()
if draw: # ha rajzolunk
    plt.clf()
    env.draw_track()
v = np.array(env.starting_spd)  # az elején a sebesség a startvonalra meroleges
pos = np.array(env.starting_pos)  # kezdőpozíció beállítása

reward = 0
epreward = 0
end = False

color = (0, 0, 1)

steps_nr = range(0, len(ref_actions))

ref_steps = np.zeros(len(ref_actions))
ref_dist = np.zeros(len(ref_actions))

for i in steps_nr:
    nye = input('Give input')
    action = env.ref_actions[i]
    # print("action: ", action, "=============================")
    gg_action = env.gg_action(action)  # action-höz tartozó vektor lekérése
    # print("gg:", gg_action, "v:", v, "posold:", pos, "------")
    pos_new_to_chk = env.step(gg_action, v, pos)
    # print("pos_chk:", pos_new_to_chk, "---------------------")
    pos_old, pos_new, reward, end, section_nr = env.step_check(pos, pos_new_to_chk, 'blue')
    # print("Aftstp posold:", pos_old, "posnew:", pos_new, "--")
    curr_dist_in, pos_in, curr_dist_out, pos_out = env.get_ref(pos_new)
    print("currdistin:---------------", curr_dist_in)
    print("reward:", reward)
    ref_dist[i] = curr_dist_in
    epreward = epreward + reward
    ref_steps[i] = -epreward

    pos = pos_old
    v_new = pos_new - pos

    if draw:
        plt.pause(0.001)
        plt.draw()


    v = v_new
    pos = pos_new




print(ref_dist)
print(ref_steps)