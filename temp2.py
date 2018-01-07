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
ref_dist = np.array([37, 71, 108, 164, 207, 232, 250, 271]) # most csak igy fixen
ref_steps = np.array([1, 2, 3, 4, 5, 6, 7, 7.33053])
print("ref dist:", ref_dist)
print("ref steps:", ref_steps)


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

steps = range(0, len(ref_actions))
ref_pos = np.zeros(len(ref_actions))
# ref_dist = np.zeros(len(ref_actions))
i = 0

while not end:
    i += 1
    action = int(input('Give inut (-180..180 number)'))
    #ref_action = env.ref_actions[i]
    # print("action: ", action, "=============================")
    gg_action = env.gg_action(action)  # action-höz tartozó vektor lekérése
    # print("gg:", gg_action, "v:", v, "posold:", pos, "------")
    pos_new_to_chk = env.step(gg_action, v, pos)
    # print("pos_chk:", pos_new_to_chk, "---------------------")
    pos_old, pos_new, reward, end, section_nr = env.step_check(pos, pos_new_to_chk, 'blue')
    # print("Aftstp posold:", pos_old, "posnew:", pos_new, "--")

    curr_dist_in, curr_pos_in, curr_dist_out, curr_pos_out = env.get_ref(pos_new)
    pre_dist_in, pre_pos_in, pre_dist_out, pre_pos_out = env.get_ref(pos_old)

    # amennyi ido (lepes) alatt a ref_actionok, a pre_dist-ből a curr_dist-be eljutottak--------------------------
    # look-up szerűen lesz. Először a bemenetek:
    x = ref_dist
    y = ref_steps

    xvals = np.array([pre_dist_in, curr_dist_in])
    print("elozo es aktualis tav:", xvals)

    # ezekre a tavolsagokra a referencia lepessor ennyi ido alaptt jutott el
    yinterp = np.interp(xvals, x, y, 0)
    print("ref ennyi ido alatt jutott ezekre:", yinterp)

    # tehat ezt a negyasau lepest a referencia ennyi ido alatt tette meg (- legyen, hogy a kisebb ido legyen a magasabb
    # reward) :
    ref_delta = -yinterp[1] + yinterp[0]
    print("a ref. ezen lepesnyi ideje:", ref_delta)

    # az atualis lepesben az eltelt ido nyilvan -1, illetve ha ido-bunti van akkor ennel tobb, eppen a reward
    print("elozo es aktualis lepes kozott eltelt ido:", reward)

    # amenyivel az aktualis ebben a lepesben jobb, azaz kevesebb ido alatt tette meg ezt a elmozdulat, mint a ref
    # lepessor, az:
    rew_dt = -ref_delta + reward
    print("az aktualis, ebben a lepesben megtett tavot ennyivel kevesebb ido alatt tette meg mint a ref. (ha (-) akkor meg több):", rew_dt)

    epreward = epreward + rew_dt

    pos = pos_old
    v_new = pos_new - pos

    if draw:
        plt.pause(0.001)
        plt.draw()

    v = v_new
    pos = pos_new

    print("ref epizod reward:",- ref_steps[-1])
    print("aktual epreward:", epreward)



# Megnezzuk hogy a pos_old es a pos_new milyen dist_old es dist_new-hez tartozik (in vagy out, vagy atlag...)

# Ehez a dist_old es dist new-hoz megnezzuk hogy a referencia lepessor mennyi ido alatt jutott el ezek lesznek
# step_old es step_new.

# A step_old es step_new kulonbsege azt adja hogy azt a tavot, szakaszt, amit a jelenlegi pos_old, pos_new
# megad, azt a ref lepessor, mennyi ido tette meg. A jelenlegi az 1 ido, hiszen egy lepes. A ketto kulonbsege
# adja majd pillanatnyi rewardot.
