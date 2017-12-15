import numpy as np
import matplotlib.pyplot as plt

from environment import PaperRaceEnv
from replay_buffer import ReplayBuffer

trk_col = np.array([99, 99, 99]) # pálya színe (szürke)

sections = np.array([[350,  60, 350, 100],
                     [560, 130, 535, 165],
                     [348, 354, 348, 326]])
#                     [ 35, 200,  70, 200],
#                     [250,  60, 250, 100]])

# start_line = np.array([32, 393, 32, 425]) # sigmoid alakú pálya

env = PaperRaceEnv('PALYA3.bmp', trk_col, 'GG1.bmp', sections, random_init=False) # paperrace környezet létrehozása

mem_size = 100 # a memória mérete, amiből a batch-be válogatunk
batch_size = 10 # batch mérete, ami a tanítási adatokat tartalmazza
episodes = 1000 # hányszor fusson a tanítás
random_seed = 123

s_dim = 4 #állapottér dimenziója
a_dim = 1 #action space dimenzója

replay_buffer = ReplayBuffer(int(mem_size), int(random_seed))

draw = True

for ep in range(episodes):
    env.reset()
    print("----EP.: ", ep) # epizód számának kiírása
    if draw: # ha rajzolunk
        plt.clf()
        env.draw_track()
    v = np.array([20, 0])  # az elején a sebesség jobbra 1
    # ezt könnyen megváltoztatja, tulajdonképen csak arra jó, hogy nem 0
    pos = np.array(env.starting_pos)  # kezdőpozíció beállítása
    reward = 0
    epreward = 0
    end = False
    color = (1 , 0, 0)

    while not end:
        action = int(input('Give inut (-180..180 number)'))
        #action = int(np.random.randint(-180, 180, size=1))
        print("action: ",action)
        gg_action = env.gg_action(action)  # action-höz tartozó vektor lekérése
        v_new, pos_new, reward, end, section_nr = env.step(gg_action, v, pos, draw, color)
        s = [v[0], v[1], pos[0], pos[1]]
        s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]
        a = action
        terminal = end
        r = reward

        #print(s)
        #print(s2)
        epreward = epreward + reward
        print("reward: ",reward)
        print("Section: ", section_nr)

        replay_buffer.add(np.reshape(s, (s_dim,)), np.reshape(a, (a_dim,)), r,
                          terminal, np.reshape(s2, (s_dim,)))

        if draw:
            plt.pause(0.001)
            plt.draw()

        v = v_new
        pos = pos_new

    if replay_buffer.size() > int(batch_size):
        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(batch_size))
        #TODO: valami olvashatóbb formátumban kiiratni. pl. táblázat
        #print('batch: ', s_batch, a_batch, r_batch, t_batch, s2_batch)

    print("Eprew.: ",epreward)


