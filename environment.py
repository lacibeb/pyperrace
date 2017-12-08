import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import linalg as LA
from random import randint
from skimage.morphology import disk
from skimage.color import rgb2gray

#TODO: megcsinálnia rewardot a szakaszok szerint. Megkeresni a 'celbaer' fuggvenyt az eredeti matlab kódomban
class PaperRaceEnv:
    """ez az osztály biztosítja a tanuláshoz a környezetet"""

    def __init__(self, trk_pic, trk_col, gg_pic, start_line, random_init=False, track_inside_color=None):

        # ha nincs megadva a pálya belsejének szín, akkor pirosra állítja
        # ez a rewardokat kiszámoló algoritmus működéséhez szükséges
        if track_inside_color is None:
            self.track_inside_color = np.array([255, 0, 0], dtype='uint8')
        else:
            self.track_inside_color = np.array(track_inside_color, dtype='uint8')

        self.trk_pic = mpimg.imread(trk_pic)  # beolvassa a pályát
        self.trk_col = trk_col  # trk_pic-en a pálya színe
        self.gg_pic = mpimg.imread(gg_pic) # beolvassa a GG diagramot
        self.steps = 0  # az eddig megtett lépések száma

        self.start_line = start_line  # az első szkasz közepén áll először az autó
        start_x = int(np.floor((start_line[0] + start_line[2]) / 2))
        start_y = int(np.floor((start_line[1] + start_line[3]) / 2))

        self.starting_pos = np.array([start_x, start_y]) # az autó kezdő pozíciója
        self.random_init = random_init # True, ha be van kapcsolva az autó véletlen pozícióból való indítása
        self.gg_actions = None # az action-ökhöz tartozó vektor értékeit cash-eli a legelajén és ebben tárolja
        self.prev_dist = 0

        self.track_indices = [] # a pálya (szürke) pixeleinek pozícióját tartalmazza
        for x in range(self.trk_pic.shape[1]):
            for y in range(self.trk_pic.shape[0]):
                if np.array_equal(self.trk_pic[y, x, :], self.trk_col):
                    self.track_indices.append([x, y])

        self.dists = self.__get_dists(True) # a kezdőponttól való "távolságot" tárolja a reward fv-hez

    def draw_track(self):
        plt.imshow(self.trk_pic)  # pálya kirajzolása

        X = np.array([self.start_line[0], self.start_line[2]])  # kezdővonal kirajzolása
        Y = np.array([self.start_line[1], self.start_line[3]])
        plt.plot(X, Y, color='red')

    def step(self, spd_chn, spd_old, pos_old, draw, color):

        """
        ez a függvény számolja a lépést

        :param spd_chn:  a sebesség megváltoztatása.(De ez relatívban van!!!)
        :param spd_old: az aktuális sebességvektor
        :param pos_old: aktuélis pozíció
        :return:
            spd_new: Az új(a lépés utáni) sebességvektor[Vx, Vy]
            pos_new: Az új(a lépés utáni) pozíció[Px, Py]
            reward: A kapott jutalom
            end: logikai érték, igaz ha vége van valamiért az epizódnak

        """

        end = False
        reward = 0

        # az aktuális sebesség irányvektora:
        e1_spd_old = spd_old / np.linalg.norm(spd_old)
        e2_spd_old = np.array([-1 * e1_spd_old[1], e1_spd_old[0]])

        spd_chn = np.asmatrix(spd_chn)

        # a valtozás globálisban:
        spd_chn_glb = np.round(np.column_stack((e1_spd_old, e2_spd_old)) * spd_chn.transpose())

        # az új sebességvektor:
        spd_new = spd_old + np.ravel(spd_chn_glb)
        pos_new = pos_old + spd_new


        if draw: # kirajzolja az autót
            X = np.array([pos_old[0], pos_new[0]])
            Y = np.array([pos_old[1], pos_new[1]])
            plt.plot(X, Y, color=color)

        if not self.is_on_track(pos_new): # ha kisiklik
            reward = -50
            reward, curr_dist, pos = self.get_reward(pos_new)

            #TODO 1: kisikláskor ne legyen vége

            """
            Az init-ben meghivasra kerul ez a get_dist fuggveny, ami elvileg csinál egy dists nevu dict-et, amiben
            összetartozó pálya belső ív koordináták és hozzájuk tartozó távolság értékek vannak. Namost a gondolat a 
            következő: 
            A dict-ből le lehet kérdezni a kieséskor évényez pozícióhoz rendelhető, belső ív érvényes pontját. Ezt adja
            a get_reward 'pos' néven. 
            Adunk egy kurva nagy büntit, és az kiesés után azt mondjuk hogy induljon tovább ebből a pontból az autó, 
            mégpedig a legutolsó lépés sebességével párhuzamos irányba. Kis kezdeti sebességgel.
            """
            pos_new = pos
            spd_new = spd_new/LA.norm(spd_new)
            #TODO 1.1: INNEN folytatni. Nem fasza. random lepkedve kifagy. manualban jatszva faja, de a szakaszok es
            #celbaerkezes meg hianyzik.

        elif self.start_line[1] < pos_new[1] < self.start_line[3] \
                and pos_old[0] >= self.start_line[0] > pos_new[0]:  # ha visszafelé indul
            reward = -10
            end = True
        else:
            reward, curr_dist, pos = self.get_reward(pos_new) # normál esetben a reward
        if np.array_equal(spd_new, [0, 0]): # ha az autó megáll, vége
            end = True
        return spd_new, pos_new, reward if reward >= 0 else 2 * reward, end
        #return np.array([spd_new[0], spd_new[1], pos_new[0], pos_new[1]]), reward if reward >= 0 else 2*reward, end

    #TODO 1.2: lecsekkolni mukodik-e ez acelbaer fuggveny
    def celbaer(self, pos, spd, celbal, celjob):
        """
        Ha a Pos - ból húzott Spd vektor metszi a celvonalat(Szakasz(!),nem egynes) akkor 1 - et ad vissza(true)
        t2 az az ertek ami mgmondja hogy a Spd hanyadánál metszi a celvonalat.Ha t2 = 1 akkor a Spd vektor eppenhogy
        eleri a celvonalat.
        """
        """
        keplethez kello ertekek. p1, es p2 pontokkal valamint v1 es v2 iranyvektorokkal adott egyenesek metszespontjat
        nezzuk, ugy hogy a celvonal egyik pontjabol a masikba mutat a v1, a v2 pedig a sebesseg, p2pedig a pozicio
        """

        v1y = celjob[0] - celbal[0]
        v1z = celjob[1] - celbal[1]
        v2y = spd[0]
        v2z = spd[1]

        p1y = celbal[0]
        p1z = celbal[1]
        p2y = pos[0]
        p2z = pos[1]

        """
        t2 azt mondja hogy a p1 pontbol v1 iranyba indulva v1 hosszanak hanyadat kell megtenni hogy elerjunk a 
        metszespontig. Ha t2=1 epp v2vegpontjanal van a metszespopnt. t1,ugyanez csak p1 es v2-vel.
        """
        t2 = (-v1y * p1z + v1y * p2z + v1z * p1y - v1z * p2y) / (-v1y * v2z + v1z * v2y)
        t1 = (p1y * v2z - p2y * v2z - v2y * p1z + v2y * p2z) / (-v1y * v2z + v1z * v2y)

        """
        Annak eldontese hogy akkor az egyenesek metszespontja az most a
        szakaszokon belulre esik-e: Ha mindket t, t1 es t2 is kisebb mint 1 és
        nagyobb mint 0
        """
        celba = (0 < t1) and (t1 < 1) and (0 < t2) and (t2 < 1)

        if not celba:
            t2 = 0

        return celba, t2

    def is_on_track(self, pos):
        # a pálya színe és a pozíciónk pixelének színe alapján
        # visszaadja, hogy rajta vagyunk -e a pályán

        if pos[0] > np.shape(self.trk_pic)[1] or pos[1] > np.shape(self.trk_pic)[0] or \
                        pos[0] < 0 or pos[1] < 0 or np.isnan(pos[0]) or np.isnan(pos[1]):
            return False
        else:
            return np.array_equal(self.trk_pic[int(pos[1]), int(pos[0])], self.trk_col)

    def gg_action(self, action):
        # az action-ökhöz tartozó vektor értékek
        # első futáskor cash-eljúk

        if self.gg_actions is None:
            self.gg_actions = [None] * 361 # -180..180-ig, fokonként megnézzük a sugarat.
            for act in range(-180, 180):
                if -180 <= act < 180:
                    # a GGpic 41x41-es B&W bmp. A közepétől nézzük, meddig fehér. (A közepén,
                    # csak hogy látszódjon, van egy fekete pont!
                    xsrt, ysrt = 21, 21
                    r = 1
                    pix_in_gg = True
                    x, y = xsrt, ysrt
                    while pix_in_gg:
                        # lépjünk az Act irányba +1 pixelnyit, mik x és y ekkor:
                        #rad = np.pi / 4 * (act + 3)
                        rad = (act+180) * np.pi / 180
                        y = ysrt + round(np.sin(rad) * r)
                        x = xsrt + round(np.cos(rad) * r)
                        r = r + 1

                        # GG-n belül vagyunk-e még?
                        pix_in_gg = np.array_equal(self.gg_pic[int(x - 1), int(y - 1)], [255, 255, 255, 255])

                    self.gg_actions[act - 1] = (-(x - xsrt), y - ysrt)
                else:
                    self.gg_actions[act - 1] = (0, 0)

        return self.gg_actions[action - 1]

    def reset(self):
        """ha vmiért vége egy menetnek, meghívódik"""

        # 0-ázza a start poz-tól való távolságot a reward fv-hez
        self.prev_dist = 0

        # ha a random indítás be van kapcsolva, akkor új kezdő pozíciót választ
        if self.random_init:
            self.starting_pos = self.track_indices[randint(0, len(self.track_indices) - 1)]
            self.prev_dist = self.get_reward(self.starting_pos)





    def normalize_data(self, data_orig):
        """
        a háló könnyebben, tanul, ha az értékek +-1 közé esnek, ezért normalizáljuk őket
        pozícióból kivonjuk a pálya méretének a felét, majd elosztjuk a pálya méretével
        """

        n_rows = data_orig.shape[0]
        data = np.zeros((n_rows, 4))
        sizeX = np.shape(self.trk_pic)[1]
        sizeY = np.shape(self.trk_pic)[0]
        data[:, 0] = (data_orig[:, 0] - sizeX / 2) / sizeX
        data[:, 2] = (data_orig[:, 2] - sizeX / 2) / sizeX
        data[:, 1] = (data_orig[:, 1] - sizeY / 2) / sizeY
        data[:, 3] = (data_orig[:, 3] - sizeY / 2) / sizeY

        return data

    #TODO: ezt átírni, mert így nem tetszik. :)
    def get_reward(self, pos_new):

        #reward függvény
        #az előzőpontból a cél felé megtett utat "díjazza"
        #:param pos_new: a mostani pozíció
        #:return: reward értéke

        trk = rgb2gray(self.trk_pic) # szürkeárnyalatosban dolgozunk
        col = rgb2gray(np.reshape(self.track_inside_color, (1, 1, 3)))
        pos_new = np.array(pos_new, dtype='int32')
        tmp = [0]
        r = 0

        # az algoritmus úgy működik, hogy az aktuális pozícióban egyre negyobb sugárral
        # létrehoz egy diszket, amivel megnézi, hogy van -e r sugarú környezetében piros pixel
        # ha igen, akkor azt a pixelt kikeresi a dist_dict-ből, majd a kapott értéket kivonja
        # az előző lépésben kapott-ból, így megkapjuk, hogy mennyit tett meg azóta és ez a reward

        while not np.any(tmp):
            r = r + 1 # növeljük a disc sugarát
            tmp = trk[pos_new[1] - r:pos_new[1] + r + 1, pos_new[0] - r:pos_new[0] + r + 1] # vesszük az aktuális pozíció körüli 2rx2r-es négyzetet
            mask = disk(r)
            tmp = np.multiply(mask, tmp) # maszkoljuk a disc-kel
            tmp[tmp != col] = 0 # megnézzük, hogy van -e benne piros

        indices = [p[0] for p in np.nonzero(tmp)] # ha volt benne piros, akkor lekérjük a pozícióját
        offset = [indices[1] - r, indices[0] - r] # eltoljuk, hogy megkapjuk a kocsihoz viszonyított relatív pozícióját
        pos = np.array(pos_new + offset) # kiszámoljuk a pályán lévő pozícióját a pontnak
        curr_dist = self.dists[tuple(pos)] # a dist_dict-ből lekérjük a start-tól való távolságát
        reward = curr_dist - self.prev_dist # kivonjuk az előző lépésben kapott távolságból
        self.prev_dist = curr_dist # atz új lesz a régi, hogy a követkző lépésben legyen miből kivonni

        return reward, curr_dist, pos



    def __get_dists(self, rajz=False):
        """
        "feltérképezi" a pályát a reward fv-hez
        a start pontban addig növel egy korongot, amíg a korong a pálya egy belső pixelét (piros) nem fedi
        ekkor végigmegy a belső rész szélén és eltárolja a távolságokat a kezdőponttól úgy,
        hogy közvetlenül a pálya széle mellett menjen
        úgy kell elképzelni, mint a labirintusban a falkövető szabályt

        :return: dictionary, ami (pálya belső pontja, távolság) párokat tartalmaz
        """

        dist_dict = {} # dictionary, (pálya belső pontja, távolság) párokat tartalmaz
        start_point = self.starting_pos
        trk = rgb2gray(self.trk_pic) # szürkeárnyalatosban dolgozunk
        col = rgb2gray(np.reshape(np.array(self.track_inside_color), (1, 1, 3)))[0, 0]
        tmp = [0]
        r = 0 # a korong sugarát 0-ra állítjuk
        while not np.any(tmp): # amíg nincs belső pont fedésünk
            r = r + 1 # növeljük a sugarat
            mask = disk(r) # létrehozzuk a korongot (egy mátrixban 0-ák és egyesek)
            tmp = trk[start_point[1] - r:start_point[1] + r + 1, start_point[0] - r:start_point[0] + r + 1] # kivágunk
            # a képből egy kezdőpont kp-ú, ugyanekkora részt
            tmp = np.multiply(mask, tmp) # maszkoljuk a koronggal
            tmp[tmp != col] = 0 # a kororngon ami nem piros azt 0-ázzuk

        indices = [p[0] for p in np.nonzero(tmp)] #az első olyan pixel koordinátái, ami piros
        offset = [indices[1] - r, indices[0] - r] # eltoljuk, hogy a kp-tól megkapjuk a relatív távolságvektorát
        # (a mátrixban ugye a kp nem (0, 0) (easter egg bagoly) indexű, hanem középen van a sugáral le és jobbra eltolva)
        start_point = np.array(start_point + offset) # majd a kp-hoz hozzáadva megkapjuk a képen a pozícióját az első referenciapontnak
        dist = 0
        dist_dict[tuple(start_point)] = dist # ennek 0 a távolsága a kp-tól
        JOBB, FEL, BAL, LE = [1, 0], [0, -1], [-1, 0], [0, 1]  # [x, y], tehát a mátrix indexelésekor fordítva, de a pozícióhoz hozzáadható azonnal
        dirs = [JOBB, FEL, BAL, LE]
        direction_idx = 0
        point = start_point
        if rajz:
            self.draw_track()
        while True:
            dist += 1 # a távolságot növeli 1-gyel
            bal_ford = dirs[(direction_idx + 1) % 4] # a balra lévő pixel eléréséhez
            jobb_ford = dirs[(direction_idx - 1) % 4] # a jobbra lévő pixel eléréséhez
            if trk[point[1] + bal_ford[1], point[0] + bal_ford[0]] == col: # ha a tőlünk balra lévő pixel piros
                direction_idx = (direction_idx + 1) % 4 # akkor elfordulunk balra
                point = point + bal_ford
            elif trk[point[1] + dirs[direction_idx][1], point[0] + dirs[direction_idx][0]] == col: # ha az előttünk lévő pixel piros
                point = point + dirs[direction_idx] # akkor arra megyünk tovább
            else:
                direction_idx = (direction_idx - 1) % 4 # különben jobbra fordulunk
                point = point + jobb_ford

            dist_dict[tuple(point)] = dist # a pontot belerakjuk a dictionarybe

            if rajz:
                plt.plot([point[0]], [point[1]], 'yo')

            if np.array_equal(point, start_point): # ha visszaértünk az elejére, akkor leállunk
                break
        if rajz:
            plt.draw()
            plt.pause(0.001)

        return dist_dict
