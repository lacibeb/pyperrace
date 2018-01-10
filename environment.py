"Ez a HPC-s verziÃ³"

OnHPC = True

if OnHPC:
    use_matplotlib = False
else:
    use_matplotlib = True

if use_matplotlib:
    import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import numpy as np
import random as rnd
from numpy import linalg as LA
from random import randint
from skimage.morphology import disk
from skimage.color import rgb2gray

class PaperRaceEnv:
    """ez az osztály biztosítja a tanuláshoz a környezetet"""

    def __init__(self, trk_pic, trk_col, gg_pic, sections, random_init, track_inside_color=None,):

        # ha nincs megadva a pálya belsejének szín, akkor pirosra állítja
        # ez a rewardokat kiszámoló algoritmus működéséhez szükséges
        if track_inside_color is None:
            self.track_inside_color = np.array([255, 0, 0], dtype='uint8')
        else:
            self.track_inside_color = np.array(track_inside_color, dtype='uint8')

        # a palya kulso szine is jobb lenne nem itt, de most ganyolva ide rakjuk
        self.track_outside_color = np.array([255, 255, 255], dtype='uint8')

        self.trk_pic = mpimg.imread(trk_pic)  # beolvassa a pályát
        self.trk = rgb2gray(self.trk_pic)  # szürkeárnyalatosban dolgozunk
        self.trk_col = trk_col  # trk_pic-en a pálya színe
        self.gg_pic = mpimg.imread(gg_pic) # beolvassa a GG diagramot
        self.steps = 0  # az eddig megtett lépések száma

        self.col_in = rgb2gray(np.reshape(np.array(self.track_inside_color), (1, 1, 3)))[0, 0]
        self.col_out = rgb2gray(np.reshape(np.array(self.track_outside_color), (1, 1, 3)))[0, 0]

        # Ha be van kapcsolva az autó véletlen pozícióból való indítása, random szakaszból indulunk
        self.random_init = random_init

        # Az első szakasz a sectionban, lesz a startvonal
        self.sections = sections

        # A kezdo pozicio a startvonal fele, es onnan 1-1 pixellel "arrebb" Azert hogy ne legyen a startvonal es a
        # kezdeti sebesseg metszo.
        # ezen a ponton section_nr = 0, az elso szakasz a listaban (sections) a startvonal
        start_line = sections[0]
        #ez valmiert igy volt, egyelore igy hagyom...
        self.start_line = start_line

        # A startvonalra meroleges iranyvektor:
        e_start_x = int(np.floor((start_line[0] - start_line[2])))
        e_start_y = int(np.floor((start_line[1] - start_line[3])))
        self.e_start_spd = np.array([e_start_y, -e_start_x]) / np.linalg.norm(np.array([e_start_y, -e_start_x]))

        # A startvonal közepe:
        self.start_x = int(np.floor((start_line[0] + start_line[2]) / 2))
        self.start_y = int(np.floor((start_line[1] + start_line[3]) / 2))
        # A kezdő pozíció, a startvonal közepétől, a startvonalra merőleges irányba egy picit eltolva:
        self.starting_pos = np.array([self.start_x, self.start_y]) + np.array([int(self.e_start_spd[0] * 10), int(self.e_start_spd[1] * 10)])

        #a kezdo sebesseget a startvonalra merolegesre akarjuk:
        self.starting_spd = self.e_start_spd * 40

        self.gg_actions = None # az action-ökhöz tartozó vektor értékeit cash-eli a legelajén és ebben tárolja

        # van egy ref... fv. Ahhoz hog az jol mukodjon, kellenek mindig egy "előző" lépés ref adatai. Ezek:
        self.prev_dist_in = 0
        self.prev_dist_out = 0
        self.prev_pos_in = np.array([0, 0])
        self.prev_pos_out = np.array([0, 0])

        self.track_indices = [] # a pálya (szürke) pixeleinek pozícióját tartalmazza
        for x in range(self.trk_pic.shape[1]):
            for y in range(self.trk_pic.shape[0]):
                if np.array_equal(self.trk_pic[y, x, :], self.trk_col):
                    self.track_indices.append([x, y])

        self.dists_in = self.__get_dists_in(True) # a kezdőponttól való "távolságot" tárolja a reward fv-hez
        self.dists_out = self.__get_dists_out(True) # a kezdőponttól való "távolságot" tárolja
        # print("DictIn:", self.dists_in)
        # print("DictOut:", self.dists_out)


        # van egy referencia lepessor ami valahogy beér a célba (palya4) :
        self.ref_actions = np.array([0, -180, -95, -95, -90, -95])

        # van egy referencia lepessor ami valahogy beér a célba (palya5) :
        # self.ref_actions = np.array([0, 150, 180, -160, -160, -160, -150, -90, -90, -110, -110, -120, -110, -110, 0,
        #                             90, -90, 90, -140, 90, 110, 90, 120, 120, 120, 120, 100, -20, -10, 0, 0, 0])

        # ehhez van egy init, ami eloallitja a belso iv menten mert elorehaladast minden lepesben
        #self.ref_dist = self.__get_ref_dicts(self.ref_actions)
        self.ref_dist, self.ref_steps = self.__get_ref_dicts(self.ref_actions)


    def draw_track(self):
        # pálya kirajzolása
        if use_matplotlib:
            plt.imshow(self.trk_pic)

        # Szakaszok kirajzolása
        for i in range(len(self.sections)):

            if use_matplotlib:
                X = np.array([self.sections[i][0], self.sections[i][2]])
                Y = np.array([self.sections[i][1], self.sections[i][3]])
                plt.plot(X, Y, color='blue')

    def step(self, spd_chn, spd_old, pos_old, draw, color):

        end = False

        # az aktuális sebesség irányvektora:
        e1_spd_old = spd_old / np.linalg.norm(spd_old)
        e2_spd_old = np.array([-1 * e1_spd_old[1], e1_spd_old[0]])

        # a sebessegvaltozas lokalisban
        spd_chn = np.asmatrix(spd_chn)

        # a sebesseg valtozás globálisban:
        spd_chn_glb = np.round(np.column_stack((e1_spd_old, e2_spd_old)) * spd_chn.transpose())

        # az új sebességvektor globalisban:
        spd_new = spd_old + np.ravel(spd_chn_glb)

        # az uj pozicio globalisban:
        pos_new = pos_old + spd_new
        # print("PN:", pos_new, "PO:", pos_old)

        # Az adott lépésben érvényes referencia jellmezök:
        # # curr_dist_in_old, pos_temp_in_old, curr_dist_out_old, pos_temp_out_old = self.get_ref(pos_old)
        # # megnezzuk, az uj pozicioban hol jarunk:
        # curr_dist_in_new, pos_temp_in_new, curr_dist_out_new, pos_temp_out_new = self.get_ref(pos_new)

        # meghivjuk a sectionpass fuggvenyt, hogy megkapjuk szakitottunk-e at szakaszt, es ha igen melyiket,
        # es az elmozdulas hanyad reszenel
        crosses, t2, section_nr = self.sectionpass(pos_old, spd_new)

        # megnezzuk palyan van-e es ha lemegya kkor kint vagy bent:
        step_on_track, inside, outside = self.is_on_track(pos_new)

        # ===================
        # Lépések:
        # ===================

        # Ha lemegy a palyarol:
        if not step_on_track:
            # ha atszakit egy szakaszhatart, es ez az utolso is, tehat pont celbaert es ugy esett le a palyarol:
            if crosses and section_nr == len(self.sections) - 1:
                print("\033[91m {}\033[00m" .format("CELBAERT KI"))
                reward = -t2 - 3
                end = True
            # nem ert celba:
            else:
                # eloszor meg kell hatarozni hogy hol megy le a palyarol:
                pos_chk_tmp = np.array([int(pos_old[0]), int(pos_old[1])])
                while True:
                    pos_chk_int = np.array([int(pos_chk_tmp[0]), int(pos_chk_tmp[1])])
                    ontrack, inside, outside = self.is_on_track(pos_chk_int)
                    if not ontrack and ((tuple(pos_chk_int) in self.dists_out) or (tuple(pos_chk_int) in self.dists_in)):
                        break
                    pos_chk_tmp = pos_chk_tmp + (pos_new - pos_old)/np.linalg.norm((pos_new - pos_old))

                pos_chk = pos_chk_int
                # print("PKi: ", pos_chk, inside, outside)

                # ebben a pozicioban kell "visszapattanni". Mas lesz ha bent es mas ha kint mentunk le
                curr_dist_in, pos_in, curr_dist_out, pos_out = self.get_ref(pos_chk)

                # a palya "szelessege"
                tck_wdt = np.linalg.norm((pos_in - pos_out))

                # a visszapattinto felulet normalisa
                if inside:
                    en_ref = (pos_in - pos_out) / tck_wdt
                else:
                    en_ref = (pos_out - pos_in) / tck_wdt

                # a visszapattanasi pont ne pont a fal legyen, mert akkor neha van hogy megsem palyan
                # marad a pattanas utan. A pattanasi ponttol beljebb, a normalils iranyaba a szelesseg
                # 10%-át
                pos_patt = pos_chk - en_ref * tck_wdt * 0 # nem 10% hanem 0

                # visszapattanas elotti vektor
                v_bef = pos_chk - pos_old

                # a visszapattano vektor
                v_aft = v_bef - 2 * (np.dot(v_bef, en_ref)) * en_ref

                # az uj iranyba legyen kb 10 pixel hosszu a sebesseg
                spd_new = (v_aft / np.linalg.norm(v_aft)) * 40

                # kirajzoljuk a falig megtett szakaszt
                if draw:  # kirajzolja az autót
                    if use_matplotlib:
                        X = np.array([pos_old[0], pos_chk[0]])
                        Y = np.array([pos_old[1], pos_chk[1]])
                        plt.plot(X, Y, color='green')

                    if use_matplotlib:
                        X = np.array([pos_in[0], pos_out[0]])
                        Y = np.array([pos_in[1], pos_out[1]])
                        plt.plot(X, Y, color='yellow')


                # megcsinlajuk a visszapattanas utani lepest
                pos_old = pos_patt
                pos_new = pos_old + spd_new

                # ha ez alepes valamiert palyan kivulre esik azt is kezelni kell
                step_on_track, inside, outside = self.is_on_track(pos_new)
                crosses, t2, section_nr = self.sectionpass(pos_old, spd_new)
                # ha tehat nincs palyan:
                if not step_on_track:
                    print("PATTAN LE")
                    end = True
                    reward = -10
                else:
                    #ha palyan van de pattanas kozben epp celbaert, azt is kezelni kell
                    if crosses and section_nr == len(self.sections) - 1:
                        print("PATTAN CELBA")
                        end = True
                        reward = -2
                    else:
                        print("\033[95m {}\033[00m" .format("PATTAN"))
                        reward = -5
                # Ezt az egeszet augy nem igy kellene csinalni, hanem ug hogy ilyenkor meghivja sajat magat
                # ez a step fv. nem ilyen esetekeles azon belul esetekkel benazni... Csak nem vok benne biztos hogy
                # kell az ilyet csinalni, inkabb benzok ezzel



        # Ha nem ment ki a palyarol:
        else:

            # megnezzuk hol jarunk (get_dist.. majd atirni ezt a fugvenyt)
            # print("PosOld:", pos_old, "PosNew:", pos_new)
            curr_dist_in_old, pos_temp_in_old, curr_dist_out_old, pos_temp_out_old = self.get_ref(pos_old)
            # megnezzuk, az uj pozicioban hol jarunk:
            curr_dist_in_new, pos_temp_in_new, curr_dist_out_new, pos_temp_out_new = self.get_ref(pos_new)
            # print(curr_dist_in_old, curr_dist_in_new)
            curr_dist_in = curr_dist_in_old
            reward = -1

            # ha visszafordulna:
            if curr_dist_in_new < curr_dist_in_old:
                print("\033[96m {}\033[00m".format("FORDUL"))
                # Eleinte ezt is sokat csinalja ami nem jo, mert enm megy vegig a celig. legyen az hogy ilyenkor is
                # mint a palyaelhagyaskor a "visszapattanas". berakjuk a palya kozepere, "jo" iranyba

                # az uj pozicio a palya kozepe:
                pos_new = pos_temp_in_old + (pos_temp_out_old - pos_temp_in_old) * rnd.uniform(0.4, 0.6)

                # az uj sebesseg meroleges a kozepvonalra. ehhez a ket szel osszakoto iranyvekror:
                e_szel = (pos_temp_out_old - pos_temp_in_old) / np.linalg.norm([pos_temp_out_old - pos_temp_in_old])

                # a fentire meroleges irany:
                n_szel = np.array([-e_szel[1], e_szel[0]])

                # az uj sebesseg:
                spd_new = n_szel * 30

                # a jutalom (bunti)
                reward = -2

                # es ne legyen vege
                end = False

                # Es akkor most persze mindent megint meg kellene nezni, hogy mivan ha ezzel a lepessel most kiesne, vagy
                # szakaszt lepne, vegy celbaerne vagy stb... szoval az egeszet at ken irni, hogy ilyenkor meghivhassa
                # sajat magat a step fv. vagy valmai ilyesmi... nemtom




            # ha a 0. szakaszt, azaz startvonalat szakit at (nem visszafordult hanem eleve visszafele indul):
            elif (crosses and section_nr == 0):
                print("VISSZAKEZD")
                reward = -33
                curr_dist_in = 0.1
                end = True

            # ha atszakit egy szakaszhatart, es ez az utolso is, tehat pont celbaert:
            elif crosses and section_nr == len(self.sections) - 1 and step_on_track:
                print("\033[92m {}\033[00m".format("CELBAERT BE"))
                reward = -t2
                curr_dist_in, pos_in, curr_dist_out, pos_out = self.get_ref(pos_new)
                end = True

            #ha atszakitunk egy szakaszt (senem elso, senem utolso) kapjon kis jutalmat. hatha segit tanulaskor a hulyejenek
            elif crosses and section_nr < len(self.sections)-1:
                print("SZAKASZ")
                curr_dist_in, pos_in, curr_dist_out, pos_out = self.get_ref(pos_new)
                reward = 1

            if use_matplotlib:
                X = np.array([pos_temp_in_old[0], pos_temp_out_old[0]])
                Y = np.array([pos_temp_in_old[1], pos_temp_out_old[1]])
                plt.plot(X, Y, color='magenta')

        # ha barmi miatt az autó megáll, sebesseg zerus, akkor vége
        if np.array_equal(spd_new, [0, 0]):
            end = True

        # Ha akarjuk, akkor itt rajzoljuk ki az aktualis lepes abrajat (lehet maskor kene)
        if draw: # kirajzolja az autót
            if use_matplotlib:
                X = np.array([pos_old[0], pos_new[0]])
                Y = np.array([pos_old[1], pos_new[1]])
                plt.plot(X, Y, color=color)

        return spd_new, pos_new, reward, end, section_nr




    def is_on_track(self, pos):
        """ a pálya színe és a pozíciónk pixelének színe alapján visszaadja, hogy rajta vagyunk -e a pályán, illetve kint
           vagy bent csusztunk le rola... Lehetne tuti okosabban mint ahogy most van."""

        # meg kell nezni egyatalan a palya kepen belul van-e a pos
        # print(pos)
        if int(pos[1]) > (self.trk_pic.shape[0] - 1) or int(pos[0]) > (self.trk_pic.shape[1] - 1):
            inside = True
            outside = True
            ontrack = False
        else:
            inside = False
            outside = False
            ontrack = True

            if np.array_equal(self.trk_pic[int(pos[1]), int(pos[0])], self.track_inside_color):
                inside = True
            if np.array_equal(self.trk_pic[int(pos[1]), int(pos[0])], self.track_outside_color):
                outside = True
            if inside or outside:
                ontrack = False

            """
            if pos[0] > np.shape(self.trk_pic)[1] or pos[1] > np.shape(self.trk_pic)[0] or pos[0] < 0 or pos[1] < 0 or np.isnan(pos[0]) or np.isnan(pos[1]):
                ontrack = False
            else:
                ontrack = np.array_equal(self.trk_pic[int(pos[1]), int(pos[0])], self.trk_col)
            """

        return ontrack, inside, outside

    def gg_action(self, action):
        # az action-ökhöz tartozó vektor értékek
        # első futáskor cash-eljúk

        if self.gg_actions is None:
            self.gg_actions = [None] * 361 # -180..180-ig, fokonként megnézzük a sugarat.
            for act in range(-180, 181):
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

        """
        # ha a random indítás be van kapcsolva, akkor új kezdő pozíciót választ
        if self.random_init:
            self.starting_pos = self.track_indices[randint(0, len(self.track_indices) - 1)]
            self.prev_dist = self.get_ref_time(self.starting_pos)
        """
        if self.random_init:
            self.section_nr = randint(0, len(self.sections) - 2)
        else:
            self.section_nr = 0 # kezdetben a 0. szakabol indul a jatek
        # print("SectNr: ", self.section_nr)

        start_line = self.sections[self.section_nr]

        e_start_x = int(np.floor((start_line[0] - start_line[2])))
        e_start_y = int(np.floor((start_line[1] - start_line[3])))
        self.e_start_spd = np.array([e_start_y, -e_start_x]) / np.linalg.norm(np.array([e_start_y, -e_start_x]))

        # A startvonal közepe:
        self.start_x = int(np.floor((start_line[0] + start_line[2]) / 2))
        self.start_y = int(np.floor((start_line[1] + start_line[3]) / 2))
        # A kezdő pozíció, a startvonal közepétől, a startvonalra merőleges irányba egy picit eltolva:
        self.starting_pos = np.array([self.start_x, self.start_y]) + np.array([int(self.e_start_spd[0] * 10), int(self.e_start_spd[1] * 10)])

        #a kezdo sebesseget a startvonalra merolegesre akarjuk:
        #self.starting_spd = self.starting_spd



    def sectionpass(self, pos, spd):
        """
                Ha a Pos - ból húzott Spd vektor metsz egy szakaszt(Szakasz(!),nem egynes) akkor crosses = 1 - et ad vissza(true)
                A t2 az az ertek ami mgmondja hogy a Spd vektor hanyadánál metszi az adott szakaszhatart. Ha t2 = 1 akkor a Spd
                vektor eppenhogy eleri a celvonalat.

                Ezt az egeszet nezi a kornyezet, azaz a palya definialasakor kapott osszes szakaszra (sectionlist) Ha a
                pillanatnyi pos-bol huzott spd barmely section-t jelzo szakaszt metszi, visszaadja hogy:
                crosses = True, azaz hogy tortent szakasz metszes.
                t2 = annyi amennyi, azaz hogy spd hanyadanal metszette
                section_nr = ahanyadik szakaszt metszettuk epp.
                """
        """
        keplethez kello idediglenes ertekek. p1, es p2 pontokkal valamint v1 es v2 iranyvektorokkal adott egyenesek metszespontjat
        nezzuk, ugy hogy a celvonal egyik pontjabol a masikba mutat a v1, a v2 pedig a sebesseg, p2pedig a pozicio
        """
        section_nr = 0
        t2 = 0
        crosses = False

        for i in range(len(self.sections)):
            v1y = self.sections[i][2] - self.sections[i][0]
            v1z = self.sections[i][3] - self.sections[i][1]
            v2y = spd[0]
            v2z = spd[1]

            p1y = self.sections[i][0]
            p1z = self.sections[i][1]
            p2y = pos[0]
            p2z = pos[1]


            # mielott vizsgaljuk a metszeseket, gyorsan ki kell zarni, ha a parhuzamosak a vizsgalt szakaszok
            # ilyenkor 0-val osztas lenne a kepletekben
            if -v1y * v2z + v1z * v2y == 0:
                crosses = False
            # Amugy mehetnek a vizsgalatok
            else:
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
                cross = (0 < t1) and (t1 < 1) and (0 < t2) and (t2 < 1)

                if cross:
                    crosses = True
                    section_nr = i
                    break
                else:
                    crosses = False
                    t2 = 0
                    section_nr = 0
        #print("CR: ",crosses,"t2: ",t2)
        return crosses, t2, section_nr

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

    def get_ref(self, pos_new):

        """Ref adatokat ado fuggveny.
        pos_new csak palyan levo pont lehet, ha enm akkor hibat fog adni.
        INput tehat:
        pos_new: a palya egy adott pontja

        Output:
        belso iv menten megtett ut,kulso iv menten megtett ut, belso iv referencia pontja, es kulso iv ref pontja"""

        # az algoritmus alapvetoen  úgy működik, hogy az aktuális pozícióban egyre negyobb sugárral
        # létrehoz egy diszket, amivel megnézi, hogy van -e r sugarú környezetében piros pixel
        # ha igen, akkor azt a pixelt kikeresi a dist_dict-ből, majd megnezi ehhez mennyi a ref sebesseggel mennyi ido
        # jar

        #self.trk = rgb2gray(self.trk_pic) # szürkeárnyalatosban dolgozunk
        pos_new = np.array(pos_new, dtype='int32')

        # pos_new-el egy vonalban levo belso pont meghatarozasa---------
        tmp_in = [0]
        #_in = rgb2gray(np.reshape(self.track_inside_color, (1, 1, 3)))
        r_in = 0
        while not np.any(tmp_in):
            r_in = r_in + 1 # növeljük a disc sugarát
            tmp_in = self.trk[pos_new[1] - r_in:pos_new[1] + r_in + 1, pos_new[0] - r_in:pos_new[0] + r_in + 1] # vesszük az aktuális pozíció körüli 2rx2r-es négyzetet
            mask_in = disk(r_in)
            tmp_in = np.multiply(mask_in, tmp_in) # maszkoljuk a disc-kel
            tmp_in[tmp_in != self.col_in] = 0 # megnézzük, hogy van -e benne belso szin
        indices_in = [p[0] for p in np.nonzero(tmp_in)] # ha volt benne piros, akkor lekérjük a pozícióját
        offset_in = [indices_in[1] - r_in, indices_in[0] - r_in] # eltoljuk, hogy megkapjuk a kocsihoz viszonyított relatív pozícióját
        pos_in = np.array(pos_new + offset_in) # kiszámoljuk a pályán lévő pozícióját a pontnak

        # pos_new-el egy vonalban levo kulso pont meghatarozasa---------
        tmp_out = [0]
        #col_out = rgb2gray(np.reshape(self.track_outside_color, (1, 1, 3)))
        r_out = 0
        while not np.any(tmp_out):
            r_out = r_out + 1  # növeljük a disc sugarát
            tmp_out = self.trk[pos_new[1] - r_out:pos_new[1] + r_out + 1, pos_new[0] - r_out:pos_new[0] + r_out + 1]  # vesszük az aktuális pozíció körüli 2rx2r-es négyzetet
            mask_out = disk(r_out)
            tmp_out = np.multiply(mask_out, tmp_out)  # maszkoljuk a disc-kel
            tmp_out[tmp_out != self.col_out] = 0  # megnézzük, hogy van -e benne kulso szin
        indices_out = [p[0] for p in np.nonzero(tmp_out)]  # ha volt benne piros, akkor lekérjük a pozícióját
        offset_out = [indices_out[1] - r_out, indices_out[0] - r_out]  # eltoljuk, hogy megkapjuk a kocsihoz viszonyított relatív pozícióját
        pos_out = np.array(pos_new + offset_out)  # kiszámoljuk a pályán lévő pozícióját a pontnak


        # A kapott belso es kulso pontokrol megnezni milyen messze vannak a starttol:--------------------------

        # Ha kozel vagyunk a falhoz, 1 sugaru r adodik, es egy pixelnyivel mindig pont mas key-t ker mint ami letezik.
        # Ezt  elkerulendo, megnezzuk hogy amivel meg akarjuk hivni az valoban benne van-e a dictekben.
        # tehat ha a dict-ek tartalmazzak pos_in es pos_out-ot:
        if tuple(pos_in) in self.dists_in and tuple(pos_out) in self.dists_out:
            curr_dist_in = self.dists_in[tuple(pos_in)] # a dist_dict-ből lekérjük a start-tól való távolságát
            curr_dist_out = self.dists_out[tuple(pos_out)] # a dist_dict-ből lekérjük a start-tól való távolságát
        else:
            # ha nincsennek a kapott potok a dict-ben, akkor a külsö-belsö pontokat osszekoto szakaszon levo ponthoz
            # kerunk ref-et. Ez sem igazan jo megoldas dehat...
            pos_fel = pos_out + (pos_in - pos_out) * 0.5 # rnd.uniform(0.4, 0.6)
            curr_dist_in, pos_in, curr_dist_out, pos_out = self.get_ref(pos_fel)

        return curr_dist_in, pos_in, curr_dist_out, pos_out

    """
    def get_reward(self, pos_old, pos_new, step_nr):
        ""Reard ado fuggveny. Egy adott lepeshez (pos_old - pos new) ad jutalmat. Eredetileg az volt hogy -1 azaz mint
        mint eltelt idő. Most megnezzuk mivan ha egy referencia lepessorhoz kepest a nyert vagy veszetett ido lesz.
        kb. mint a delta_time channel a MOTEC-ben""

        # Fent az env initbe kell egy referencia lepessor. Actionok, egy vektorban...vagy akarhogy.
        # Az Actionokhoz tudjuk a pos-okat minden lepesben
        # Es tudjuk a dist_in es dist_outokat is minden lepeshez (a lepes egy timestep elvileg)
        # A fentiek alapjan pl.:Look-up table szeruen tudunk barmilyen dist-hez lepest (idot) rendelni

        # Megnezzuk hogy a pos_old es a pos_new milyen dist_old es dist_new-hez tartozik (in vagy out, vagy atlag...)

        # Ehez a dist_old es dist new-hoz megnezzuk hogy a referencia lepessor mennyi ido alatt jutott el ezek lesznek
        # step_old es step_new.

        # A step_old es step_new kulonbsege azt adja hogy azt a tavot, szakaszt, amit a jelenlegi pos_old, pos_new
        # megad, azt a ref lepessor, mennyi ido tette meg. A jelenlegi az 1 ido, hiszen egy lepes. A ketto kulonbsege
        # adja majd pillanatnyi rewardot.



        return reward
    """

    def get_time_diff(self, pos_old, pos_new, act_rew):
        """Reward ado fuggveny. Egy adott lepeshez (pos_old - pos new) ad jutalmat. Eredetileg az volt hogy -1 azaz mint
        mint eltelt idő. Most megnezzuk mivan ha egy referencia lepessorhoz kepest a nyert vagy veszetett ido lesz.
        kb. mint a delta_time channel a MOTEC-ben

        pre_dist_in, az aktualis lepessor, elozo pozicio belso iv menti tavolsaga,
        curr_dist_in, az aktualis lepessor, jelenlegi pozicio belso iv menti tavolsaga
        act_rew: az aktualis lepessor mennyi ido alatt ert e tavolsagok kozott

        visszater a rew_dt, ami azt adja, hogy a referencia lepessorhoz kepest ez mennyivel tobb ido"""

        curr_dist_in, curr_pos_in, curr_dist_out, curr_pos_out = self.get_ref(pos_new)
        pre_dist_in, pre_pos_in, pre_dist_out, pre_pos_out = self.get_ref(pos_old)

        # amennyi ido (lepes) alatt a ref_actionok, a pre_dist-ből a curr_dist-be eljutottak--------------------------
        # look-up szerűen lesz. Először a bemenetek:
        x = self.ref_dist
        y = self.ref_steps

        xvals = np.array([pre_dist_in, curr_dist_in])
        print("elozo es aktualis tav:", xvals)

        # ezekre a tavolsagokra a referencia lepessor ennyi ido alaptt jutott el
        yinterp = np.interp(xvals, x, y, 0)
        # print("ref ennyi ido alatt jutott ezekre:", yinterp)

        # tehat ezt a negyasau lepest a referencia ennyi ido alatt tette meg (- legyen, hogy a kisebb ido legyen a magasabb
        # reward) :
        ref_delta = -yinterp[1] + yinterp[0]
        # print("a ref. ezen lepesnyi ideje:", ref_delta)

        # az atualis lepesben az eltelt ido nyilvan -1, illetve ha ido-bunti van akkor ennel tobb, eppen a reward
        # print("elozo es aktualis lepes kozott eltelt ido:", act_rew)

        # amenyivel az aktualis ebben a lepesben jobb, azaz kevesebb ido alatt tette meg ezt a elmozdulat, mint a ref
        # lepessor, az:
        rew_dt = -ref_delta + act_rew
        # print("az aktualis, ebben a lepesben megtett tavot ennyivel kevesebb ido alatt tette meg mint a ref. (ha (-) akkor meg több):", rew_dt)

        return rew_dt

    def __get_ref_dicts(self, ref_actions):

        # Fent az env initbe kell egy referencia lepessor. Actionok, egy vektorban...vagy akarhogy.
        # Az Actionokhoz tudjuk a pos-okat minden lepesben

        v = np.array(self.starting_spd)  # az elején a sebesség a startvonalra meroleges
        pos = np.array(self.starting_pos)  # kezdőpozíció beállítása
        epreward = 0

        steps_nr = range(0, len(ref_actions))

        ref_steps = np.zeros(len(ref_actions))
        ref_dist = np.zeros(len(ref_actions))

        if use_matplotlib:
             plt.clf()
        self.draw_track()

        for i in steps_nr:
            #nye = input('Give input')
            action = self.ref_actions[i]
            print(action)
            gg_action = self.gg_action(action)  # action-höz tartozó vektor lekérése
            v_new, pos_new, reward, end, section_nr = self.step(gg_action, v, pos, True, 'blue')
            curr_dist_in, pos_in, curr_dist_out, pos_out = self.get_ref(pos_new)
            ref_dist[i] = curr_dist_in
            epreward = epreward + reward
            ref_steps[i] = -epreward

            if True:

                if use_matplotlib:
                    plt.draw()

            v = v_new
            pos = pos_new

        print(ref_dist)
        print(ref_steps)

        return ref_dist, ref_steps

    def __get_dists_in(self, rajz=False):
        """
        "feltérképezi" a pályát a reward fv-hez
        a start pontban addig növel egy korongot, amíg a korong a pálya egy belső pixelét (piros) nem fedi
        ekkor végigmegy a belső rész szélén és eltárolja a távolságokat a kezdőponttól úgy,
        hogy közvetlenül a pálya széle mellett menjen
        úgy kell elképzelni, mint a labirintusban a falkövető szabályt

        :return: dictionary, ami (pálya belső pontja, távolság) párokat tartalmaz
        """

        dist_dict_in = {} # dictionary, (pálya belső pontja, távolság) párokat tartalmaz

        # a generalashoz a start pozicio alapbol startvonal kozepe lenne. De valahogy a startvonal kozeleben a dist az
        # szar tud lenni ezert az algoritmus kezdo pontjat a startvonal kicsit visszabbra tesszuk.
        # (TODO: megerteni miert szarakodik a dist, es kijavitani)
        start_point = np.array([self.start_x, self.start_y]) - np.array([int(self.e_start_spd[0] * 10), int(self.e_start_spd[1] * 10)])
        #trk = rgb2gray(self.trk_pic) # szürkeárnyalatosban dolgozunk
        #col_in = rgb2gray(np.reshape(np.array(self.track_inside_color), (1, 1, 3)))[0, 0]
        tmp = [0]
        r = 0 # a korong sugarát 0-ra állítjuk
        while not np.any(tmp): # amíg nincs belső pont fedésünk
            r = r + 1 # növeljük a sugarat
            mask = disk(r) # létrehozzuk a korongot (egy mátrixban 0-ák és egyesek)
            tmp = self.trk[start_point[1] - r:start_point[1] + r + 1, start_point[0] - r:start_point[0] + r + 1] # kivágunk
            # a képből egy kezdőpont kp-ú, ugyanekkora részt
            tmp = np.multiply(mask, tmp) # maszkoljuk a koronggal
            tmp[tmp != self.col_in] = 0 # a kororngon ami nem piros azt 0-ázzuk

        indices = [p[0] for p in np.nonzero(tmp)] #az első olyan pixel koordinátái, ami piros
        offset = [indices[1] - r, indices[0] - r] # eltoljuk, hogy a kp-tól megkapjuk a relatív távolságvektorát
        # (a mátrixban ugye a kp nem (0, 0) (easter egg bagoly) indexű, hanem középen van a sugáral le és jobbra eltolva)
        start_point = np.array(start_point + offset) # majd a kp-hoz hozzáadva megkapjuk a képen a pozícióját az első referenciapontnak
        dist = 0
        dist_dict_in[tuple(start_point)] = dist # ennek 0 a távolsága a kp-tól
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
            if self.trk[point[1] + bal_ford[1], point[0] + bal_ford[0]] == self.col_in: # ha a tőlünk balra lévő pixel piros
                direction_idx = (direction_idx + 1) % 4 # akkor elfordulunk balra
                point = point + bal_ford
            elif self.trk[point[1] + dirs[direction_idx][1], point[0] + dirs[direction_idx][0]] == self.col_in: # ha az előttünk lévő pixel piros
                point = point + dirs[direction_idx] # akkor arra megyünk tovább
            else:
                direction_idx = (direction_idx - 1) % 4 # különben jobbra fordulunk
                point = point + jobb_ford

            dist_dict_in[tuple(point)] = dist # a pontot belerakjuk a dictionarybe

            if rajz:
                if use_matplotlib:
                    plt.plot([point[0]], [point[1]], 'yo')

            if np.array_equal(point, start_point): # ha visszaértünk az elejére, akkor leállunk
                break
        if rajz:
            if use_matplotlib:
                plt.draw()

        return dist_dict_in

    def __get_dists_out(self, rajz=False):
        """
        "feltérképezi" a pályát a reward fv-hez
        a start pontban addig növel egy korongot, amíg a korong a pálya egy belső pixelét (piros) nem fedi
        ekkor végigmegy a belső rész szélén és eltárolja a távolságokat a kezdőponttól úgy,
        hogy közvetlenül a pálya széle mellett menjen
        úgy kell elképzelni, mint a labirintusban a falkövető szabályt

        :return: dictionary, ami (pálya belső pontja, távolság) párokat tartalmaz
        """

        dist_dict_out = {} # dictionary, (pálya belső pontja, távolság) párokat tartalmaz

        # a generalashoz a start pozicio alapbol startvonal kozepe lenne. De valahogy a startvonal kozeleben a dist az
        # szar tud lenni ezert az algoritmus kezdo pontjat a startvonal kicsit visszabbra tesszuk.
        # (TODO: megerteni miert szarakodik a dist, es kijavitani)
        start_point = np.array([self.start_x, self.start_y])
        #- np.array([int(self.e_start_spd[0] * 10), int(self.e_start_spd[1] * 10)])
        #trk = rgb2gray(self.trk_pic) # szürkeárnyalatosban dolgozunk
        #col_out = rgb2gray(np.reshape(np.array(self.track_outside_color), (1, 1, 3)))[0, 0]
        tmp = [0]
        r = 0 # a korong sugarát 0-ra állítjuk
        while not np.any(tmp): # amíg nincs belső pont fedésünk
            r = r + 1 # növeljük a sugarat
            mask = disk(r) # létrehozzuk a korongot (egy mátrixban 0-ák és egyesek)
            tmp = self.trk[start_point[1] - r:start_point[1] + r + 1, start_point[0] - r:start_point[0] + r + 1] # kivágunk
            # a képből egy kezdőpont kp-ú, ugyanekkora részt
            tmp = np.multiply(mask, tmp) # maszkoljuk a koronggal
            #???? con_in-nek kellene lennie
            tmp[tmp != self.col_out] = 0 # a kororngon ami nem piros azt 0-ázzuk

        indices = [p[0] for p in np.nonzero(tmp)] #az első olyan pixel koordinátái, ami piros
        offset = [indices[1] - r, indices[0] - r] # eltoljuk, hogy a kp-tól megkapjuk a relatív távolságvektorát
        # (a mátrixban ugye a kp nem (0, 0) (easter egg bagoly) indexű, hanem középen van a sugáral le és jobbra eltolva)
        start_point = np.array(start_point + offset) # majd a kp-hoz hozzáadva megkapjuk a képen a pozícióját az első referenciapontnak
        dist = 0
        dist_dict_out[tuple(start_point)] = dist # ennek 0 a távolsága a kp-tól
        JOBB, FEL, BAL, LE = [1, 0], [0, -1], [-1, 0], [0, 1]  # [x, y], tehát a mátrix indexelésekor fordítva, de a pozícióhoz hozzáadható azonnal

        # INNENTOL KEZDVE A LENTI KOMMENTEK SZAROK!!! A KULSO IVEN MAS "FORGASIRANY" SZERINT KELL KORBEMENNI EZERT MEG
        # VANNAK MASITVA A dirs-benAZ IRANYOK SORRENDJE A __get_dist_in-hez kepest!!!
        dirs = [BAL, LE, JOBB, FEL]
        direction_idx = 0
        point = start_point
        if rajz:
            self.draw_track()
        while True:
            dist += 1 # a távolságot növeli 1-gyel
            bal_ford = dirs[(direction_idx + 1) % 4] # a balra lévő pixel eléréséhez
            jobb_ford = dirs[(direction_idx - 1) % 4] # a jobbra lévő pixel eléréséhez
            if self.trk[point[1] + bal_ford[1], point[0] + bal_ford[0]] == self.col_out: # ha a tőlünk balra lévő pixel piros
                direction_idx = (direction_idx + 1) % 4 # akkor elfordulunk balra
                point = point + bal_ford
            elif self.trk[point[1] + dirs[direction_idx][1], point[0] + dirs[direction_idx][0]] == self.col_out: # ha az előttünk lévő pixel piros
                point = point + dirs[direction_idx] # akkor arra megyünk tovább
            else:
                direction_idx = (direction_idx - 1) % 4 # különben jobbra fordulunk
                point = point + jobb_ford

            dist_dict_out[tuple(point)] = dist # a pontot belerakjuk a dictionarybe

            if rajz:
                if use_matplotlib:
                    plt.plot([point[0]], [point[1]], 'yo')

            if np.array_equal(point, start_point): # ha visszaértünk az elejére, akkor leállunk
                break
        if rajz:
            if use_matplotlib:
                plt.draw()

        return dist_dict_out
