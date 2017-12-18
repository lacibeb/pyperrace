"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""

"""
Megpróbálva átalakítani Cart-Pole-ra
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #a kirajzoláshoz kell, de lassú szar
#import gym
from environment import PaperRaceEnv
from gym import wrappers
import tflearn
import argparse
import pprint as pp
import random as rnd

from replay_buffer import ReplayBuffer


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound

        scaled_out = tf.multiply(out, self.action_bound)
        # scaled_out = np.sign(out)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    #TODO: Lassú kirajzolást megcsinálni, ez most csak temp
    draws = 1 #nem minden epizodot fogunk kirajzolni, mert lassú. Lásd később

    #addig ahány epizódot akarunk:
    for i in range(int(args['max_episodes'])):

        env.reset() #ezt nem ertem pontosan mit csinal, de jobb ha itt van :)
        v = np.array([1, 0])  #az elején a sebesség jobbra 1 - talán jobb lenne ha a reset része lenne
        pos = np.array(env.starting_pos)  #sebesség mellé a kezdeti poz. is kell. Ez a kezdőpozíció beállítása
        ep_reward = 0
        ep_ave_max_q = 0
        color = (1 , 0, 0) #kornyezet kirajzolasahoz
        draw = False

        """
        Exploration: bizonyos valoszinuseggel beiktat egy total veletlen lepest. Egyreszt lehet olyan epizod
        amiben csak ilyen lepesek vannak, raadasul ez a valoszinuseg a tanulasra szant epizidszam elejen magas a 
        vegen meg alacsony
        """

        # Ha mas nincs, ne veletlenszeruen lepkedjen
        random_episode = False

        # generalunk egy 0-1 kozotti szamot, aminel majd kell random szamnak kisebbnek kell lenni es akkor teljesul
        # egy feltetel
        # rand_chk = max(0, int(args['max_episodes']) - (i * 3)) / int(args['max_episodes'])
        # teszt jelleggel ha sose akarunk randomot
        rand_chk = 0.2
        # ellenorzeskepp kiirva:
        # print("randomhoz:", rand_chk)

        # Ha tehat egy random szam kisebb mint egy adott, akkor random lesz az epizod
        if rnd.uniform(0, 1) < rand_chk:
            random_episode = True


        #hanyadik epizod lepeseit jelenitjuk meg (nem mindet, mert a kirajzolas lassu)
        if i == draws:
            draw = True #kornyezet kirajzolasahoz
            draws = draws + int(args['max_episodes']) / 100

        #egy egy epizódon belül ennyi lépés van maximum:
        for j in range(int(args['max_episode_len'])):

            s = [v[0], v[1], pos[0], pos[1]] #az eredeti kodban s-be van gyujtve az ami a masikban pos és v

            # Hasonloan, epizodon belul is lesznek random lepesek. Ha maga az epizod nem random, akkoris, egy egy lepest
            # random csinalunk. Itt viszont a random lepesek gyakorisaga no ahogy egyre elore haladunk az epizodban, es
            # az hogy mennyire az pedig a epizodszammal no (TODO: ezt meg megcsinalni, most csak siman 0.5 a valoszinuseg)

            random_step = False

            if rnd.uniform(0, 1) < 0:
                random_step = True

            #Actionok:

            # az elso lepest mindenkepp elore tegyuk meg
            if j == 0:
                a = 0

            # Ha az adott felteltel teljesult korabban, es most egy random epizodban vagyunk, vagy nem random az epizod,
            # de a lepes random, na akkor randomot lepunk:
            elif random_episode or random_step:
                a = int(np.random.randint(-180, 180, size=1))
                print("Random action:", a)
            # ha semmifeltetel a fentiekbol nem teljesul, akkor meg a halo altal mondott lepest lepjuk
            else:
                a = int(actor.predict(np.reshape(s, (1, actor.s_dim))) + 0*actor_noise())
                print(a)

            gg_action = env.gg_action(a)  # action-höz tartozó vektor lekérése
            #általában ez a fenti két sor egymsor. csak nálunk most így van megírva a környezet, hogy így kell neki beadni az actiont

            #megnézzük mit mond a környezet az adott álapotban az adott action-ra:
            #s2, r, terminal, info = env.step(a)
            v_new, pos_new, reward, end, section_nr, curr_dist = env.step(gg_action, v, pos, draw, color)

            #megintcsak a kétfelől összemásolgatott küdok miatt, feleltessünkk meg egymásnak változókat:
            s2 = [v_new[0], v_new[1], pos_new[0], pos_new[1]]
            """
            # a "faja" reward szamitashoz definialunk egy atlag sebesseget: [pixel/lepes] (A picxel helet, egy lepes
            # pedig egy egyseg idot jelent)
            ref_spd = 15

            #Az adott helyre eljutni az atlag "sebesseggel" elvileg ennyi lepes (ido):
            ref_time = curr_dist / ref_spd

            # az alap reward elvileg az eltelt idot adja negativban. Tehat r = reward -
            """
            r = reward
            terminal = end

            #és akkor a megfeleltetett változókkal már lehet csinálni a replay memory-t:
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            #TODO: ez a kirajzolás lassú. Vagy kivenni, vagy magcsinálni máshogy (temp:nem mindig rajzolunk
            if draw:
                plt.pause(0.001)
                plt.draw()

            #a kovetkezo lepeshez uj s legyen egyenlo az aktualis es folytatjuk
            #s = s2
            v = v_new
            pos = pos_new

            ep_reward += r

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), i, (ep_ave_max_q / float(j))))

                break


def main(args):
    with tf.Session() as sess:

        #env = gym.make(args['env'])

        trk_col = np.array([99, 99, 99])  # pálya színe (szürke), a kornyezet inicializalasahoz kell

        sections = np.array([[350,  60, 350, 100],
                             [425, 105, 430, 95],
                             [500, 140, 530, 110],
                             [520, 160, 580, 150]])
        #                     [ 35, 200,  70, 200],
        #                     [250,  60, 250, 100]])

        #env = PaperRaceEnv('PALYA3.bmp', trk_col, 'GG1.bmp', start_line, random_init=False)
        env = PaperRaceEnv('PALYA3.bmp', trk_col, 'GG1.bmp', sections, random_init=False)

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        #env.seed(int(args['random_seed']))

        state_dim = 4 #[vx, vy, posx, posy]
        action_dim = 1 #szam (fok) ami azt jelenti hogy a gg diagramon melikiranyba gyorsulunk
        action_bound = 180 #0: egyenesen -180,180: fék, -90: jobbra kanyar
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        '''if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)'''

        train(sess, env, args, actor, critic, actor_noise)

        '''if args['use_gym_monitor']:
            env.monitor.close()'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.000001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.00001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.98)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=32)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Acrobot-v1')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1324)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=5000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=100)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)