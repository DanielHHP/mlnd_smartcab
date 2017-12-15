import random
import math
import os
import itertools
import pandas as pd
import ast
import cPickle as pickle
from datetime import datetime
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from agent import exploration_factor_funcs, LearningAgent

def score():
    """ Calculate score based on reliability and safety, return with train count
        return (score, train trial)
    """
    data = pd.read_csv(os.path.join("logs", 'sim_improved-learning.csv'))
    if len(data) < 10:
        # too less data
        return (-1, -1)
    data['good_actions'] = data['actions'].apply(lambda x: ast.literal_eval(x)[0])
    testing_data = data[data['testing'] == True]
    train_trial = len(data[data['testing'] == False])

    reliability = testing_data['success'].sum() * 1.0 / len(testing_data)

    safety = 0.0
    
    good_ratio = testing_data['good_actions'].sum() * 1.0 / \
    (testing_data['initial_deadline'] - testing_data['final_deadline']).sum()

    if good_ratio == 1: # Perfect driving
        safety = 1.0
    else: # Imperfect driving
        if testing_data['actions'].apply(lambda x: ast.literal_eval(x)[4]).sum() > 0: # Major accident
            safety = 0
        elif testing_data['actions'].apply(lambda x: ast.literal_eval(x)[3]).sum() > 0: # Minor accident
            safety = 0.2
        elif testing_data['actions'].apply(lambda x: ast.literal_eval(x)[2]).sum() > 0: # Major violation
            safety = 0.4
        else: # Minor violation
            minor = testing_data['actions'].apply(lambda x: ast.literal_eval(x)[1]).sum()
            if minor >= len(testing_data)/2: # Minor violation in at least half of the trials
                safety = 0.6
            else:
                safety = 0.8
    if (reliability + safety) == 0:
        return (0, train_trial)
    return (2 * reliability * safety / (reliability + safety), train_trial)

def log(fp, content, flush=True):
    """Log content to fp and console
    """
    print datetime.now().strftime('%Y/%m/%d %H:%M:%S'), content
    print >> fp, datetime.now().strftime('%Y/%m/%d %H:%M:%S'), content
    if flush:
        fp.flush()

def run(alpha, epsilon_decay_alpha, tolerance, explore_factor_func):
    """ Run in text mode """

    env = Environment(verbose=False)
    
    agent = env.create_agent(LearningAgent, alpha=alpha, 
                             learning=True,
                             epsilon_decay_alpha=epsilon_decay_alpha,
                             explore_factor_func=explore_factor_func
                            )
    
    env.set_primary_agent(agent, enforce_deadline=True)

    sim = Simulator(env, display=False, update_delay=0.001, log_metrics=True, optimized=True)
    
    sim.run(tolerance=tolerance, n_test=20)

def optimize_run():
    """ Use grid search to find optimize params for
        tolerance, alpha, epsilon_decay_alpha, and explore_factor_func
        Will prune too small (<100) or too large (>1000) train trial combination

        Has incremental search feature.
        Touch file with name "stop" will gracefully stop the process
    """
    # suite 1 find tolerance, decay_alpha
    #param_map = {'tolerance': [.05, .001, .0001, .00001],
    #          'alpha': [.8, .5, .3],
    #          'epsilon_decay_alpha': [.1, .05, .03, .01],
    #          'explore_factor_func': [3]}
    # suite 2 find alpha
    param_map = {'tolerance': [.001],
              'alpha': [.9, .8, .6, .5, .4, .3],
              'epsilon_decay_alpha': [.01],
              'explore_factor_func': [3]}

    # test suite
    #param_map = {'tolerance': [.05, .2, .00001],
    #           'alpha': [.3],
    #           'explore_factor_func': [1],
    #           'epsilon_decay_alpha': [.9],
    #            }

    count = 0
    param_score = {}
    if os.path.isfile('param_score.pickle'):
        with open('param_score.pickle') as fp:
            param_score = pickle.load(fp)

    log_fp = open('optimize.log', 'a')

    keys = param_map.keys()
    
    # iter all combinations
    for params in itertools.product(*[param_map[k] for k in keys]):
        if os.path.isfile('stop'):
            log(log_fp, 'Stop gracefully')
            return
        param_info = ','.join(['{}:{}'.format(k, params[i]) for i, k in enumerate(keys)])

        if params not in param_score:
            log(log_fp, 'Try params: {}'.format(param_info))
            
            # make sure train trial will be between 100 and 1000
            epsilon_after_100_trial = exploration_factor_funcs[params[keys.index('explore_factor_func')]](
                                            params[keys.index('epsilon_decay_alpha')], 100)
            log(log_fp, 'epsilon_after_100_trial: {}'.format(epsilon_after_100_trial))

            epsilon_after_1000_trial = exploration_factor_funcs[params[keys.index('explore_factor_func')]](
                                            params[keys.index('epsilon_decay_alpha')], 1000)
            log(log_fp, 'epsilon_after_1000_trial: {}'.format(epsilon_after_1000_trial))

            if epsilon_after_100_trial < params[keys.index('tolerance')] \
                or epsilon_after_1000_trial >= params[keys.index('tolerance')]:
                log(log_fp, 'train trial will be too small or too large. Prune!')
                param_score[params] = -2
            else:
                param_dict = {}
                param_dict.update(zip(keys, params))

                run(**param_dict)

                cur_score = score()
                param_score[params] = cur_score[0]
                log(log_fp, 'score: {}, train trial: {}'.format(cur_score[0], cur_score[1]))
        else:
            log(log_fp, 'Continue running, params: {} skiped'.format(param_info))

        with open('param_score.pickle', 'wb') as fp:
            pickle.dump(param_score, fp)

        count += 1
    
    # find max score param
    max_param = None
    max_score = -1

    for p, s in param_score.iteritems():
        if s > max_score:
            max_score = s
            max_param = p

    log(log_fp, 'total param combination: {}'.format(count))
    log(log_fp, 'max param score: ' + ','.join(['{}:{}'.format(k, max_param[i]) for i, k in enumerate(keys)]))
    log(log_fp, 'max score: {}'.format(max_score))
    log_fp.close()


if __name__ == '__main__':
    optimize_run()