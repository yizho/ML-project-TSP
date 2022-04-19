import os
import sys
import argparse
import pathlib
import ecole
import numpy as np
import parameters
import skopt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-g',
        '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )

    parser.add_argument(
        '-n',
        '--num',
        help='tsp size train on. -1 if mixed training',
        type=int,
        default=15,
    )

    parser.add_argument(
        '-l',
        '--load_n',
        help=
        'load model from file (specify by number of tsp size trained on). -1 if mixed trained model, 0 to train from scratch',
        type=int,
        default=15,
    )

    args = parser.parse_args()

    if int(args.load_n) == -1:
        load_model = f'model/imitation/mixed/train_params.pkl'
    else:
        load_model = f'model/imitation/{args.load_n}n/train_params.pkl'

    ### HYPER PARAMETERS ###
    max_epochs = 100000
    optim_n_burnins = 100
    lr = 1e-4
    seed = parameters.seed + 4
    tsp_size = int(args.num)

    if tsp_size == -1:
        if int(args.load_n) == -1:
            running_dir = f"model/reinforce/mixed-mixed"
        elif int(args.load_n) == 0:
            running_dir = f"model/reinforce/{tsp_size}n"
        else:
            running_dir = f"model/reinforce/{args.load_n}-mixed"
    else:
        if int(args.load_n) == -1:
            running_dir = f"model/reinforce/mixed-{tsp_size}n"
        elif int(args.load_n) == 0:
            running_dir = f"model/reinforce/{tsp_size}n"
        else:
            running_dir = f"model/reinforce/{args.load_n}-{tsp_size}n"

    os.makedirs(running_dir, exist_ok=True)

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"

    env = ecole.environment.Branching(
        reward_function=-1.5 * ecole.reward.LpIterations()**2,
        observation_function=ecole.observation.NodeBipartite(),
    )

    import torch
    from utilities import log, Scheduler
    sys.path.insert(0, os.path.abspath(f'model'))
    from model import GNNPolicy

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    ### LOG ###
    import datetime
    timestampstr = datetime.datetime.now().strftime('%H_%M_%d_%m')
    logfile = os.path.join(running_dir, f'train_log_{timestampstr}.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"lr: {lr}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {seed}", logfile)

    env = ecole.environment.Configuring(
        # set up a few SCIP parameters
        scip_params={
            "branching/scorefunc": "s",  # sum score function
            "branching/vanillafullstrong/priority": 666666,  # use vanillafullstrong (highest priority)
            "presolving/maxrounds": 0,  # deactivate presolving
        },
        # pure bandit, no observation
        observation_function=None,

        # minimize the total number of nodes
        reward_function=-ecole.reward.LpIterations(),

        # collect additional metrics for information purposes
        information_function={
            "nnodes": ecole.reward.NNodes().cumsum(),
            "lpiters": ecole.reward.LpIterations().cumsum(),
            "time": ecole.reward.SolvingTime().cumsum(),
        },
    )

    policy = GNNPolicy().to(device)
    # set up the optimizer
    optimizer = skopt.Optimizer(
        dimensions=[(0.0, 1.0)],
        base_estimator="GP",
        n_initial_points=optim_n_burnins,
        random_state=rng,
        acq_func="PI",
        acq_optimizer="sampling",
        acq_optimizer_kwargs={"n_points": 10},
    )

    assert max_epochs > optim_n_burnins

    train_instances = [
        str(file)
        for file in (pathlib.Path(f'data/tsp{tsp_size}/instances') / f'train_{tsp_size}n').glob('instance_*.lp')
    ]
    valid_instances = [
        str(file)
        for file in (pathlib.Path(f'data/tsp{tsp_size}/instances') / f'valid_{tsp_size}n').glob('instance_*.lp')
    ]

    dataset_size = len(train_instances)
    avg_lp_iters = 0
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        instance = rng.choice(train_instances, size=1, replace=True)[0]
        env.reset(instance)

        # get the next action from the optimizer
        x = optimizer.ask()
        action = {"branching/scorefac": x[0]}

        # apply the action and collect the reward
        observation, action_set, reward, done, info = env.step(action)

        # update the optimizer
        optimizer.tell(x, -reward)  # minimize the negated reward (eq. maximize the reward)

        log(
            f'TRAIN LOSS: {-reward:0.3f} ' + f'LP Iter: {info["lpiters"]} ' + f'time: {info["time"]:0.3f} ' +
            f'nnodes: {info["nnodes"]}', logfile)

        # TEST
        instance = rng.choice(train_instances, size=1, replace=True)[0]
        env.reset(instance)

        # get the next action from the optimizer
        x = optimizer.ask()
        action = {"branching/scorefac": x[0]}

        # apply the action and collect the reward
        _, _, reward, _, info = env.step(action)
        log(
            f'VALID LOSS: {-reward:0.3f} ' + f'LP Iter: {info["lpiters"]} ' + f'time: {info["time"]:0.3f} ' +
            f'nnodes: {info["nnodes"]}', logfile)

    torch.save(policy.state_dict(), pathlib.Path(running_dir) / f'train_params_{timestampstr}.pkl')
    policy.load_state_dict(torch.load(pathlib.Path(running_dir) / f'train_params_{timestampstr}.pkl'))
    instance = rng.choice(train_instances, size=1, replace=True)[0]
    env.reset(instance)
    x = optimizer.ask()
    _, _, reward, _, info = env.step({"branching/scorefac": x[0]})
    log(
        f'VALID LOSS: {-reward:0.3f} ' + f'LP Iter: {info["lpiters"]} ' + f'time: {info["time"]:0.3f} ' +
        f'nnodes: {info["nnodes"]}', logfile)
