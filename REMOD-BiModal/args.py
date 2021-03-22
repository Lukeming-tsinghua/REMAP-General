import argparse


def boolean_string(s):
    if s not in {'True', 'False'}:
        raise ValueError("%s: Not a valid Boolean argument string" % s)
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument('--path', type=str, help="path of data")
    parser.add_argument('--pred',
                        type=boolean_string,
                        help="whether predict or not",
                        default=False)
    parser.add_argument('--result', type=str, help="path of result dir")
    parser.add_argument('--name', type=str, help="task name")
    parser.add_argument('--tokenizer', type=str, help="tokenizer config")
    parser.add_argument('--text_model', type=str, help="text model config")
    parser.add_argument('--graph_model', type=str, help="graph model config")
    parser.add_argument('--maxLength',
                        type=int,
                        help="max length of sentences in an entity pair")
    parser.add_argument('--trainBatchSize',
                        type=int,
                        help="batchsize for training set")
    parser.add_argument('--testBatchSize',
                        type=int,
                        help="batchsize for validation set")
    parser.add_argument('--weightDecay', type=float, help="l2 penalty")
    parser.add_argument('--epoch', type=int, help="training epoch")
    parser.add_argument('--feat_dim', type=int, help="feature dimension")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--warmup_rate',
                        type=float,
                        help="warmup rate",
                        default=0.1)
    parser.add_argument('--accumulate_step',
                        type=int,
                        help="accumulate_step",
                        default=1)
    parser.add_argument('--nworkers',
                        type=int,
                        help="number of workers for dataloaders")
    parser.add_argument('--distill', type=boolean_string, help="distillation")
    parser.add_argument('--pinMemory', type=boolean_string, help="pin memory")
    parser.add_argument('--cuda', type=int, help="cuda rank id")
    parser.add_argument('--do_train', default=False, type=boolean_string, help="do train or not")
    parser.add_argument('--do_eval', default=False, type=boolean_string, help="do eval or not")
    parser.add_argument('--do_predict', default=False, type=boolean_string, help="do prediction or not")
    parser.add_argument('--seed',
                        type=int,
                        help="torch manual seed",
                        default=0)
    args = parser.parse_args()
    return args


def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
