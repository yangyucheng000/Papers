import argparse
import datetime
from train import train
import os
import utils
import mindspore.context as context
# from openi import c2net_multidataset_to_env as DatasetToEnv
# from openi import env_to_openi

def parse_args():

    parser = argparse.ArgumentParser(description='Learn with Divergence')
    
    # parser.add_argument('--grampus_code_file_name', default='pre_and_suf.py')

    # parser.add_argument('--multi_data_url',
    #                   help='使用单数据集或多数据集时，需要定义的参数',
    #                   default= '[{}]')
    # parser.add_argument('--pretrain_url', default='')
    # parser.add_argument('--model_url',
    #                     help='使用智算集群回传结果到启智，需要定义的参数',
    #                     default= '')

    # data args
    parser.add_argument('--dataset_name', type=str, default='dogs', metavar='DS', help="dataset name (default: mini, options: mini|cub|tiered|mini2cub|tiered2cub)")
    parser.add_argument('--way', type=int, default=5, metavar='WAY', help="number of classes per episode (default: 5)")
    parser.add_argument('--shot', type=int, default=5, metavar='SHOT', help="number of support examples per class (default: 5)")
    parser.add_argument('--query', type=int, default=10, metavar='QUERY', help="number of query examples per class (default: 10)")
    parser.add_argument('--train_episodes', type=int, default=400000, metavar='ETRAIN', help="number of train episodes (default: 400000)")
    parser.add_argument('--test_episodes', type=int, default=600, metavar='ETEST', help="number of test episodes (default: 600)")
    parser.add_argument('--episodes_per_epoch', type=int, default=10, metavar='NTEST', help="number of test episodes per epoch (default: 10)")

    # model args
    parser.add_argument('--model_name', type=str, default='diver', help="the name of the loss function (default:'diver', options: conv_diver|pure_diver)")
    parser.add_argument('--orain_dim', type=str, default=3, help="dimensionality of input images (default:'3')")
    parser.add_argument('--enc_channel', type=int, default=64, help="Dimensionality of channels of the hidden layers in encoder (default: 64)")
    parser.add_argument('--diver_ops', type=str, default='sub', help="the operation for Divergence (default:'sub')")
    parser.add_argument('--layer_num', type=int, default=4, help="the layer of model (default: 4)")

    # parser.add_argument('--model.repad', action='store_true', help="using repad or not (default: False)")

    # train args
    parser.add_argument('--istrain', default=True, action='store_true', help="ture means is train, false means is test (default: False)")
    parser.add_argument('--loss', type=str, default='cce', help="Loss Function for PDMN (default:'cce')")
    parser.add_argument('--optim_method', type=str, default='Adam', metavar='OPTIM', help="optimization method (default:Adam)")
    parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='LR', help="Learning Rate (default: 0.001)")
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD', help="Lambda for L2 Normalization (default: 0.0)")
    parser.add_argument('--decay_every', type=int, default=40000, metavar='LRDECAY', help="number of epochs after which to decay the learning rate")
    parser.add_argument('--patience', type=int, default=100, metavar='PATIENCE', help="number of epochs to wait before validation improvement (default: 10)")

    # test args
    parser.add_argument('--query_no', type=int, default=0, metavar='QN', help="query_no from 0 (default:0)")
    parser.add_argument('--support_no_same', type=int, default=0, metavar='SNS', help="support_no_same from 0 (default:0)")
    parser.add_argument('--support_no_diff', type=int, default=1, metavar='SND', help="support_no_diff from 0 (default:1)")
    parser.add_argument('--visual', action='store_true', help="if execute the visualization during test (default: False)")
    parser.add_argument('--gb', action='store_true', help="if execute the gb during test (default: False)")
    parser.add_argument('--same', action='store_true', help="extract the same or diff samples (default: False)")

    # log args
    default_fields = 'loss, acc'
    parser.add_argument('--fields', type=str, default=default_fields, metavar='FIELDS', help="fields to monitor training (default: {:s})".format(default_fields))
    default_exp_dir = 'results'
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))
    parser.add_argument('--st', type=str, default='2021-01-01_00-00-00', metavar='START_TIME', help="log dir of saved model (default: 2021-01-01_00-00-00)")
    parser.add_argument('--st1', type=str, default='2021-01-01_00-00-00', metavar='START_TIME', help="log dir of saved model (default: 2021-01-01_00-00-00)")
    parser.add_argument('--st2', type=str, default='2021-01-01_00-00-00', metavar='START_TIME', help="log dir of saved model (default: 2021-01-01_00-00-00)")
    
    args = vars(parser.parse_args())
    
    return args

def main():

    print('Project Start!')
    args = parse_args()
    utils.set_global_config(args)
    utils.change_global_config('project_dir', os.getcwd())
    utils.change_global_config('data_dir', os.path.join(os.getcwd(), 'data', 'StanfordDogs'))
    utils.change_global_config('model_dir', os.path.join(os.getcwd(), 'data', 'model', 'model.ckpt'))
    utils.change_global_config('train_dir', os.path.join(os.getcwd(), 'data', 'output')) #?
    cfg = utils.get_global_config()
    context.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
  # DatasetToEnv(args['multi_data_url'], data_dir)

    if cfg['istrain']==True:
     
        print("Training Started!")
      
        FORMAT = '%Y-%m-%d_%H-%M-%S'
        start_time = datetime.datetime.now().strftime(FORMAT)
        train()
    
    else:
        if cfg['gb']==True:
            print('Running GuidedBackpropagation!')
        #   run_gb(args)
        else:
            print("This is the Testing period!")
        #   test(args)

if __name__ == '__main__':
    main()