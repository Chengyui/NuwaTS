import argparse
import os
import torch

from exp.exp_imputation import Exp_Imputation
from exp.exp_imputation_patchtst import Exp_Imputation as Exp_Imputation_PatchTST
from exp.exp_forecasting import  Exp_Imputation as Exp_forecasting
import random
import numpy as np
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    pid = os.getpid()

    print(f"The PID of the current process is: {pid}")
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')


    # basic config
    parser.add_argument('--task_name', type=str, default='denoise',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, denoise]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # imputation task
    parser.add_argument('--test_mask_rate', type=float, default=0.8, help='test mask ratio')
    parser.add_argument('--max_iterations', type=int, default=10, help='max iterations')
    parser.add_argument('--max_optimization_iterations', type=int, default=10, help='max optimization iterations')
    parser.add_argument('--regularization_weight', type=float, default=0.05, help='regularization weight')


    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # patching
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--ln', type=int, default=0)
    parser.add_argument('--mlp', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0)
    parser.add_argument('--percent', type=int, default=5)


    #prefix tuning
    parser.add_argument('--prefix_tuning', action='store_true', help='', default=False)
    parser.add_argument('--prefix_tuningv2', action='store_true', help='', default=False)
    parser.add_argument('--continue_tuning', action='store_true', help='', default=False)
    parser.add_argument('--continue_tuningv2', action='store_true', help='', default=False)


    parser.add_argument('--frozen_lm', action='store_true', help='', default=False)
    parser.add_argument('--prefix_length',type=int, default=1)
    parser.add_argument('--train_all_lm', action='store_true', help='', default=False)
    parser.add_argument('--use_llama', action='store_true', help='', default=False)
    parser.add_argument('--use_bert', action='store_true', help='', default=False)
    parser.add_argument('--alignment', action='store_true', help='', default=False)



    #contrastive
    parser.add_argument('--con_weight', type=float, default=0.01, help='')
    parser.add_argument('--patch_con', action='store_true', help='', default=False)
    parser.add_argument('--temporal_con', action='store_true', help='', default=False)
    parser.add_argument('--flatten_con', action='store_true', help='', default=False)
    parser.add_argument('--best_con_num', type=int, default=128)
    # output learnable token
    parser.add_argument('--seq_token',type=int, default=0)
    parser.add_argument('--word_prompt', action='store_true', help='', default=False)
    parser.add_argument('--cov_prompt', action='store_true', help='', default=False)
    parser.add_argument('--output_token', action='store_true', help='', default=False)

    # test
    parser.add_argument('--test_all',action='store_true', help='', default=False)

    #forecasting
    parser.add_argument('--is_forecasting', action='store_true', help='', default=False)
    parser.add_argument('--auto_regressive', action='store_true', help='', default=False)

    parser.add_argument('--origin_missrate', type=float, default=0, help='')



    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    if args.model == 'PatchTST' or args.model == 'Autoformer' or args.model == 'DLinear' or args.model == 'iTransformer' or args.model == 'FEDformer':
        Exp = Exp_Imputation_PatchTST
    else:
        if args.is_forecasting:
            Exp = Exp_forecasting
        else:
            Exp= Exp_Imputation



    if args.is_training:
        for ii in range(1):

            setting = '{}_{}_{}'.format(
                args.model,
                args.data,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:

        if args.test_all:
            ii = 0
            data_path = ['ETTh1.csv','ETTh2.csv','ETTm1.csv','ETTm2.csv','electricity.csv','weather.csv','PEMS03.npz','PEMS04.npz','PEMS07.npz','PEMS08.npz']
            data_type = ['ETTh1','ETTh2','ETTm1','ETTm2','custom','custom','pems','pems','pems','pems']
            des = ['ETTh1','ETTh2','ETTm1','ETTm2','electricity','weather','pems03','pems04','pems07','pems08']
            for i in range(0,10):
                args.data_path = data_path[i]
                args.data = data_type[i]
                ddes = args.des+des[i]
                setting = '{}_{}_{}'.format(
                    args.model,
                    args.data,
                    ddes, ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
            torch.cuda.empty_cache()
        else:
            ii = 0

            setting = '{}_{}_{}'.format(
                args.model,
                args.data,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
