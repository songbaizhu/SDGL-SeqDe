import argparse
import os
import sys

sys.path.append(os.path.abspath(__file__+'/..'))

def str2bool(str):
    return True if str.lower() == 'true' else False

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data_path', type=str, default=r'D:\文档\PycharmProjects\time_series\raw_data\exchange_rate\exchange_rate.txt', help='dataset')
# parser.add_argument('--data_path', type=str, default=r'D:\文档\PycharmProjects\time_series\raw_data\traffic\traffic.txt', help='dataset')

parser.add_argument('--gcn_bool', type=str2bool, default='True', help='whether to add graph convolution layer')
parser.add_argument('--short_bool', type=str2bool, default='True', help='是否使用序列分解')
parser.add_argument('--addaptadj', type=int, default=1, help='whether add adaptive adj')

parser.add_argument('--seq_length', type=int, default=168, help='')
# parser.add_argument('--nhid', type=int, default=16, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=8, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # 0.001
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay rate')  # 0.0001
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./model/pems4', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--log_file', type=str, default='./log/pems4_log', help='log file')

parser.add_argument('--normalize', type=int, default=2,)

parser.add_argument('--layers', type=int, default=3, help='number of layers')

parser.add_argument('--column_wise', type=str2bool, default='False')
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--lag', type=int, default=168, help='input time windows length')
parser.add_argument('--horizon', type=int, default=3, help='')
parser.add_argument('--output_len', type=int, default=3)
parser.add_argument('--dilation_exponential_', type=int, default=1)

parser.add_argument('--MLP_layer', type=int, default=3) # 3
parser.add_argument('--MLP_dim', type=int, default=16) # 16 32
parser.add_argument('--if_node', type=str2bool, default='True')
parser.add_argument('--if_T_i_D', type=str2bool, default='True')
parser.add_argument('--if_D_i_W', type=str2bool, default='True')

parser.add_argument('--s_period', type=int, default=24)
parser.add_argument('--b_period', type=int, default=7)
parser.add_argument('--time_norm', type=str2bool, default='True')
parser.add_argument('--MLP_indim', type=int, default=3, help='inputs dimension of MLP')

parser.add_argument('--steps_per_day', type=int, default=24)

parser.add_argument('--embed_dim', type=int, default=10, help='node dim')

parser.add_argument('--add_time_in_day', type=int, default=1)
parser.add_argument('--add_day_in_week', type=int, default=1)

parser.add_argument('--use_RevIN', type=int, default=1, help='')


parser.add_argument('--smooth_depth', type=int, default=1, help='短期分量的平滑次数')
parser.add_argument('--patch_len', type=int, default=12, help='短期分量的平滑次数')
parser.add_argument('--residual_channels', type=int, default=32, help='短期分量的平滑次数')
parser.add_argument('--skip_channels', type=int, default=128, help='短期分量的平滑次数')
parser.add_argument('--end_channels', type=int, default=256, help='短期分量的平滑次数')
parser.add_argument('--accumulation_steps', type=int, default=1, help='累计梯度的次数')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]

import time
from util import *
from engine import trainer
from torch import nn
from util_series import generate_metric, DataLoaderS_stamp

log = open(args.log_file, 'w')


def log_string(string, log=log):
    log.write(string + '\n')
    log.flush()
    print(string)


def main():
    # set seed
    args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    is_train = True # False #
    # load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DataLoaderS_stamp(args.data_path, train=0.6, valid=0.2, device=device,
                       horizon=args.horizon, window=args.lag, normalize=args.normalize, args=args)

    print('loda dataset done')

    log_string(str(args))

    engine = trainer(device=device,data=data, distribute_data=None, args=args)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    if is_train:
        for i in range(1, args.epochs + 1):
            t1 = time.time()
            train_total_loss = 0
            n_samples = 0
            for iter, (x, y) in enumerate(data.train_loader):
                trainx = x[..., :1]
                trainx = trainx.transpose(1, 3)
                trainy = y
                metrics = engine.train(trainx, trainy, data=data, pred_time_embed=x, iter=iter)
                train_total_loss += metrics
                n_samples += (y.size(0) * data.m)
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, {:.4f} :Train Loss'
                    log_string(log.format(iter, metrics / (y.size(0) * data.m)))

                # break

            t2 = time.time()
            train_time.append(t2 - t1)
            # validation

            valid_total_loss = 0
            valid_total_loss_l1 = 0
            valid_n_samples = 0
            valid_output = None
            valid_label = None
            s1 = time.time()
            for iter, (x, y) in enumerate(data.valid_loader):
                trainx = x[..., :1]
                trainx = trainx.transpose(1, 3)
                trainy = y

                metrics = engine.eval(trainx, trainy, data, pred_time_embed=x)
                valid_total_loss += metrics[1]
                valid_total_loss_l1 += metrics[0]
                valid_n_samples += metrics[2]
                if valid_output is None:
                    valid_output = metrics[3]
                    valid_label = y
                else:
                    valid_output = torch.cat((valid_output, metrics[3]))
                    valid_label = torch.cat((valid_label, y))

            valid_rse, valid_rae, valid_correlation = generate_metric(valid_total_loss, valid_total_loss_l1,
                                                    valid_n_samples, data, valid_output, valid_label)

            engine.scheduler.step(valid_rse)

            s2 = time.time()
            val_time.append(s2 - s1)
            mtrain_loss = train_total_loss / n_samples

            mvalid_rse = valid_rse
            mvalid_rae = valid_rae
            mvalid_corr = valid_correlation
            his_loss.append(valid_rse.item())

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, ' \
                  'Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            log_string(log.format(i, mtrain_loss,
                                  mvalid_rse, mvalid_rae, mvalid_corr, (t2 - t1)))

            torch.save(engine.model.state_dict(),
                       args.save + "_epoch_" + str(i) + ".pth")

        log_string("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        log_string("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        # testing
        realy = []
        for x, y in data.test_loader:
            scale = data.scale.expand(y.size(0), data.m)
            realy.append(y * scale)

        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(
            torch.load(args.save + "_epoch_" + str(bestid + 1) + ".pth"))
        engine.model.eval()
        outputs_r = []
        test_total_loss = 0
        test_total_loss_l1 = 0
        test_n_samples = 0
        test_predict = None
        test = None
        evaluatel2 = nn.MSELoss(size_average=False).to(device)
        evaluatel1 = nn.L1Loss(size_average=False).to(device)
        for iter, (x, y) in enumerate(data.test_loader):
            testx = x[..., :1]
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds, _, _, = engine.model(testx, pred_time_embed=x)
                preds = preds[:, -1, :, :].clone()
            preds = preds.squeeze(dim=-1)
            # preds = preds[:,-1,:].clone()
            scale = data.scale.expand(preds.size(0), data.m)
            preds = preds * scale
            outputs_r.append(preds)

            test_total_loss += evaluatel2(preds, y * scale).item()
            test_total_loss_l1 += evaluatel1(preds, y * scale).item()
            test_n_samples += (preds.size(0) * data.m)

            if test_predict is None:
                test_predict = preds
                test = y
            else:
                test_predict = torch.cat((test_predict, preds))
                test = torch.cat((test, y))

        rse, rae, correlation = generate_metric(test_total_loss, test_total_loss_l1,
                                                test_n_samples, data, test_predict, test)

        log_string("The valid loss on best model is {}".format(str(round(his_loss[bestid], 4))))
        log_string('seed is {}'.format(args.seed))

        log = 'Evaluate best model on test data, Test rse: {:.4f}, Test rae: {:.4f}, Test corr: {:.4f}'
        log_string(log.format(rse, rae, correlation))
        torch.save(engine.model.state_dict(),
                   args.save + "_exp" + str(args.expid) + "_best_" + str(args.seed) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
