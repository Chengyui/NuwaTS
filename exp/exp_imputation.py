from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,mask_visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.scheduler.cosine_lr import CosineLRScheduler
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
def fill_missing_data_ON(data, mask):
    """
    Fill missing data in a batch of time series data using the last previous observation.

    Args:
        data (Tensor): Batch of time series data with shape (B, T, N)
        mask (Tensor): Mask matrix with shape (B, T, N) where 0 indicates missing data

    Returns:
        Tensor: Filled batch of time series data with shape (B, T, N)
    """
    # Create a tensor to store the cumulative sum of the mask
    cumsum_mask = torch.cumsum(mask, dim=1)

    # Create a tensor to store the last previous observation
    last_observation = data.clone()

    # Fill the missing values with the last previous observation
    last_observation[mask == 0] = 0
    last_observation = last_observation.cumsum(dim=1)
    last_observation[cumsum_mask == 0] = 0
    last_observation = last_observation / cumsum_mask.clamp(min=1)

    # Fill the missing values in the original data
    data[mask == 0] = last_observation[mask == 0]
    return data

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                random_mask_rate = torch.rand(1).item() * 0.8 + 0.1
                num_masked = int(T * random_mask_rate)
                shuffle_indices = torch.rand(B, T, N, device=self.device).argsort(1)
                mask_ind, unmask_ind = shuffle_indices[:, :num_masked, :], shuffle_indices[:, num_masked:, :]
                batch_ind = torch.arange(B, device=self.device).unsqueeze(-1).unsqueeze(-1)
                sensor_ind = torch.arange(N, device=self.device).unsqueeze(0).unsqueeze(0)
                mask[batch_ind, mask_ind, sensor_ind] = 0  # masked
                mask[batch_ind, unmask_ind, sensor_ind] = 1  # remained

                inp = batch_x.masked_fill(mask == 0, 0)

                outputs,_ = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -T:, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()
                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
                # if i==5000:
                #     break
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("train steps:", train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        if self.args.prefix_tuningv2 or self.args.prefix_tuning or self.args.continue_tuningv2 or self.args.continue_tuning:
            Path = 'your pretrained checkpoint'
            ckpt = torch.load(Path,map_location=self.device)
            self.model.load_state_dict(ckpt,strict=False)
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if 'prefix' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = CosineLRScheduler(
            model_optim,
            t_initial=30,
            t_in_epochs=True  # update lr by_epoch(True) steps(False)
        )

        total_params = sum(p.numel() for p in self.model.parameters())

        print(f'{total_params:,} total parameters.')

        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                print(name)
        if self.args.origin_missrate > 0:
            print("Training With dataset origin missrate:{}".format(self.args.origin_missrate))

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape

                mask = torch.rand((B, T, N)).to(self.device)
                random_mask_rate = torch.rand(1).item() * 0.8 + 0.1
                num_masked = int(T *random_mask_rate)
                shuffle_indices = torch.rand(B,T,N,device=self.device).argsort(1)
                mask_ind, unmask_ind = shuffle_indices[:, :num_masked,:], shuffle_indices[:, num_masked:,:]
                batch_ind = torch.arange(B, device=self.device).unsqueeze(-1).unsqueeze(-1)
                sensor_ind = torch.arange(N, device=self.device).unsqueeze(0).unsqueeze(0)
                mask[batch_ind,mask_ind,sensor_ind] = 0  # masked
                mask[batch_ind,unmask_ind,sensor_ind] = 1  # remained

                inp = batch_x.masked_fill(mask == 0, 0)
                outputs,representation = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -T:, f_dim:]
                if self.args.patch_con or self.args.temporal_con or self.args.flatten_con:
                    con_loss,con_output = self.contrastive_loss(batch_x, batch_x_mark,representation)
                    loss = criterion(outputs, batch_x) + criterion(con_output,batch_x) + con_loss * self.args.con_weight
                else:
                    con_loss = torch.tensor(0).float().to(self.device)
                    loss = criterion(outputs, batch_x)
                # loss = criterion(outputs[mask == 0], batch_x[mask == 0]) + con_loss * self.args.con_weight
                train_loss.append(loss.item())

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} con loss : {3:.7f}".format(i + 1, epoch + 1, loss.item(),con_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
            scheduler.step(epoch)
            print("Adam lr epoch:{} lr:{}".format(epoch, model_optim.param_groups[0]['lr']))


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # adjust_learning_rate(model_optim, epoch + 1, self.args)



        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path,map_location=self.device))


        return self.model

    def test(self, setting, test=0,mask_rate=0.8):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        mse_list,mae_list = [],[]
        for mask_rate in range(1,10):
            mask_rate = mask_rate/10

            preds = []
            trues = []
            masks = []
            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    # random mask
                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    # random_mask_rate = self.args.test_mask_rate
                    random_mask_rate = mask_rate
                    num_masked = int(T * random_mask_rate)
                    shuffle_indices = torch.rand(B, T, N, device=self.device).argsort(1)
                    mask_ind, unmask_ind = shuffle_indices[:, :num_masked, :], shuffle_indices[:, num_masked:, :]
                    batch_ind = torch.arange(B, device=self.device).unsqueeze(-1).unsqueeze(-1)
                    sensor_ind = torch.arange(N, device=self.device).unsqueeze(0).unsqueeze(0)
                    mask[batch_ind, mask_ind, sensor_ind] = 0  # masked
                    mask[batch_ind, unmask_ind, sensor_ind] = 1  # remained

                    inp = batch_x.masked_fill(mask == 0, 0)

                    # imputation
                    outputs,_ = self.model(inp, batch_x_mark, None, None, mask)
                    # outputs = fill_missing_data_ON(inp, mask)

                    # eval
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -T:, f_dim:]
                    outputs = outputs.detach().cpu().numpy()
                    pred = outputs
                    true = batch_x.detach().cpu().numpy()
                    mask = mask.detach().cpu().numpy()
                    mae, mse, rmse, mape, mspe = metric(pred[mask == 0], true[mask == 0])
                    preds.append(pred)
                    trues.append(true)
                    masks.append(mask)
                    if i % 200 == 0:
                        filled = true[0, :, -1].copy()
                        filled_pred = filled * mask[0, :, -1] + \
                                 pred[0, :, -1] * (1 - mask[0, :, -1])
                        visual(true[0, :, -1], filled_pred, os.path.join(folder_path, str(i) + 'rate{}.pdf'.format(mask_rate)),mask_rate = mask_rate)
                        visual(true[0, :, -1], pred[0,:,-1],
                               os.path.join(folder_path, str(i) + 'origin_rate{}.pdf'.format(mask_rate)), mask_rate=mask_rate)


            preds = np.concatenate(preds, 0)
            trues = np.concatenate(trues, 0)
            masks = np.concatenate(masks, 0)
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result_imputation.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            mse_list.append(mse)
            mae_list.append(mae)
        print("mse list",mse_list)
        print("mae list",mae_list)
        print('mean mse',sum(mse_list)/len(mse_list))
        print("mean mae", sum(mae_list)/len(mae_list))
        return

    def contrastive_loss(self, batch_x, batch_x_mark,representation):
        B, T, N = batch_x.shape

        mask1 = torch.rand((B, T, N)).to(self.device)
        random_mask_rate = torch.rand(1).item() * 0.8 + 0.1
        num_masked = int(T * random_mask_rate)
        shuffle_indices = torch.rand(B, T, N, device=self.device).argsort(1)
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked, :], shuffle_indices[:, num_masked:, :]
        batch_ind = torch.arange(B, device=self.device).unsqueeze(-1).unsqueeze(-1)
        sensor_ind = torch.arange(N, device=self.device).unsqueeze(0).unsqueeze(0)
        mask1[batch_ind, mask_ind, sensor_ind] = 0  # masked
        mask1[batch_ind, unmask_ind, sensor_ind] = 1  # remained
        masked_inp1 = batch_x.masked_fill(mask1 == 0, 0)

        Impute,outputs1 = self.model(masked_inp1, batch_x_mark, None, None, mask1)

        outputs2 = representation


        con_loss = self.infoNCE(outputs1, outputs2, is_patch=self.args.patch_con,
                                is_temporal_patch=self.args.temporal_con, flatten_con=self.args.flatten_con)
        return con_loss,Impute


    def infoNCE(self,x1,x2,temperature=0.1,is_instance=False,is_patch=False,is_temporal_patch=False,flatten_con=False):
        """
        x1 shape:(B*c,patch_nun,dmodel)
        """
        loss = torch.tensor(0).float().to(x1.device)
        patch_num = x1.shape[1]
        batch_size = x1.shape[0]

        best_contrastive_num = self.args.best_con_num
        flatten_batch = batch_size*patch_num//best_contrastive_num
        res_flatten = (batch_size*patch_num)%best_contrastive_num

        if is_instance:
            x1_ = rearrange(x1,'b patch_num dmodel -> b (patch_num dmodel)')
            x2_ = rearrange(x2, 'b patch_num dmodel -> b (patch_num dmodel)')
            z_a = self.model.contrastive_instance_projector(x1_)
            z_pos = self.model.contrastive_instance_projector(x2_)
            Wz = torch.matmul(self.model.instance_W, z_pos.T)  # (z_dim,B)
            logits = torch.matmul(z_a, Wz)  # (B,B)
            logits = logits - torch.max(logits, 1)[0][:, None]

            labels = torch.arange(logits.shape[0]).long().to(self.device)
            instance_loss = self.model.cross_entropy_loss(logits, labels)
            loss+=instance_loss

        z_a = self.model.contrastive_patch_projector(x1)
        z_pos = self.model.contrastive_patch_projector(x2)
        if is_patch:

            z_a_p = rearrange(z_a,'b patch_num dmodel -> patch_num b dmodel')
            z_pos_p = rearrange(z_pos,'b patch_num dmodel -> patch_num b dmodel')
            patch_W = repeat(self.model.patch_W,'dim1 dim2 -> repeat dim1 dim2', repeat=patch_num)
            Wz = torch.bmm(patch_W,z_pos_p.transpose(1,2))
            logits = torch.bmm(z_a_p,Wz)
            logits = logits - torch.max(logits, 2)[0][:,:, None]
            labels = torch.arange(logits.shape[1]).long().to(self.device)
            labels = repeat(labels,'range -> repeat range',repeat=patch_num)
            patch_loss = self.model.cross_entropy_loss(logits, labels)
            loss+=patch_loss
        if is_temporal_patch:

            z_a_t = rearrange(z_a,'b patch_num dmodel -> b patch_num dmodel')
            z_pos_t = rearrange(z_pos,'b patch_num dmodel -> b patch_num dmodel')
            patch_W = repeat(self.model.patch_W, 'dim1 dim2 -> repeat dim1 dim2', repeat=batch_size)
            Wz = torch.bmm(patch_W, z_pos_t.transpose(1, 2))
            logits = torch.bmm(z_a_t, Wz)
            logits = logits - torch.max(logits, 2)[0][:, :, None]
            labels = torch.arange(logits.shape[1]).long().to(self.device)
            labels = repeat(labels, 'range -> repeat range', repeat=batch_size)
            patch_loss = self.model.cross_entropy_loss(logits, labels)
            loss += patch_loss
        if flatten_con:

            z_a_f = rearrange(z_a, 'b patch_num dmodel -> (b patch_num) dmodel')
            z_pos_f = rearrange(z_pos, 'b patch_num dmodel -> (b patch_num) dmodel')
            z_a_f = z_a_f[res_flatten:]
            z_pos_f = z_pos_f[res_flatten:]
            z_a_f = rearrange(z_a_f, '(b con_num) dmodel -> b con_num dmodel',con_num=best_contrastive_num)
            z_pos_f = rearrange(z_pos_f, '(b con_num) dmodel -> b con_num dmodel',con_num=best_contrastive_num)
            patch_W = repeat(self.model.patch_W, 'dim1 dim2 -> repeat dim1 dim2', repeat=flatten_batch)
            Wz = torch.bmm(patch_W, z_pos_f.transpose(1, 2))
            logits = torch.bmm(z_a_f, Wz)
            logits = logits - torch.max(logits, 2)[0][:, :, None]
            labels = torch.arange(logits.shape[1]).long().to(self.device)
            labels = repeat(labels, 'range -> repeat range', repeat=flatten_batch)
            patch_loss = self.model.cross_entropy_loss(logits, labels)
            loss += patch_loss
        return loss
