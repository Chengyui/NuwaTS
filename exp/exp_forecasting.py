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
def fill_missing_data(data, mask):
    """
    Fill missing data in a batch of time series data using the last previous observation.

    Args:
        data (Tensor): Batch of time series data with shape (B, T, N)
        mask (Tensor): Mask matrix with shape (B, T, N) where 0 indicates missing data

    Returns:
        Tensor: Filled batch of time series data with shape (B, T, N)
    """
    # Get the indices of the missing data
    missing_idx = (mask == 0).nonzero()

    # Iterate over the missing indices and fill with the last previous observation
    for idx in missing_idx:
        b, t, n = idx
        if t == 0:
            data[b, t, n] = 0
        else:
            data[b, t, n] = data[b, t-1, n]

    return data
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
def fill_missing_values_mean(data, mask):
    """
    Fill missing values in a tensor using the mean of the unmasked values in each variable.

    Args:
        data (Tensor): Tensor with shape (B, T, N) containing the data
        mask (Tensor): Tensor with shape (B, T, N) containing the mask, where 0 indicates missing values

    Returns:
        Tensor: Filled tensor with shape (B, T, N)
    """
    # Compute the sum of unmasked values for each variable
    sum_unmasked = (data * mask).sum(dim=1)

    # Compute the count of unmasked values for each variable
    count_unmasked = mask.sum(dim=1)

    # Compute the mean of unmasked values for each variable
    mean_unmasked = sum_unmasked / count_unmasked.clamp(min=1)

    # Fill the missing values with the mean
    filled_data = data.clone()
    num_masked = data.shape[1] - mask.sum(dim=1)[0][0]
    filled_data[mask == 0] = mean_unmasked.repeat(1, int(num_masked), 1).view(-1)

    return filled_data

def fill_missing_medians(data, mask):
    """
    Fill missing values in a tensor using the median value of the unmasked values in each variable.

    Args:
        data (Tensor): Tensor with shape (B, T, N) containing the data
        mask (Tensor): Tensor with shape (B, T, N) containing the mask, where 0 indicates missing values

    Returns:
        Tensor: Filled tensor with shape (B, T, N)
    """
    # Get the unmasked values
    num_masked = int(data.shape[1] - mask.sum(dim=1)[0][0])
    B,T,N = data.shape
    unmasked_values = data[mask == 1].reshape(B,int(data.shape[1]-num_masked),N)

    # Compute the median of the unmasked values
    median_values, _ = torch.median(unmasked_values, dim=1)

    # Fill the missing values with the median
    filled_data = data.clone()

    filled_data[mask == 0] = median_values.repeat(1, int(num_masked), 1).view(-1)

    return filled_data


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
        trained_parameters = []
        for p in self.model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.Adam(trained_parameters, lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self,vali_data, vali_loader, criterion):

        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_x = batch_x[:,self.args.patch_size:,:]
                outputs = []
                if self.args.auto_regressive:
                    for auto_regressive_i in range(6):
                        # random mask
                        B, T, N = batch_x.shape

                        mask = torch.rand((B, T + self.args.patch_size, N)).to(self.device)

                        mask[:, -self.args.patch_size:, :] = 0  # masked
                        mask[:, :-self.args.patch_size, :] = 1  # remained
                        input_padding = torch.rand((B, self.args.patch_size, N)).to(self.device)

                        batch_x = torch.concatenate([batch_x, input_padding], dim=1)

                        inp = batch_x.masked_fill(mask == 0, 0)
                        output, representation = self.model(inp, batch_x_mark, None, None, mask)
                        batch_x = batch_x[:, self.args.patch_size:-self.args.patch_size, :]
                        output_padding = output[:, -self.args.patch_size:, :]
                        batch_x = torch.concatenate([batch_x, output_padding], dim=1)
                        outputs.append(output_padding)
                        loss_step = criterion(output_padding, batch_y[:, auto_regressive_i * self.args.patch_size:( auto_regressive_i + 1) * self.args.patch_size,
                                                              :])
                        loss += loss_step
                    outputs = torch.concatenate(outputs, dim=1)
                else:
                    # random mask
                    B, T, N = batch_x.shape

                    mask = torch.rand((B, T + self.args.pred_len, N)).to(self.device)
                    mask[:, -self.args.pred_len:, :] = 0  # masked
                    mask[:, :-self.args.pred_len, :] = 1  # remained
                    input_padding = torch.rand((B, self.args.pred_len, N)).to(self.device)

                    batch_x = torch.concatenate([batch_x, input_padding], dim=1)

                    inp = batch_x.masked_fill(mask == 0, 0)
                    output, representation = self.model(inp, batch_x_mark, None, None, mask)
                    # batch_x = batch_x[:, self.args.patch_size:-self.args.patch_size, :]
                    output_padding = output[:, -self.args.pred_len:, :]
                    outputs = output_padding

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                mask = mask.detach().cpu()
                # loss = criterion(pred[mask == 0], true[mask == 0])
                loss = criterion(pred, true)
                total_loss.append(loss)
                # if i==5000:
                #     break
        total_loss = np.average(total_loss)
        self.model.train()
        print(total_loss)
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
            keys_to_delete = [key for key in ckpt.keys() if 'out' in key or 'instance' in key]
            for key in keys_to_delete:
                del ckpt[key]
                print(f'Deleted layer: {key}')
            self.model.load_state_dict(ckpt,strict=False)
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if 'prefix' in name or 'miss' in name or 'ln' in name or 'wpe' in name or 'out' in name:
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = []
                output = torch.tensor(0)

                loss = 0
                if self.args.auto_regressive:
                    for auto_regressive_i in range(6):

                        B, T, N = batch_x.shape

                        mask = torch.rand((B, T+self.args.patch_size, N)).to(self.device)

                        mask[:,-self.args.patch_size:,:] = 0  # masked
                        mask[:,:-self.args.patch_size,:] = 1  # remained
                        input_padding = torch.rand((B,self.args.patch_size,N)).to(self.device)

                        batch_x = torch.concatenate([batch_x,input_padding],dim=1)

                        inp = batch_x.masked_fill(mask == 0, 0)
                        output,representation = self.model(inp, batch_x_mark, None, None, mask)
                        batch_x = batch_x[:,self.args.patch_size:-self.args.patch_size,:]
                        output_padding = output[:, -self.args.patch_size:,:]
                        batch_x = torch.concatenate([batch_x,output_padding],dim=1)
                        outputs.append(output_padding)
                        loss_step = criterion(output_padding, batch_y[:,auto_regressive_i*self.args.patch_size:(auto_regressive_i+1)*self.args.patch_size,:])
                        loss += loss_step
                    outputs = torch.concatenate(outputs, dim=1)
                else:
                    # random mask
                    B, T, N = batch_x.shape

                    mask = torch.rand((B, T + self.args.pred_len, N)).to(self.device)
                    mask[:, -self.args.pred_len:, :] = 0  # masked
                    mask[:, :-self.args.pred_len, :] = 1  # remained
                    input_padding = torch.rand((B, self.args.pred_len, N)).to(self.device)

                    batch_x = torch.concatenate([batch_x, input_padding], dim=1)

                    inp = batch_x.masked_fill(mask == 0, 0)
                    output, representation = self.model(inp, batch_x_mark, None, None, mask)
                    # batch_x = batch_x[:, self.args.patch_size:-self.args.patch_size, :]
                    output_padding = output[:, -self.args.pred_len:, :]

                    loss = criterion(output_padding, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} ".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()


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
            # scheduler.step(epoch)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path,map_location=self.device))


        return self.model

    def test(self, setting, test=0,mask_rate=0.8):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


        for mask_rate in range(1,2):
            mask_rate = mask_rate/10

            preds = []
            trues = []
            masks = []
            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            import matplotlib.pyplot as plt
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    outputs = []
                    if self.args.auto_regressive:
                        for auto_regressive_i in range(6):
                            # random mask
                            B, T, N = batch_x.shape

                            mask = torch.rand((B, T + self.args.pred_len, N)).to(self.device)

                            mask[:, -self.args.pred_len:, :] = 0  # masked
                            mask[:, :-self.args.pred_len, :] = 1  # remained
                            input_padding = torch.rand((B, self.args.pred_len, N)).to(self.device)

                            batch_x = torch.concatenate([batch_x, input_padding], dim=1)

                            inp = batch_x.masked_fill(mask == 0, 0)
                            output, representation = self.model(inp, batch_x_mark, None, None, mask)
                            batch_x = batch_x[:, self.args.patch_size:-self.args.pred_len, :]
                            output_padding = output[:, :self.args.patch_size, :]
                            batch_x = torch.concatenate([batch_x, output_padding], dim=1)
                            outputs.append(output_padding)
                        outputs = torch.concatenate(outputs, dim=1)
                    else:
                        # random mask
                        B, T, N = batch_x.shape

                        mask = torch.rand((B, T + self.args.pred_len, N)).to(self.device)
                        mask[:, -self.args.pred_len:, :] = 0  # masked
                        mask[:, :-self.args.pred_len, :] = 1  # remained
                        input_padding = torch.rand((B, self.args.pred_len, N)).to(self.device)

                        batch_x = torch.concatenate([batch_x, input_padding], dim=1)

                        inp = batch_x.masked_fill(mask == 0, 0)
                        output, representation = self.model(inp, batch_x_mark, None, None, mask)
                        # batch_x = batch_x[:, self.args.patch_size:-self.args.patch_size, :]
                        output_padding = output[:, -self.args.pred_len:, :]
                        outputs = output_padding

                    # outputs = torch.concatenate(outputs,dim=1)
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :self.args.seq_len, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :self.args.seq_len, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                preds = np.array(preds)
                trues = np.array(trues)
                print('test shape:', preds.shape, trues.shape)
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                print('test shape:', preds.shape, trues.shape)

                # result save
                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)



                mae, mse, rmse, mape, mspe = metric(preds, trues)
                print('mse:{}, mae:{}'.format(mse, mae))
                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('\n')
                f.write('\n')
                f.close()

                np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(folder_path + 'pred.npy', preds)
                np.save(folder_path + 'true.npy', trues)

                return