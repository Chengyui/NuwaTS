import numpy as np

from torch.utils.data import DataLoader,ConcatDataset
from data_provider.allinone_dataset import ForecastGrid as all_in_one_dataset
import time
import os


pid = os.getpid()

print(f"The PID of the current process is: {pid}")


def load_path_interval():
    paths = [
         "dataset/ETTh1.npy",
        "dataset/ETTm1.npy",
        "dataset/ETTh2.npy",
        "dataset/ETTm2.npy",
        "dataset/weather.npy",
        "dataset/ECL.npy",
        "dataset/PEMS03_fill.npy",
        "dataset/PEMS04_fill.npy",
        "dataset/PEMS07_fill.npy",
        "dataset/PEMS08_fill.npy",
    ]

    return paths

def load_dataset_dataloader(input_length=64,horizon=64,batch_size=4096,num_workers=1,channel_group=32,flag='train',origin_missrate=0):
    paths = load_path_interval()
    combined_dataset = []

    for i in range(len(paths)):

        data_load = np.load(paths[i], allow_pickle=True)
        # data_load = data_load.reshape(data_load.shape[0], -1)

        length = data_load.shape[0]
        cols_num = data_load.shape[1]
        num_sensors_per_set = cols_num // 3
        if flag == 'train':
            dataset = all_in_one_dataset(data_load[:, :num_sensors_per_set], window_size=input_length, horizon=horizon,
                                     normalize_method='z_score', norm_statistic=None, interval=1,origin_missrate=origin_missrate
                                    )
        elif flag == 'val':
            dataset = all_in_one_dataset(data_load[:,num_sensors_per_set:2*num_sensors_per_set], window_size=input_length, horizon=horizon,
                                   normalize_method='z_score', norm_statistic=None, interval=1,
                                   )
        elif flag == 'test':
            dataset = all_in_one_dataset(data_load[:,2*num_sensors_per_set:3 * num_sensors_per_set], window_size=input_length, horizon=horizon,
                                   normalize_method='z_score', norm_statistic=None, interval=1,
                                   )
        combined_dataset.append(dataset)

        print("{} samples added".format(len(dataset)))

    combined_dataset = ConcatDataset(combined_dataset)


    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, drop_last=False, shuffle=True,
                              num_workers=num_workers,persistent_workers=False,pin_memory=False)

    print("totally number of batch: {}".format(len(combined_dataloader)))

    return combined_dataset,combined_dataloader



if __name__ == '__main__':
    begin_time = time.time()
    _,train_loader = load_dataset_dataloader(input_length=96,horizon=96,batch_size=128,num_workers=8,channel_group=1,flag='test')
    print("time consuming:{}".format(time.time() - begin_time))
    begin_time = time.time()
    for i, (inputs, target, batch_x_mark, batch_y_mark) in enumerate(train_loader):

        if i%1000==0:
            print(inputs.shape, target.shape)
            print("batch {}, time consuming:{}".format(i,time.time()-begin_time))

            #break

    print("time consuming:{}".format(time.time() - begin_time))






