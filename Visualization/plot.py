import numpy as np
import torch
import matplotlib.pyplot as plt
from math import sqrt

def draw_missingrate():
    root_path = ""
    GPT4TS_09 = np.load(root_path+"GPT4TS09.npy")
    GPT4TS_05 = np.load(root_path+"GPT4TS05.npy")
    GPT4TS_01 = np.load(root_path+"GPT4TS01.npy")
    GPT4TS_seg = np.load(root_path+"GPT4TSseg.npy")
    NuwaTS_09 = np.load(root_path+"NuwaTS09.npy")
    NuwaTS_05 = np.load(root_path+"NuwaTS05.npy")
    NuwaTS_01 = np.load(root_path+"NuwaTS01.npy")
    NuwaTS_seg = np.load(root_path + "NuwaTSseg.npy")
    BRITS_09 = np.load(root_path+"BRITS09.npy")
    BRITS_05 = np.load(root_path+"BRITS05.npy")
    BRITS_01 = np.load(root_path+"BRITS01.npy")
    BRITS_seg = np.load(root_path + "BRITSseg.npy")
    PatchTST_09 = np.load(root_path+"PatchTST09.npy")
    PatchTST_05 = np.load(root_path+"PatchTST05.npy")
    PatchTST_01 = np.load(root_path+"PatchTST01.npy")
    PatchTST_seg = np.load(root_path + "PatchTSTseg.npy")
    TimesNet_09 = np.load(root_path+"TimesNet09.npy")
    TimesNet_05 = np.load(root_path+"TimesNet05.npy")
    TimesNet_01 = np.load(root_path+"TimesNet01.npy")
    TimesNet_seg = np.load(root_path + "TimesNetseg.npy")
    mask09 = np.load(root_path+"mask09.npy")
    mask05 = np.load(root_path+"mask05.npy")
    mask01 = np.load(root_path+"mask01.npy")
    mask_seg = np.load(root_path + "mask_seg.npy")
    batch_x = np.load(root_path+"batch_x.npy")


    batch_id = 0
    t_id = 75 # 104 101 97 90 83 80


    method_list = ["NuwaTS","PatchTST","GPT4TS","TimesNet","BRITS"]
    pred_list = [[NuwaTS_01,PatchTST_01,GPT4TS_01,TimesNet_01,BRITS_01],
            [NuwaTS_05,PatchTST_05,GPT4TS_05,TimesNet_05,BRITS_05],
            [NuwaTS_09,PatchTST_09,GPT4TS_09,TimesNet_09,BRITS_09],
                 [NuwaTS_seg,PatchTST_seg,GPT4TS_seg,TimesNet_seg,BRITS_seg]]
    mask_list = [mask01,mask05,mask09,mask_seg]
    mask_ratio = ["Missing Rate: 0.1","Missing Rate: 0.5","Missing Rate: 0.9","Two patches missed"]
    true = batch_x
    # for t_id in range(107):
    plt.figure(figsize=(72, 35))
    for maskid in range(4):
        pred = pred_list[maskid]
        mask = mask_list[maskid][batch_id,:,t_id]
        ax = plt.subplot(4, 6, 6 * maskid+1)
        start_index = None
        end_index = None
        masked_start_index = None
        masked_end_index = None
        index = 96
        time_serie = batch_x[batch_id,:,t_id]
        time_serie = np.append(time_serie,0)
        if mask[-1] == 0:
            mask = np.append(mask, 1)
        else:
            mask = np.append(mask, 0)
        for i, value in enumerate(time_serie):
            # if i ==80:
            #     print("hello")
            if mask[i] != 0 and start_index is None:
                start_index = i
            # 如果当前值不为零，并且开始绘制的索引已经设置，那么绘制折线
            if start_index is not None and mask[i] == 0:
                end_index = i
                if start_index>0:
                    start_index-=1

                plt.plot(range(start_index,end_index),time_serie[start_index:end_index],linewidth=5,color='black',linestyle='-')
                start_index = None
                end_index = None

            if mask[i] != 1 and masked_start_index is None:
                masked_start_index = i
            if masked_start_index is not None and mask[i] == 1:
                masked_end_index = i
                if masked_start_index>0:
                    masked_start_index-=1
                plt.plot(range(masked_start_index,masked_end_index),time_serie[masked_start_index:masked_end_index],linewidth=5,color='black',linestyle=':')
                masked_start_index = None
                masked_end_index = None
        plt.title(mask_ratio[maskid], fontsize=43, fontweight='bold')

        for i in range(len(method_list)):


            true1 = true[batch_id,:,t_id]
            pred1 = pred[i][batch_id].transpose(1, 0)[t_id]

            ax = plt.subplot(4, 6, 6*maskid+i + 2)

            ax.tick_params(axis='both', labelsize=32)
            if str(method_list[i]) == 'NuwaTS':
                plt.title(str(method_list[i]), fontsize=43, fontweight='bold')
            else:
                plt.title(str(method_list[i]), fontsize=43)

            plt.plot(true1[:], linewidth=3, label="GroundTruth")
            plt.plot(pred1[:], linewidth=4.3,  label="Prediction")

            plt.legend(loc='upper left', fontsize=24)


    plt.subplots_adjust(hspace=0.4)
    plt.savefig(root_path + "{}_{}test.svg".format(batch_id, t_id))
    # plt.show()
    plt.close()
    plt.clf()

if __name__ == '__main__':
    draw_missingrate()

