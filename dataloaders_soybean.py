# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict, namedtuple
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder


def load_all_data(path, predict_year):
    device = 'cuda:0'

    print("Start preprocessing")

    data = np.genfromtxt(path, delimiter=',')[1:, :]

    loc_path = "./data/Soybeans_Loc_ID.csv"

    df = pd.read_csv(loc_path)

    states = df.State.unique()

    state_id = {'illinois': 0, 'indiana': 1, 'iowa': 2, 'kansas': 3, 'minnesota': 4,
                'missouri': 5, 'nebraska': 6, 'north dakota': 7, 'south dakota': 8}

    d = np.zeros(shape=[data.shape[0], data.shape[1] + 1])

    for i in range(data.shape[0]):
        state_loc_id = state_id[str(df.iloc[df.index[df['ID_loc'] == int(data[i, 0])], 0].values[0])]
        d[i, :] = np.insert(data[i, :], 0, state_loc_id, axis=0)

    train_fed_loaders = []
    val_fed_loaders = []
    test_fed_loaders = []

    for s in range(9):

        locations = (df.loc[df['State'] == list(state_id.keys())[s]]).ID_loc.unique()

        idx_s = np.where(d[:, 0] == s)

        data_s = np.squeeze(d[idx_s, :], axis=0)

        data_train = data_s[data_s[:, 2] < predict_year]

        if predict_year == 1995:
            data_train = np.concatenate((data_train, data_s[data_s[:, 2] > predict_year]), axis=0)

        data_test = data_s[data_s[:, 2] == predict_year]

        # Train
        data_x = data_train[:, 4:]

        mn = np.mean(data_x, axis=0, keepdims=True)

        sd = np.std(data_x, axis=0, keepdims=True)

        data_train[:, 4:] = (data_train[:, 4:] - mn) / sd
        # data_train = data_train

        # Test
        data_test[:, 4:] = (data_test[:, 4:] - mn) / sd


        # data_test = data_test

        comb_data = np.vstack((data_train, data_test))

        comb_data = np.nan_to_num(comb_data)
        index_low_yield = comb_data[:, 3] < 2
        comb_data = comb_data[np.logical_not(index_low_yield)]

        # Split into year windows + avg predictions
        avg, dic, A, mean_last = averages(comb_data, predict_year)

        train_years = A[:-1, :]

        # sample_num = 10000
        sample_num = count_train(train_years, data_train, locations)
        num_features = 312 + 66 + 14 + 4

        out = np.zeros(shape=[sample_num, 5, num_features + 1])

        out_idx = 0

        for l in range(len(locations)):
            loc = locations[l]
            # print(loc)
            for j in range(len(train_years)):
                temp1 = np.where((data_train[:, 1] == loc) & (data_train[:, 2] == train_years[j][0]))[0].size
                temp2 = np.where((data_train[:, 1] == loc) & (data_train[:, 2] == train_years[j][1]))[0].size
                temp3 = np.where((data_train[:, 1] == loc) & (data_train[:, 2] == train_years[j][2]))[0].size
                temp4 = np.where((data_train[:, 1] == loc) & (data_train[:, 2] == train_years[j][3]))[0].size
                temp5 = np.where((data_train[:, 1] == loc) & (data_train[:, 2] == train_years[j][4]))[0].size
                check = temp1 + temp2 + temp3 + temp4 + temp5
                if check == 5:
                    for k, y in enumerate(train_years[j]):
                        X = dic[str(y)]
                        ym = avg[str(y)]
                        out[out_idx, k, :] = np.concatenate((X[(X[:, 1] == loc), :], np.array([[ym]])), axis=1)
                    out_idx = out_idx + 1

        '''
        for i in range(sample_num):

            r1 = np.squeeze(np.random.randint(train_years.shape[0], size=1))

            years = train_years[r1, :]

            r2 = np.random.randint(locations[0], locations[-1], size=1)
            # r2 = [60]
            # Check if farm exists for all the sampled years if not choose different farm
            counter = 0
            while counter < 5:
                for j, y in enumerate(years):
                    X = dic[str(y)]
                    farm_idx = np.where(X[:, 1] == r2[0])[0]
                    if farm_idx.size != 0:
                        counter += 1
                    else:
                        counter = 0
                        r2 = np.random.randint(locations[0], locations[-1], size=1)

            # print(f"years: {years}")
            # print(f"r2: {r2}")

            for j, y in enumerate(years):
                X = dic[str(y)]
                ym = avg[str(y)]

                farm_idx = np.where(X[:, 1] == r2[0])[0]

                out[i, j, :] = np.concatenate((X[farm_idx, :], np.array([[ym]])), axis=1)
        '''

        Data = namedtuple('Data',
                          ['w_t1', 'w_t2', 'w_t3', 'w_t4', 'w_t5', 'w_t6', 's_t1', 's_t2', 's_t3', 's_t4', 's_t5',
                           's_t6', 's_t7', 's_t8', 's_t9', 's_t10', 'p_t', 'y_bar', 'y_t', 'y_t2'])

        Ybar = out[:, :, -1].reshape(-1, 5, 1)

        Batch_X_e = out[:, :, 4:-1]

        Batch_Y = out[:, -1, 3]

        Batch_Y = Batch_Y.reshape((len(Batch_Y), 1))

        Batch_Y_2 = out[:, np.arange(0, 4), 3]

        # loc_ID,year,yield,W_1_1,W_1_2,W_1_3,W_1_4,W_1_5,W_1_6,W_1_7,W_1_8,W_1_9,W_1_10,W_1_11,W_1_12,W_1_13,W_1_14,W_1_15,W_1_16,W_1_17,W_1_18,W_1_19,W_1_20,W_1_21,W_1_22,W_1_23,W_1_24,W_1_25,W_1_26,W_1_27,W_1_28,W_1_29,W_1_30,W_1_31,W_1_32,W_1_33,W_1_34,W_1_35,W_1_36,W_1_37,W_1_38,W_1_39,W_1_40,W_1_41,W_1_42,W_1_43,W_1_44,W_1_45,W_1_46,W_1_47,W_1_48,W_1_49,W_1_50,W_1_51,W_1_52,W_2_1,W_2_2,W_2_3,W_2_4,W_2_5,W_2_6,W_2_7,W_2_8,W_2_9,W_2_10,W_2_11,W_2_12,W_2_13,W_2_14,W_2_15,W_2_16,W_2_17,W_2_18,W_2_19,W_2_20,W_2_21,W_2_22,W_2_23,W_2_24,W_2_25,W_2_26,W_2_27,W_2_28,W_2_29,W_2_30,W_2_31,W_2_32,W_2_33,W_2_34,W_2_35,W_2_36,W_2_37,W_2_38,W_2_39,W_2_40,W_2_41,W_2_42,W_2_43,W_2_44,W_2_45,W_2_46,W_2_47,W_2_48,W_2_49,W_2_50,W_2_51,W_2_52,W_3_1,W_3_2,W_3_3,W_3_4,W_3_5,W_3_6,W_3_7,W_3_8,W_3_9,W_3_10,W_3_11,W_3_12,W_3_13,W_3_14,W_3_15,W_3_16,W_3_17,W_3_18,W_3_19,W_3_20,W_3_21,W_3_22,W_3_23,W_3_24,W_3_25,W_3_26,W_3_27,W_3_28,W_3_29,W_3_30,W_3_31,W_3_32,W_3_33,W_3_34,W_3_35,W_3_36,W_3_37,W_3_38,W_3_39,W_3_40,W_3_41,W_3_42,W_3_43,W_3_44,W_3_45,W_3_46,W_3_47,W_3_48,W_3_49,W_3_50,W_3_51,W_3_52,W_4_1,W_4_2,W_4_3,W_4_4,W_4_5,W_4_6,W_4_7,W_4_8,W_4_9,W_4_10,W_4_11,W_4_12,W_4_13,W_4_14,W_4_15,W_4_16,W_4_17,W_4_18,W_4_19,W_4_20,W_4_21,W_4_22,W_4_23,W_4_24,W_4_25,W_4_26,W_4_27,W_4_28,W_4_29,W_4_30,W_4_31,W_4_32,W_4_33,W_4_34,W_4_35,W_4_36,W_4_37,W_4_38,W_4_39,W_4_40,W_4_41,W_4_42,W_4_43,W_4_44,W_4_45,W_4_46,W_4_47,W_4_48,W_4_49,W_4_50,W_4_51,W_4_52,W_5_1,W_5_2,W_5_3,W_5_4,W_5_5,W_5_6,W_5_7,W_5_8,W_5_9,W_5_10,W_5_11,W_5_12,W_5_13,W_5_14,W_5_15,W_5_16,W_5_17,W_5_18,W_5_19,W_5_20,W_5_21,W_5_22,W_5_23,W_5_24,W_5_25,W_5_26,W_5_27,W_5_28,W_5_29,W_5_30,W_5_31,W_5_32,W_5_33,W_5_34,W_5_35,W_5_36,W_5_37,W_5_38,W_5_39,W_5_40,W_5_41,W_5_42,W_5_43,W_5_44,W_5_45,W_5_46,W_5_47,W_5_48,W_5_49,W_5_50,W_5_51,W_5_52,W_6_1,W_6_2,W_6_3,W_6_4,W_6_5,W_6_6,W_6_7,W_6_8,W_6_9,W_6_10,W_6_11,W_6_12,W_6_13,W_6_14,W_6_15,W_6_16,W_6_17,W_6_18,W_6_19,W_6_20,W_6_21,W_6_22,W_6_23,W_6_24,W_6_25,W_6_26,W_6_27,W_6_28,W_6_29,W_6_30,W_6_31,W_6_32,W_6_33,W_6_34,W_6_35,W_6_36,W_6_37,W_6_38,W_6_39,W_6_40,W_6_41,W_6_42,W_6_43,W_6_44,W_6_45,W_6_46,W_6_47,W_6_48,W_6_49,W_6_50,W_6_51,W_6_52,bdod_mean_0-5cm,bdod_mean_5-15cm,bdod_mean_15-30cm,bdod_mean_30-60cm,bdod_mean_60-100cm,bdod_mean_100-200cm,cec_mean_0-5cm,cec_mean_5-15cm,cec_mean_15-30cm,cec_mean_30-60cm,cec_mean_60-100cm,cec_mean_100-200cm,cfvo_mean_0-5cm,cfvo_mean_5-15cm,cfvo_mean_15-30cm,cfvo_mean_30-60cm,cfvo_mean_60-100cm,cfvo_mean_100-200cm,clay_mean_0-5cm,clay_mean_5-15cm,clay_mean_15-30cm,clay_mean_30-60cm,clay_mean_60-100cm,clay_mean_100-200cm,nitrogen_mean_0-5cm,nitrogen_mean_5-15cm,nitrogen_mean_15-30cm,nitrogen_mean_30-60cm,nitrogen_mean_60-100cm,nitrogen_mean_100-200cm,ocd_mean_0-5cm,ocd_mean_5-15cm,ocd_mean_15-30cm,ocd_mean_30-60cm,ocd_mean_60-100cm,ocd_mean_100-200cm,ocs_mean_0-5cm,ocs_mean_5-15cm,ocs_mean_15-30cm,ocs_mean_30-60cm,ocs_mean_60-100cm,ocs_mean_100-200cm,phh2o_mean_0-5cm,phh2o_mean_5-15cm,phh2o_mean_15-30cm,phh2o_mean_30-60cm,phh2o_mean_60-100cm,phh2o_mean_100-200cm,sand_mean_0-5cm,sand_mean_5-15cm,sand_mean_15-30cm,sand_mean_30-60cm,sand_mean_60-100cm,sand_mean_100-200cm,silt_mean_0-5cm,silt_mean_5-15cm,silt_mean_15-30cm,silt_mean_30-60cm,silt_mean_60-100cm,silt_mean_100-200cm,soc_mean_0-5cm,soc_mean_5-15cm,soc_mean_15-30cm,soc_mean_30-60cm,soc_mean_60-100cm,soc_mean_100-200cm,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8,P_9,P_10,P_11,P_12,P_13,P_14

        train_data = Data(
            w_t1=torch.as_tensor(Batch_X_e[:, :, 0:52], device=device).float(),
            w_t2=torch.as_tensor(Batch_X_e[:, :, 52 * 1:2 * 52], device=device).float(),
            w_t3=torch.as_tensor(Batch_X_e[:, :, 52 * 2:3 * 52], device=device).float(),
            w_t4=torch.as_tensor(Batch_X_e[:, :, 52 * 3:4 * 52], device=device).float(),
            w_t5=torch.as_tensor(Batch_X_e[:, :, 52 * 4:5 * 52], device=device).float(),
            w_t6=torch.as_tensor(Batch_X_e[:, :, 52 * 5:52 * 6], device=device).float(),
            s_t1=torch.as_tensor(Batch_X_e[:, :, 312:318], device=device).float(),
            s_t2=torch.as_tensor(Batch_X_e[:, :, 318:324], device=device).float(),
            s_t3=torch.as_tensor(Batch_X_e[:, :, 324:330], device=device).float(),
            s_t4=torch.as_tensor(Batch_X_e[:, :, 330:336], device=device).float(),
            s_t5=torch.as_tensor(Batch_X_e[:, :, 336:342], device=device).float(),
            s_t6=torch.as_tensor(Batch_X_e[:, :, 342:348], device=device).float(),
            s_t7=torch.as_tensor(Batch_X_e[:, :, 348:354], device=device).float(),
            s_t8=torch.as_tensor(Batch_X_e[:, :, 354:360], device=device).float(),
            s_t9=torch.as_tensor(Batch_X_e[:, :, 366:372], device=device).float(),
            s_t10=torch.as_tensor(Batch_X_e[:, :, 372:378], device=device).float(),
            p_t=torch.as_tensor(Batch_X_e[:, :, 378:392], device=device).float(),
            y_bar=torch.as_tensor(Ybar, device=device).float(),
            y_t=torch.as_tensor(Batch_Y, device=device).float(),
            y_t2=torch.as_tensor(Batch_Y_2, device=device).float(),
        )

        # train_dataset, val_dataset = random_split(
        #     TensorDataset(train_data.w_t1, train_data.w_t2, train_data.w_t3, train_data.w_t4, train_data.w_t5,
        #                   train_data.w_t6,
        #                   train_data.s_t1, train_data.s_t2, train_data.s_t3, train_data.s_t4, train_data.s_t5,
        #                   train_data.s_t6, train_data.s_t7, train_data.s_t8, train_data.s_t9, train_data.s_t10,
        #                   train_data.p_t, train_data.y_bar, train_data.y_t, train_data.y_t2), (int(sample_num*0.95), sample_num-int(sample_num*0.95)))

        train_dataset = TensorDataset(train_data.w_t1, train_data.w_t2, train_data.w_t3, train_data.w_t4,
                                      train_data.w_t5,
                                      train_data.w_t6,
                                      train_data.s_t1, train_data.s_t2, train_data.s_t3, train_data.s_t4,
                                      train_data.s_t5,
                                      train_data.s_t6, train_data.s_t7, train_data.s_t8, train_data.s_t9,
                                      train_data.s_t10,
                                      train_data.p_t, train_data.y_bar, train_data.y_t, train_data.y_t2)
        train_fed_loaders.append(train_dataset)
        # val_fed_loaders.append(val_dataset)

        X_test = dic[str(predict_year)][:, 1:]
        years = []
        if predict_year == 2018:
            years = [2014, 2015, 2016, 2017, 2018]
        elif predict_year == 2017:
            years = [2013, 2014, 2015, 2016, 2017]
        elif predict_year == 2016:
            years = [2012, 2013, 2014, 2015, 2016]
        elif predict_year == 1995:
            years = [1991, 1992, 1993, 1994, 1995]

        test_num = count_test(predict_year, X_test, data_train)
        out_test = np.zeros(shape=(test_num, 5, num_features + 1))

        out_test_idx = 0
        for i in range(X_test.shape[0]):
            idx = int(X_test[i, 0])
            count = 0
            for j in range(4):  # check if 4 years before pred year are present
                temp = np.where((data_train[:, 1] == idx) & (data_train[:, 2] == predict_year - (j + 1)))[0].size
                if temp == 1:
                    count = count + 1
            if count == 4:
                for k, y in enumerate(years):
                    X = dic[str(y)]
                    ym = avg[str(y)]
                    out_test[out_test_idx, k, :] = np.concatenate((X[(X[:, 1] == idx), :], np.array([[ym]])), axis=1)
                out_test_idx = out_test_idx + 1

        Ybar_test = out_test[:, :, -1].reshape(-1, 5, 1)

        # print(out[:, :, 3:-1].shape)

        # Batch_X_e = np.reshape(out[:, :, 3:-1], (-1,5,6*52+66+14))

        Batch_X_e_test = out_test[:, :, 4:-1]  # np.expand_dims(Batch_X_e,axis=-1)

        Batch_Y_test = out_test[:, -1, 3]

        Batch_Y_test = Batch_Y_test.reshape((len(Batch_Y_test), 1))

        Batch_Y_2_test = out_test[:, np.arange(0, 4), 3]

        test_data = Data(
            w_t1=torch.as_tensor(Batch_X_e_test[:, :, 0:52], device=device).float(),
            w_t2=torch.as_tensor(Batch_X_e_test[:, :, 52 * 1:2 * 52], device=device).float(),
            w_t3=torch.as_tensor(Batch_X_e_test[:, :, 52 * 2:3 * 52], device=device).float(),
            w_t4=torch.as_tensor(Batch_X_e_test[:, :, 52 * 3:4 * 52], device=device).float(),
            w_t5=torch.as_tensor(Batch_X_e_test[:, :, 52 * 4:5 * 52], device=device).float(),
            w_t6=torch.as_tensor(Batch_X_e_test[:, :, 52 * 5:52 * 6], device=device).float(),
            s_t1=torch.as_tensor(Batch_X_e_test[:, :, 312:318], device=device).float(),
            s_t2=torch.as_tensor(Batch_X_e_test[:, :, 318:324], device=device).float(),
            s_t3=torch.as_tensor(Batch_X_e_test[:, :, 324:330], device=device).float(),
            s_t4=torch.as_tensor(Batch_X_e_test[:, :, 330:336], device=device).float(),
            s_t5=torch.as_tensor(Batch_X_e_test[:, :, 336:342], device=device).float(),
            s_t6=torch.as_tensor(Batch_X_e_test[:, :, 342:348], device=device).float(),
            s_t7=torch.as_tensor(Batch_X_e_test[:, :, 348:354], device=device).float(),
            s_t8=torch.as_tensor(Batch_X_e_test[:, :, 354:360], device=device).float(),
            s_t9=torch.as_tensor(Batch_X_e_test[:, :, 366:372], device=device).float(),
            s_t10=torch.as_tensor(Batch_X_e_test[:, :, 372:378], device=device).float(),
            p_t=torch.as_tensor(Batch_X_e_test[:, :, 378:392], device=device).float(),
            y_bar=torch.as_tensor(Ybar_test, device=device).float(),
            y_t=torch.as_tensor(Batch_Y_test, device=device).float(),
            y_t2=torch.as_tensor(Batch_Y_2_test, device=device).float(),
        )

        test_dataset = TensorDataset(test_data.w_t1, test_data.w_t2, test_data.w_t3, test_data.w_t4, test_data.w_t5,
                                     test_data.w_t6,
                                     test_data.s_t1, test_data.s_t2, test_data.s_t3, test_data.s_t4, test_data.s_t5,
                                     test_data.s_t6, test_data.s_t7, test_data.s_t8, test_data.s_t9, test_data.s_t10,
                                     test_data.p_t, test_data.y_bar, test_data.y_t, test_data.y_t2)

        test_fed_loaders.append(test_dataset)
        val_fed_loaders.append(test_dataset)

    return train_fed_loaders, val_fed_loaders, test_fed_loaders


def count_train(years, d1, locations):
    loc_samples = []
    for i in range(len(locations)):
        temp_count = 0
        for j in range(len(years)):
            temp1 = np.where((d1[:, 1] == locations[i]) & (d1[:, 2] == years[j][0]))[0].size
            temp2 = np.where((d1[:, 1] == locations[i]) & (d1[:, 2] == years[j][1]))[0].size
            temp3 = np.where((d1[:, 1] == locations[i]) & (d1[:, 2] == years[j][2]))[0].size
            temp4 = np.where((d1[:, 1] == locations[i]) & (d1[:, 2] == years[j][3]))[0].size
            temp5 = np.where((d1[:, 1] == locations[i]) & (d1[:, 2] == years[j][4]))[0].size
            check = temp1 + temp2 + temp3 + temp4 + temp5
            if check == 5:  # continuous 5-year data available
                temp_count = temp_count + 1
        loc_samples.append(temp_count)
    return sum(loc_samples)


def count_test(years, d1, d2):
    sample_num = 0
    for i in range(d1.shape[0]):
        idx = int(d1[i, 0])
        count = 0
        for j in range(4):  # check if 4 years before pred year are present
            temp = np.where((d2[:, 1] == idx) & (d2[:, 2] == years - (j + 1)))[0].size
            if temp == 1:
                count = count + 1
        if count == 4:
            sample_num += 1
    return sample_num


def averages(X, predict_year):
    A = []

    if predict_year == 2016:
        year_m = 2
    elif predict_year == 2017:
        year_m = 1
    elif predict_year == 2018:
        year_m = 0
    elif predict_year == 1995:
        year_m = 23

    for i in range(4, 39 - year_m):
        A.append([i - 4, i - 3, i - 2, i - 1, i])

    A = np.vstack(A)
    A += 1980
    # print(A.shape)

    dic = {}
    # organize by year
    for i in range(39 - year_m):
        dic[str(i + 1980)] = X[X[:, 2] == i + 1980]

    avg = {}
    avg2 = []
    for i in range(39 - year_m):
        # print("Year: {}, Mean: {}".format(i + 1980, (X[X[:, 1] == i + 1980][:, 2])))
        avg[str(i + 1980)] = np.mean(X[X[:, 2] == i + 1980][:, 3])
        avg2.append(np.mean(X[X[:, 2] == i + 1980][:, 3]))

    # print('avg', avg)

    mm = np.mean(avg2)
    ss = np.std(avg2)

    avg = {}

    for i in range(39 - year_m):
        avg[str(i + 1980)] = (np.mean(X[X[:, 2] == i + 1980][:, 3]) - mm) / ss

    if predict_year == 2016:
        avg['2016'] = avg['2015']

        a8 = np.concatenate((np.mean(dic['2012'][:, 1:], axis=0), [avg['2012']]))

        a9 = np.concatenate((np.mean(dic['2013'][:, 1:], axis=0), [avg['2013']]))
        a10 = np.concatenate((np.mean(dic['2014'][:, 1:], axis=0), [avg['2014']]))

        a11 = np.concatenate((np.mean(dic['2015'][:, 1:], axis=0), [avg['2015']]))

        mean_last = np.concatenate((a8, a9, a10, a11))

    elif predict_year == 2017:
        avg['2017'] = avg['2016']

        a8 = np.concatenate((np.mean(dic['2013'][:, 1:], axis=0), [avg['2013']]))

        a9 = np.concatenate((np.mean(dic['2014'][:, 1:], axis=0), [avg['2014']]))
        a10 = np.concatenate((np.mean(dic['2015'][:, 1:], axis=0), [avg['2015']]))

        a11 = np.concatenate((np.mean(dic['2016'][:, 1:], axis=0), [avg['2016']]))

        mean_last = np.concatenate((a8, a9, a10, a11))

    elif predict_year == 2018:
        avg['2018'] = avg['2017']

        a8 = np.concatenate((np.mean(dic['2014'][:, 1:], axis=0), [avg['2014']]))

        a9 = np.concatenate((np.mean(dic['2015'][:, 1:], axis=0), [avg['2015']]))
        a10 = np.concatenate((np.mean(dic['2016'][:, 1:], axis=0), [avg['2016']]))

        a11 = np.concatenate((np.mean(dic['2017'][:, 1:], axis=0), [avg['2017']]))

        mean_last = np.concatenate((a8, a9, a10, a11))

    elif predict_year == 1995:
        avg['1995'] = avg['1994']

        a8 = np.concatenate((np.mean(dic['1991'][:, 1:], axis=0), [avg['1991']]))

        a9 = np.concatenate((np.mean(dic['1992'][:, 1:], axis=0), [avg['1992']]))
        a10 = np.concatenate((np.mean(dic['1993'][:, 1:], axis=0), [avg['1993']]))

        a11 = np.concatenate((np.mean(dic['1994'][:, 1:], axis=0), [avg['1994']]))

        mean_last = np.concatenate((a8, a9, a10, a11))
    return avg, dic, A, mean_last