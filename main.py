import pandas as pd
import matplotlib.pyplot as plt
from src.utils import *
from src.model import SoybeanCNN
from dataloaders_soybean import load_all_data
from src.client import *


ds_path = './data/soybean_data_soilgrid250_modified_states_9.csv'
loc_path = './data/Soybeans_Loc_ID.csv'
num_users = 9
rounds = 40
bs = 50
ep = 8
lr = 0.00001
threshold = 30
is_prune = True
target_prune_ratio = 0.79     # max prune ratio, 0-1
prune_percent = 41.5  # amount to prune per round, 0-100
prunt_wait = 7
year = 2018
early_stop = True
lottery_ticket = False


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("you are using GPU")
else:
    device = torch.device("cpu")
    print("you are using CPU")

train_fed_loaders, val_fed_loaders, test_fed_loaders = load_all_data(ds_path, year)
#
# train_fed_loaders[0] = ConcatDataset([train_fed_loaders[0], train_fed_loaders[1],train_fed_loaders[2],train_fed_loaders[3],
#                                           train_fed_loaders[4],train_fed_loaders[5],train_fed_loaders[6],train_fed_loaders[7],train_fed_loaders[8]])
# test_fed_loaders[0] = ConcatDataset([test_fed_loaders[0],test_fed_loaders[1],test_fed_loaders[2],test_fed_loaders[3],test_fed_loaders[4],test_fed_loaders[5],test_fed_loaders[6],test_fed_loaders[7], test_fed_loaders[8]])
# val_fed_loaders[0] = test_fed_loaders[0]

# train_fed_loaders[8]=test_fed_loaders[5]
# test_fed_loaders[8]=test_fed_loaders[5]
# val_fed_loaders[8]=val_fed_loaders[5]

# initialize models
global_model = SoybeanCNN().to(device)

users_model = [copy.deepcopy(global_model) for _ in range(num_users)]  # initialize identical models for clients
client_idx = list(range(0, num_users))

initial_state_dict = copy.deepcopy(global_model.state_dict())
global_weights = global_model.state_dict()

is_aggre = [1] * num_users
# make masks
masks_list = []
# masks = init_masks(global_model)
# masks = make_mask(global_model.module_list)
masks = make_mask(global_model, 'one')
black_mask = make_mask(global_model, 'zero')
for i in range(num_users):
    masks_list.append(copy.deepcopy(masks))

# initialize clients
clients = []
for i in range(num_users):
    users_model[i].load_state_dict(initial_state_dict)
    clients.append(Client(i, copy.deepcopy(users_model[i]), train_fed_loaders[i], val_fed_loaders[i], test_fed_loaders[i],
                          bs, ep, lr, copy.deepcopy(masks_list[i]), 0, threshold, prunt_wait, device))


Byte = 8
KiB = 1024 * Byte
KB = 0.976562 * KiB
MiB = 1024 * KiB
MB = 0.953674 * MiB
GiB = 1024 * MiB
dense_model_size = get_model_size(global_model)
print('model has parameters={} and size={:.2f} KB'.format(get_num_parameters(global_model),dense_model_size/KB))

# metrics
best_losses = []
round_losses = []
prune_ratios = []
history = []
client_history = []
param_pruned = []

# training
for r in range(rounds):
    print('Start Round {} ...'.format(r + 1))

    if not is_prune:  # fedavg only
        global_test_loss = []
        curr_parameters = global_model.get_parameters()
        new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
        # train local models
        for idx in client_idx:
            print("starting client", idx + 1)
            # train based on conditions
            _, client_parameters, loss = clients[idx].train(copy.deepcopy(global_model))
            fraction = 1.0 / num_users
            for layer_name in client_parameters:
                new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
                new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']
        global_model.apply_parameters(new_parameters)
        # test
        for i in range(num_users):
            # test_loss = global_model.evaluate(test_fed_loaders[i], bs=bs, device=device)
            test_loss = evaluate(clients[i].model, clients[i].test_fed_loader, device)
            print(
                "Client {}, test loss (aggregated): {:10.4f}".format(i + 1, test_loss))
            global_test_loss.append(test_loss)

        print("End of round {}, overall loss: {}".format(r + 1, np.mean(global_test_loss)))

    else:
        local_weights, local_losses, local_sparsity = [], [], []
        prune_ratios_this_round = []
        temp_masks = []
        avg_agg_loss_this_round = 0
        print(is_aggre)
        for client in clients:
            if r == 4 or r == 9 or r == 19:
                client.lr = lr * 0.2
                print('***** Learning rate set to ', client.lr)
            print('---------------------------client {}---------------------------'.format(client.idx+1))

            # if is_aggre[client.idx] == 0:    # if client did not aggregate last round, weights are combined with global
            #     a = copy.deepcopy(global_model.state_dict())
            #     b = copy.deepcopy(client.model.state_dict())
            #     for layer in a:
            #         a[layer] = (a[layer] + b[layer]) / 2
            #     tm = SoybeanCNN().to(device)
            #     tm.load_state_dict(a)
            #     local_model = tm
            # else:
            #     local_model = copy.deepcopy(global_model)
            local_model = copy.deepcopy(global_model)
            mask_model(local_model, client.mask, local_model.state_dict())

            # prune if target sparsity has not been reached
            if target_prune_ratio > get_model_sparsity(local_model):

                temp_w, train_loss = client.train(local_model, freeze=True, early_stop=early_stop)
                # temp_model = local_model
                temp_mask = client.mask

                acc_beforePrune = evaluate(local_model, client.test_fed_loader, device)
                print('current threshold', client.threshold)

                if r == 2:
                    client.prune_wait = 0
                print('prune_wait: ', client.prune_wait)
                if client.prune_wait <= 0 and 2 < r < rounds - 3 and acc_beforePrune < client.threshold * 1.2+0.5:
                    client.threshold = acc_beforePrune
                    print('acc_beforePrune', acc_beforePrune)
                    prune_by_percentile(local_model, client.mask, prune_percent)    # prune
                    print("sparsity after pruning ",get_model_sparsity(local_model))
                    client.prune_wait = prunt_wait
                    acc_afterPrune = evaluate(local_model, client.test_fed_loader, device)
                    print('acc_afterPrune', acc_afterPrune)
                    if lottery_ticket:
                        mask_model(local_model, client.mask, initial_state_dict)        # reset weights to initial values (LTH)
                        print('weights reset to init')
                    # recover loss
                    w, train_loss = client.train(local_model, freeze=True, early_stop=early_stop)
                    l = evaluate(local_model, client.test_fed_loader, device)
                    if l < client.threshold * 1.2+0.5:      # if loss has recovered (nearly)
                        client.threshold = l
                        is_aggre[client.idx] = 1
                    else:       # if weights have not recovered, skip aggregation
                        is_aggre[client.idx] = 0
                else:
                    w = temp_w
                    is_aggre[client.idx] = 1
                    client.prune_wait = client.prune_wait - 1
            else:
                # train
                w, train_loss = client.train(local_model, freeze=True, early_stop=early_stop)
                is_aggre[client.idx] = 1

            local_weights.append(w)
            client.model = local_model
            val_loss = evaluate(client.model, client.test_fed_loader, device)
            local_sparsity = get_model_sparsity(local_model)
            client.current_prune_ratio = local_sparsity
            # test_loss = local_model.evaluate(test_fed_loaders[client.idx], bs=len(test_fed_loaders[client.idx]), device=device)
            test_loss = evaluate(client.model, client.test_fed_loader, device)
            print(
                "user {}, train_loss = {}, val_loss = {}, test_loss = {}, sparsity = {}".format(
                    client.idx + 1, train_loss, val_loss, test_loss, local_sparsity))

            client.weights = copy.deepcopy(w)
            client.loss = val_loss
            local_losses.append(val_loss)

            # for i in range(len(is_aggre)):
            #     if is_aggre[i] == 1:
            #         temp_masks.append(copy.deepcopy(client.mask))
            #     elif is_aggre[i] == 0:
            #         temp_masks.append(black_mask)
            temp_masks.append(copy.deepcopy(client.mask))
            prune_ratios_this_round.append(client.current_prune_ratio)

        # update global weights
        global_weights_this_round = average_weights_with_masks(local_weights, temp_masks, device)
        global_weights = mix_global_weights(global_weights, global_weights_this_round, temp_masks, device)
        global_model.load_state_dict(global_weights)

        val_loss_avg = sum(local_losses) / len(local_losses)
        round_losses.append(val_loss_avg)

        # compute communication cost this round
        param_pruned.append(sum(prune_ratios_this_round) / len(prune_ratios_this_round))


        # average losses over all users
        client_agg_loss = []
        client_local_loss = []
        client_sparsity = []
        global_model.eval()
        for client in clients:
            local_mask_model = copy.deepcopy(global_model)
            mask_model(local_mask_model, client.mask, local_mask_model.state_dict())
            # agg_loss = local_mask_model.evaluate(test_fed_loaders[idx], bs=len(test_fed_loaders[idx]), device=device)
            agg_loss = evaluate(local_mask_model, client.test_fed_loader, device)
            local_loss = clients[client.idx].loss
            local_model_size = get_model_size(local_mask_model, count_nonzero_only=True)
            local_sparsity = get_model_sparsity(client.model)
            print('Client {} aggregated loss: {:.4f}, local loss: {:.4f}, sparsity: {:.2f}, size: {:.2f} KB'.format(client.idx+1, agg_loss, local_loss, local_sparsity, local_model_size/KB))
            client_agg_loss.append(agg_loss)
            client_local_loss.append(local_loss)
            client_sparsity.append(local_sparsity)
        avg_agg_loss_this_round = sum(client_agg_loss) / len(client_agg_loss)
        print('Average aggregated loss this round for year {}: {}'.format(year, avg_agg_loss_this_round))
        print('Average local loss: {}'.format(val_loss_avg))
        print('Communication cost reduced so far: {}'.format(sum(param_pruned)/len(param_pruned)))
        print('Average prune ratio: {}\n'.format(sum(prune_ratios_this_round)/len(prune_ratios_this_round)))

        # min_local_loss.append(min(local_losses))
        # max_local_loss.append(max(local_losses))
        # min_agg_loss.append(min(client_agg_loss))
        # max_agg_loss.append(max(client_agg_loss))

        history.append((avg_agg_loss_this_round, val_loss_avg, sum(prune_ratios_this_round)/len(prune_ratios_this_round),
                        min(local_losses), max(local_losses), min(client_agg_loss), max(client_agg_loss), sum(param_pruned)/len(param_pruned)))
        client_history.append((client_agg_loss, client_local_loss, client_sparsity))

# export result
history_export = pd.DataFrame(history)
history_export.to_csv('./result/result.csv')
client_history_export = pd.DataFrame(client_history)
client_history_export.to_csv('./result/result_client.csv')

# graph
if history[-1][2] > 0.05:
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot([i + 1 for i in range(len(history))], [history[i][0] for i in range(len(history))], color='b',
             label='Average global test_loss')
    ax1.plot([i + 1 for i in range(len(history))], [history[i][1] for i in range(len(history))], color='g',
             label='Average local test loss')

    ax2.plot([i + 1 for i in range(len(history))], [history[i][2] for i in range(len(history))], color='r',
             label='Average prune ratio')
    ax2.set_xlabel('Rounds')
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('Prune ratio')
    ax1.legend()
    ax2.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot([i + 1 for i in range(len(history))], [history[i][1] for i in range(len(history))], color='g',
             label='Average validation loss')
    ax1.fill_between([i + 1 for i in range(len(history))], [history[i][3] for i in range(len(history))],
                     [history[i][4] for i in range(len(history))], color='palegreen')

    ax2.plot([i + 1 for i in range(len(history))], [history[i][2] for i in range(len(history))], color='r',
             label='Average prune ratio')
    ax2.set_xlabel('Rounds')
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('Prune ratio')
    ax1.legend()
    ax2.legend()
    plt.show()

else:
    plt.plot([i + 1 for i in range(len(history))], [history[i][0] for i in range(len(history))], color='b',
             label='Average global test_loss')
    plt.plot([i + 1 for i in range(len(history))], [history[i][1] for i in range(len(history))], color='g',
             label='Average local test loss')
    plt.legend()
    # plt.title('Training history')
    plt.show()
#
# plt.plot([i + 1 for i in range(len(history))], [client_history[i] for i in range(len(client_history))])
# plt.title('client losses')
# plt.show()