import torch
from torch import nn
import numpy as np
import copy
import math


def init_masks(model):
    masks = []

    for m in model.modules():
        if isinstance(m, nn.Conv1d):  # for channel pruning
            temp = torch.ones(m.weight.data.shape[0])  # shape: out, in, kernel
            masks.append(temp)
        if isinstance(m, nn.Linear):
            tensor = m.weight.data.cpu().numpy()
            temp2 = np.ones_like(tensor)
            masks.append(temp2)

    return masks


def make_mask(model, num):
    step = 0
    for name, param in model.named_parameters():
        if ('weight' in name and 'conv' in name) or ('weight' in name and 'fc' in name):
            step = step + 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if ('weight' in name and 'conv' in name) or ('weight' in name and 'fc' in name):
            tensor = param.data.cpu().numpy()
            if num == 'one':
                mask[step] = np.ones_like(tensor)
            elif num == 'zero':
                mask[step] = np.zeros_like(tensor)
            step = step + 1
    return mask


# initates a list of masks for the fc layers
def make_init_mask_fc(model):
    step = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            step = step + 1

    mask = [None] * step

    step = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            tensor = m.weight.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            print(f'step {step}, shape: {mask[step].shape}')
            step = step + 1

    return mask


# Mask the model
def mask_model(model, mask, initial_state_dict):
    step = 0
    for name, param in model.named_parameters():
        if ('weight' in name and 'conv' in name) or ('weight' in name and 'fc' in name):
            weight_dev = param.device
            param.data = torch.from_numpy(mask[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if ('bias' in name and 'conv' in name) or ('bias' in name and 'fc' in name):
            param.data = initial_state_dict[name]


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros = 0
    num_elements = 0
    for name, param in model.named_parameters():
        if ('weight' in name and 'conv' in name) or ('weight' in name and 'fc' in name):
            num_nonzeros += param.count_nonzero()
            num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements



# Prune by Percentile module
def prune_by_percentile(model, mask, percent):
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        if ('weight' in name and 'conv' in name) or ('weight' in name and 'fc' in name):
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1


def average_weights_with_masks(w, masks, device):
    '''
    Returns the average of the weights computed with masks.
    '''
    step = 0
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if ('weight' in key and 'conv' in key) or ('weight' in key and 'fc' in key):
            mask = masks[0][step]
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
                mask += masks[i][step]
            w_avg[key] = torch.from_numpy(np.where(mask < 1, 0, w_avg[key].cpu().numpy() / mask)).to(device)
            step += 1
        else:
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# mix global_weight_this_round and global_weight
def mix_global_weights(global_weights_last, global_weights_this_round, masks, device):
    step = 0
    global_weights = copy.deepcopy(global_weights_this_round)
    for key in global_weights.keys():
        if ('weight' in key and 'conv' in key) or ('weight' in key and 'fc' in key):
            mask = masks[0][step]
            for i in range(1, len(masks)):
                mask += masks[i][step]
            global_weights[key] = torch.from_numpy(
                np.where(mask < 1, global_weights_last[key].cpu(), global_weights_this_round[key].cpu())).to(device)
            step += 1
    return global_weights


def evaluate(model, dl, device):
    # evaluation test
    val_loss = 0.0
    model.eval()
    for step, data in enumerate(dl):
        batch_weather, batch_soil, batch_management, batch_y, batch_pre_y = process_loader(data)
        batch_weather, batch_soil, batch_management, batch_y, batch_pre_y = \
            batch_weather.to(device), \
            batch_soil.to(device), \
            batch_management.to(device), \
            batch_y.to(device), \
            batch_pre_y.to(device)
        val_pred = model(batch_weather, batch_soil.unsqueeze(1), batch_management.unsqueeze(1), batch_pre_y)
        criterion = nn.MSELoss()
        loss = criterion(val_pred, batch_y)
        val_loss += loss.item()
    return math.sqrt(val_loss / len(dl))


def process_loader(data):
    batch_weather = data[0:6]
    batch_soil = data[6:16]
    batch_management = data[16]
    batch_y = data[18]
    batch_pre_y = data[19]
    batch_pre_y = batch_pre_y.repeat_interleave(25, dim=1)  # boost
    batch_weather = torch.stack(batch_weather).permute(1, 0, 2, 3)
    batch_weather = batch_weather.view(batch_weather.size(0), batch_weather.size(1), -1)  # concat 5 years
    batch_soil = torch.stack(batch_soil).permute(1, 0, 2, 3)
    batch_soil = torch.mean(batch_soil, 2)  # only need 1 year since they are same
    batch_soil = batch_soil.view(batch_soil.size(0), -1)  # (6,10) into 60 since there are so few
    batch_management = batch_management.view(batch_management.size(0), -1)
    return batch_weather, batch_soil, batch_management, batch_y, batch_pre_y