import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np


def rand_init_l2(img_variable, eps_max):
    random_vec = torch.FloatTensor(*img_variable.shape).normal_(0, 1).cuda()
    random_vec_norm = torch.max(
               random_vec.view(random_vec.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
    random_dir = random_vec/random_vec_norm.view(random_vec.size(0),1,1,1)
    random_scale = torch.FloatTensor(img_variable.size(0)).uniform_(0, eps_max).cuda()
    random_noise = random_scale.view(random_vec.size(0),1,1,1)*random_dir
    img_variable = Variable(img_variable.data + random_noise, requires_grad=True)

    return img_variable

def rand_init_linf(img_variable, eps_max):
    random_noise = torch.FloatTensor(*img_variable.shape).uniform_(-eps_max, eps_max).to(device)
    img_variable = Variable(img_variable.data + random_noise, requires_grad=True)

    return img_variable

def track_best(blosses, b_adv_x, curr_losses, curr_adv_x):
    if blosses is None:
        b_adv_x = curr_adv_x.clone().detach()
        blosses = curr_losses.clone().detach()
    else:
        replace = curr_losses < blosses
        b_adv_x[replace] = curr_adv_x[replace].clone().detach()
        blosses[replace] = curr_losses[replace]

    return blosses, b_adv_x


def cal_loss(y_out, y_true, targeted, loss_function, optimal_scores):
    if loss_function == 'CE':
        losses = torch.nn.CrossEntropyLoss(reduction='none')
        losses_cal = losses(y_out, y_true)
    elif loss_function == 'KL':
        losses = torch.nn.KLDivLoss(reduction='none')
        losses_cal = losses(y_out, optimal_scores)
        losses_cal = torch.sum(losses_cal,dim=1)
    loss_cal = torch.mean(losses_cal)
    if targeted:
        return loss_cal, losses_cal
    else:
        return -1*loss_cal, -1*losses_cal

def generate_target_label_tensor(true_label, args):
    t = torch.floor(args.n_classes*torch.rand(true_label.shape)).type(torch.int64)
    m = t == true_label
    t[m] = (t[m]+ torch.ceil((args.n_classes-1)*torch.rand(t[m].shape)).type(torch.int64)) % args.n_classes
    return t

def pgd_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted, rand_init):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    
    best_losses = None
    best_adv_x = None

    if rand_init:
        img_variable = rand_init_linf(img_variable, eps_max)

    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)
        output = model.forward(img_variable)
        loss_cal, losses_cal = cal_loss(output, tar_label_variable, targeted)
        best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)

        loss_cal, losses_cal = cal_loss(output, tar_label_variable, targeted)
        loss_cal.backward()
        x_grad = -1 * eps_step * torch.sign(img_variable.grad.data)
        adv_temp = img_variable.data + x_grad
        total_grad = adv_temp - image_tensor
        total_grad = torch.clamp(total_grad, -eps_max, eps_max)
        x_adv = image_tensor + total_grad
        x_adv = torch.clamp(torch.clamp(
            x_adv-image_tensor, -1*eps_max, eps_max)+image_tensor, clip_min, clip_max)
        img_variable.data = x_adv

    best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)
    #print("peturbation= {}".format(
    #    np.max(np.abs(np.array(x_adv)-np.array(image_tensor)))))
    return best_adv_x

def pgd_l2_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted, 
               rand_init, num_restarts, image_tensor_mod=None, unmod_indices=None, 
               loss_function='CE', optimal_scores=None):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    
    best_losses = None
    best_adv_x = None
    image_tensor_orig = image_tensor.clone().detach()
    tar_label_orig = tar_label_variable.clone().detach()

    for j in range(num_restarts):
        if rand_init:
            img_variable = rand_init_l2(img_variable, eps_max)

        if image_tensor_mod is not None:
            # print('Using hybrid attack')
            image_tensor_init = torch.cat((img_variable.data[unmod_indices],
                                            image_tensor_mod))
            image_tensor = torch.cat((image_tensor_orig[unmod_indices], 
                                        image_tensor_orig[~unmod_indices])) 
            tar_label_variable = torch.cat((tar_label_orig[unmod_indices],
                                        tar_label_orig[~unmod_indices]))
            img_variable = Variable(image_tensor_init, requires_grad=True)
        else:
            image_tensor = image_tensor_orig
            
        output = model.forward(img_variable)
        for i in range(n_steps):
            zero_gradients(img_variable)
            output = model.forward(img_variable)
            loss_cal, losses_cal = cal_loss(output, tar_label_variable, targeted, 
                                            loss_function,optimal_scores)
            best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)
            loss_cal.backward()
            raw_grad = img_variable.grad.data
            grad_norm = torch.max(
                   raw_grad.view(raw_grad.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
            if len(img_variable.size())==2:
                grad_dir = raw_grad/grad_norm.view(raw_grad.size(0),1)
            else:
                grad_dir = raw_grad/grad_norm.view(raw_grad.size(0),1,1,1)
            adv_temp = img_variable.data +  -1 * eps_step * grad_dir
            # Clipping total perturbation
            total_grad = adv_temp - image_tensor
            total_grad_norm = torch.max(
                   total_grad.view(total_grad.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
            if len(img_variable.size())==2:
                total_grad_dir = total_grad/total_grad_norm.view(total_grad.size(0),1)
            else:
                total_grad_dir = total_grad/total_grad_norm.view(total_grad.size(0),1,1,1)
            total_grad_norm_rescale = torch.min(total_grad_norm, torch.tensor(eps_max).cuda())
            if len(img_variable.size())==2:
                clipped_grad = total_grad_norm_rescale.view(total_grad.size(0),1) * total_grad_dir
            else:
                clipped_grad = total_grad_norm_rescale.view(total_grad.size(0),1,1,1) * total_grad_dir
            x_adv = image_tensor + clipped_grad
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
            img_variable.data = x_adv

        best_losses, best_adv_x = track_best(best_losses, best_adv_x, losses_cal, img_variable)

        diff_array = np.array(x_adv.cpu())-np.array(image_tensor.data.cpu())
        diff_array = diff_array.reshape(len(diff_array),-1)

        img_variable.data = image_tensor_orig
        # print("peturbation= {}".format(
        #    np.max(np.linalg.norm(diff_array,axis=1))))
    return best_adv_x


def hybrid_attack(matched_x, ez, m, data_loader, eps):
    # Fill up a tensor with actual matched data
    matched_data = []
    matched_labels = []
    for idx in m[~ez]:
        x_curr, y_curr, _, _, _ = data_loader[idx]
        matched_data.append(x_curr)
        matched_labels.append(y_curr)
    matched_data = torch.stack(matched_data).cuda()

    # Construct the perturbation
    diff_vec = matched_data-matched_x
    diff_vec_norm = torch.max(diff_vec.view(diff_vec.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
    # print(diff_vec_norm)
    diff_vec_dir = diff_vec/diff_vec_norm.view(diff_vec.size(0),1,1,1)
    x_mod = matched_x + eps * diff_vec_dir

    return x_mod

