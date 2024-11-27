import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.BrownianBridge.CT2PETDiffusionModel import CT2PETDiffusionModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image, smooothing_loss
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
import cv2


def save_deformation(defms):
    heatmapshow = None
    defms_ = defms.data.cpu().float().numpy()
    dir_x = defms_[0]
    dir_y = defms_[1]
    x_max,x_min = dir_x.max(),dir_x.min()
    y_max,y_min = dir_y.max(),dir_y.min()
    dir_x = ((dir_x-x_min)/(x_max-x_min))*255
    dir_y = ((dir_y-y_min)/(y_max-y_min))*255
    tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #tans_x[tans_x<=150] = 0
    tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
    tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #tans_y[tans_y<=150] = 0
    tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
    gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)
    
    # clean the cpu memory
    del defms_
    # cv2.imwrite(root, gradxy) 
    return gradxy
import numpy as np
def tensor2image(tensor_img, save_path = None, max_pixel=32767):
    image = tensor_img.detach().clone()
    image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    np_image = image.mul_(max_pixel).add_(0.2).clamp_(0, max_pixel).permute(1, 2, 0).to('cpu').numpy()
    # save numpy image to disk
    if save_path is not None:
        np.save(save_path, np_image)
    del image
    return np_image 

def plot_images(pet_np, registered, pet_pre, save_path, defms):
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(pet_np, cmap='gray')
    axs[0].set_title('Original PET')
    axs[0].axis('off')

    axs[1].imshow(registered.squeeze(2), cmap='gray')
    axs[1].set_title('Registered PET')
    axs[1].axis('off')

    axs[2].imshow(pet_pre.squeeze(2), cmap='gray')
    axs[2].set_title('Predicted PET')
    axs[2].axis('off')
    
    # Plot defms
    defms_img = save_deformation(defms)
    axs[3].imshow(defms_img)
    axs[3].set_title('Deformation Map')
    axs[3].axis('off')

    plt.savefig(save_path)

@Registers.runners.register_with_name('CPDMRunner')
class CPDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)
        self.reg = config.reg if config.__contains__('reg') else None

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "CPDM":
            bbdmnet = CT2PETDiffusionModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=24,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, r_a, spatial_transform, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name) = batch  # PET, CT 
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        x0_recon, loss = net(x, x_name, x_cond, stage)
        x0 = x
        if self.reg is not None: 
            Trans = r_a(x0_recon, x0)
            SysRegist_A2B = spatial_transform(x0_recon,Trans)
            regloss = (x0 - SysRegist_A2B).abs().mean()
            smooth_loss = smooothing_loss(Trans)
            total_loss = loss + self.reg.lambda1 * regloss + self.reg.lambda2 * smooth_loss
        else:
            total_loss = loss
        
        if write:
            if self.reg is not None: 
                self.writer.add_scalar(f'regloss/{stage}', regloss, step)
                self.writer.add_scalar(f'smooth_loss/{stage}', smooth_loss , step)
                self.writer.add_scalar(f'loss_x0/{stage}',  loss , step)
            self.writer.add_scalar(f'total_loss/{stage}', total_loss , step)
        
        return total_loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        (x, x_name), (x_cond, x_cond_name) = batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        sample, add_cond = net.sample(x_cond, x_name, stage, clip_denoised=self.config.testing.clip_denoised)
        sample.to('cpu')
        add_cond.to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
        
        segment_map = add_cond[:, 0].unsqueeze(1)
        image_grid = get_image_grid(segment_map, grid_size, to_normal=False)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'segmentation_map.png'))
        
        # att_map = add_cond[:, 1].unsqueeze(1)
        # att_map = ((att_map - att_map.min()) / (att_map.max() - att_map.min())).clamp_(0, 1.)
        # image_grid = get_image_grid(att_map, grid_size, to_normal=False)
        # im = Image.fromarray(image_grid)
        # im.save(os.path.join(sample_path, 'attenuation_map.png'))

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        add_condition_path = make_dir(os.path.join(sample_path, f'additional_condition'))
        condition_path = make_dir(os.path.join(sample_path, f'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        for test_batch in pbar:
            (x, x_name), (x_cond, x_cond_name) = test_batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            for j in range(sample_num):
                sample, add_cond = net.sample(x_cond, x_name, 'val_step', clip_denoised=False)
                for i in range(batch_size):
                    condition = x_cond[i].detach().clone()
                    gt = x[i]
                    result = sample[i]
                    if j == 0:
                        save_single_image(add_cond[i], add_condition_path, f'{x_name[i]}.npy', max_pixel=1, to_normal=False)
                        # save_single_image(condition, condition_path, f'{x_cond_name[i]}.npy', max_pixel=self.config.data.dataset_config.max_pixel_cond, to_normal=False)
                        # save_single_image(gt, gt_path, f'{x_name[i]}.npy', max_pixel=self.config.data.dataset_config.max_pixel_ori, to_normal=True)
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, x_name[i]))
                        save_single_image(result, result_path_i, f'output_{j}.npy', max_pixel=self.config.data.dataset_config.max_pixel_ori, to_normal=True)
                    else:
                        save_single_image(result, result_path, f'{x_name[i]}.npy', max_pixel=self.config.data.dataset_config.max_pixel_ori, to_normal=True)



