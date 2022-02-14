import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import utils
import time

#####################################################################################################
######################################## Locator trainer ############################################
#####################################################################################################
class Locator_trainer:
    def __init__(self, model, optimizer, lr_scheduler, device, train_loader, val_loader, logger, checkpoint_dir, max_num_epochs=100,
                num_iterations=1, num_epoch=0, patience=10, iters_to_accumulate=4, best_eval_score=None, eval_score_higher_is_better=False):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        # initialize the best_eval_score
        if not best_eval_score:
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')
        else:
            self.best_eval_score = best_eval_score
        self.patience = patience
        self.epochs_since_improvement = 0
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.fig_dir = os.path.join(checkpoint_dir, 'figs')
        try:
            os.mkdir(self.fig_dir)
        except OSError:
            pass
        self.num_iterations = num_iterations
        self.iters_to_accumulate = iters_to_accumulate
        self.num_epoch = num_epoch
        self.epsilon = 1e-6
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self):
        self._save_init_state()
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            t = time.time()
            should_terminate = self.train(self.train_loader)
            print("Epoch trained in " + str(int(time.time()-t)) + " seconds.")
            if should_terminate:
                print("Hit termination condition...")
                break
            self.num_epoch += 1
        self.writer.close()
        return self.num_iterations, self.best_eval_score

    def train(self, train_loader):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        improved = False        # for early stopping
        self.model.train()      # set the model in training mode
        for batch_idx, sample in enumerate(train_loader):
            self.logger.info(f'Training iteration {self.num_iterations}. Batch {batch_idx + 1}. Epoch [{self.num_epoch + 1}/{self.max_num_epochs}]')
            ct_im = sample['ct_im'].type(torch.HalfTensor)
            target = sample['target'].type(torch.FloatTensor) 
            h_target = sample['h_target'].type(torch.FloatTensor) 
            # send tensors to GPU
            ct_im = ct_im.to(self.device)
            target = target.to(self.device)
            h_target = h_target.to(self.device)
            
            # forward
            with torch.autograd.set_detect_anomaly(True):
                output, loss = self._forward_pass(ct_im, h_target, target)
            train_losses.update(loss.item(), self._batch_size(ct_im))
            
            # compute gradients and update parameters
            # simulate larger batch sizes using gradient accumulation
            loss = loss / self.iters_to_accumulate

            # Native AMP training step
            with torch.autograd.set_detect_anomaly(True):
                self.scaler.scale(loss).backward()
            
            # Every iters_to_accumulate, call step() and reset gradients:
            if self.num_iterations % self.iters_to_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # log stats
                self.logger.info(f'Training stats. Loss: {train_losses.avg}')
                self._log_stats('train', train_losses.avg)
            
            self.num_iterations += 1

        # evaluate on validation set
        self.model.eval()
        eval_score = self.validate()

        # adjust learning rate if necessary
        self.scheduler.step(eval_score)

        # log current learning rate in tensorboard
        self._log_lr()

        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)
        if(is_best):
            improved = True
        
        # save checkpoint
        self._save_checkpoint(is_best)

        # implement early stopping here
        if not improved:
            self.epochs_since_improvement += 1
        if(self.epochs_since_improvement > self.patience):  # Model has not improved for certain number of epochs
            self.logger.info(
                    f'Model not improved for {self.patience} epochs. Finishing training...')
            return True
        return False    # Continue training...
        

    def validate(self):
        self.logger.info('Validating...')
        val_losses = utils.RunningAverage()
        with torch.no_grad():
            which_to_show = np.random.randint(0, self.val_loader.batch_size)
            for batch_idx, sample in enumerate(self.val_loader):
                self.logger.info(f'Validation iteration {batch_idx + 1}')
                ct_im = sample['ct_im'].type(torch.HalfTensor) 
                target = sample['target'].type(torch.FloatTensor)
                h_target = sample['h_target'].type(torch.FloatTensor)  
                
                # send tensors to GPU
                ct_im = ct_im.to(self.device)
                target = target.to(self.device)
                h_target = h_target.to(self.device)
                
                output, loss = self._forward_pass(ct_im, h_target, target)
                val_losses.update(loss.item(), self._batch_size(ct_im))
                
                if (batch_idx == 0) and (self.num_epoch<50 or (self.num_epoch < 500 and not self.num_epoch%10) or (not self.num_epoch%100)):
                    # plot im
                    h_target = h_target.detach().cpu().numpy()[0,0]
                    output = output.detach().cpu().numpy()[0,0]
                    ct_im = ct_im.detach().cpu().numpy()[0,0]
                    # CoM of Heart
                    # axial plot
                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    ax_slice = ct_im[int(target[0,0])].astype(np.float32)              # <-- batch_num, contrast_channel, ax_slice
                    ax0.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                    ax_slice = h_target[int(target[0,0])]
                    ax1.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon, np.max(h_target)))
                    ax_slice = output[int(target[0,0])].astype(np.float32)
                    ax2.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon, np.max(output)))
                    self.writer.add_figure(tag='Val_pred_ax', figure=fig, global_step=self.num_epoch)
                    fig.savefig(os.path.join(self.fig_dir, 'Val_pred_ax_'+str(self.num_epoch)+'.png'))
                    # coronal plot
                    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
                    sag_slice = ct_im[:, int(target[0, 1])].astype(np.float32)  
                    ax3.imshow(sag_slice, aspect=2.0, cmap='Greys_r')
                    sag_slice = h_target[:, int(target[0, 1])]
                    ax4.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(h_target)))
                    sag_slice = output[:, int(target[0, 1])].astype(np.float32)  
                    ax5.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(output)))
                    self.writer.add_figure(tag='Val_pred_sag', figure=fig2, global_step=self.num_epoch)
                    fig2.savefig(os.path.join(self.fig_dir, 'Val_pred_cor_'+str(self.num_epoch)+'.png'))
                    
            self._log_stats('val', val_losses.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}')
            return val_losses.avg

    def _forward_pass(self, ct_im, h_target, target):
        with torch.cuda.amp.autocast():
            # forward pass
            output = self.model(ct_im)
            # MSE loss contribution
            loss = torch.nn.MSELoss()(output, h_target)
            # L1 loss added with fancier solution to keep it on the GPU
            # single target case -> channels dim can get absorbed into spatial ones
            argmax = torch.argmax(output.view(output.size(0), -1), dim=1)
            pred_vox = self._unravel_indices(argmax, output.size()[2:])
            loss += (torch.nn.L1Loss()(pred_vox, target) * 0.01)
            return output, loss

    def _unravel_indices(self, indices, shape):
        """Converts flat indices into unraveled coordinates in a target shape.
        Args:
            indices: A tensor of (flat) indices, (*, N).
            shape: The targeted shape, (D,).

        Returns:
            The unraveled coordinates, (*, N, D).
        """
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode='floor')

        coord = torch.stack(coord[::-1], dim=-1)
        return coord

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score
        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self._log_new_best(eval_score)
            self.best_eval_score = eval_score
            self.epochs_since_improvement = 0
        return is_best

    def _save_init_state(self):
        state = {'model_state_dict': self.model.state_dict()}
        init_state_path = os.path.join(self.checkpoint_dir, 'initial_state.pytorch')
        self.logger.info(f"Saving initial state to '{init_state_path}'")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        torch.save(state, init_state_path)

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_new_best(self, eval_score):
        self.writer.add_scalar('best_val_loss', eval_score, self.num_iterations)

    def _log_stats(self, phase, loss_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            #self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations) #not sure what this is 

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)