import torch
import numpy as np
from utils.stats_manager import StatsManager
from utils.data_logs import save_logs_train, save_logs_eval
import os


class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer, lr_scheduler, config):
        self.config = config
        self.network = network
        self.stats_manager = StatsManager(config)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.best_metric = 0.0

    def train_epoch(self, epoch):
        running_loss = []
        self.network.train()
        for idx, (inputs, labels_depth, labels_distance, labels_magnitude) in enumerate(self.train_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels_depth = labels_depth.to(self.config['device']).float()
            labels_distance = labels_distance.to(self.config['device']).float()
            labels_magnitude = labels_magnitude.to(self.config['device']).float()

            self.optimizer.zero_grad()
            pred_depth, pred_distance, pred_magnitude = self.network(inputs)

            loss = self.criterion(pred_depth, pred_distance, pred_magnitude,
                                  labels_depth, labels_distance, labels_magnitude)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            if idx % self.config['print_loss'] == 0:
                running_loss = np.mean(np.array(running_loss))
                print(f'Training loss on iteration {idx} = {running_loss}')
                save_logs_train(os.path.join(self.config['exp_path'], self.config['exp_name']),
                                f'Training loss on iteration {idx} = {running_loss}')
                running_loss = []

    def eval_net(self, epoch):
        stats_pred_depth = []
        stats_pred_distance = []
        stats_pred_magnitude = []

        stats_lbl_depth = []
        stats_lbl_distance = []
        stats_lbl_magnitude = []

        running_eval_loss = 0.0
        self.network.eval()
        for idx, (inputs, labels_depth, labels_distance, labels_magnitude) in enumerate(self.eval_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels_depth = labels_depth.to(self.config['device']).float()
            labels_distance = labels_distance.to(self.config['device']).float()
            labels_magnitude = labels_magnitude.to(self.config['device']).float()

            with torch.no_grad():
                pred_depth, pred_distance, pred_magnitude = self.network(inputs)

            eval_loss = self.criterion(pred_depth, pred_distance, pred_magnitude,
                                       labels_depth, labels_distance, labels_magnitude)
            running_eval_loss += eval_loss.item()

            stats_pred_depth.append(pred_depth.detach().cpu().numpy())
            stats_pred_distance.append(pred_distance.detach().cpu().numpy())
            stats_pred_magnitude.append(pred_magnitude.detach().cpu().numpy())

            stats_lbl_depth.append(labels_depth.detach().cpu().numpy())
            stats_lbl_distance.append(labels_distance.detach().cpu().numpy())
            stats_lbl_magnitude.append(labels_magnitude.detach().cpu().numpy())

        mean_depth_err, mean_distance_err, mean_magnitude_err = \
            self.stats_manager.get_stats(pred_depth=stats_pred_depth, pred_distance=stats_pred_distance, pred_magnitude=stats_pred_magnitude,
                                         lbl_depth=stats_lbl_depth, lbl_distance=stats_lbl_distance, lbl_magnitude=stats_lbl_magnitude)
        running_eval_loss = running_eval_loss / len(self.eval_dataloader)

        print(f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, mean DEPTH error = {mean_depth_err}, '
              f'mean DISTANCE error = {mean_distance_err}, mean MAGNITUDE error = {mean_magnitude_err}')
        save_logs_eval(os.path.join(self.config['exp_path'], self.config['exp_name']),
                       f'### Evaluation loss on epoch {epoch} = {running_eval_loss}, mean DEPTH error = {mean_depth_err}, '
                       f'mean DISTANCE error = {mean_distance_err}, mean MAGNITUDE error = {mean_magnitude_err}')

        if self.best_metric < mean_magnitude_err:
            self.best_metric = mean_magnitude_err
            self.save_net_state(None, best=True)

    def train(self):
        if self.config['resume_training'] is True:
            checkpoint = torch.load(os.path.join(self.config['exp_path'], self.config['exp_name'], 'latest_checkpoint.pkl'),
                                    map_location=self.config['device'])
            self.network.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for i in range(1, self.config['train_epochs'] + 1):
            print('Training on epoch ' + str(i))
            self.train_epoch(i)
            self.save_net_state(i, latest=True)

            if i % self.config['eval_net_epoch'] == 0:
                self.eval_net(i)

            if i % self.config['save_net_epochs'] == 0:
                self.save_net_state(i)

            self.lr_scheduler.step()

    def save_net_state(self, epoch, latest=False, best=False):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'best_model.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.network.state_dict()
            }
            torch.save(to_save, path_to_save)
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'model_epoch_{epoch}.pkl')
            torch.save(self.network, path_to_save)

    def test_net(self, test_dataloader):
        stats_pred_depth = []
        stats_pred_distance = []
        stats_pred_magnitude = []

        stats_lbl_depth = []
        stats_lbl_distance = []
        stats_lbl_magnitude = []

        running_loss = 0.0
        self.network.eval()
        for idx, (inputs, labels_depth, labels_distance, labels_magnitude) in enumerate(test_dataloader, 0):
            inputs = inputs.to(self.config['device']).float()
            labels_depth = labels_depth.to(self.config['device']).float()
            labels_distance = labels_distance.to(self.config['device']).float()
            labels_magnitude = labels_magnitude.to(self.config['device']).float()

            with torch.no_grad():
                pred_depth, pred_distance, pred_magnitude = self.network(inputs)

            eval_loss = self.criterion(pred_depth, pred_distance, pred_magnitude,
                                       labels_depth, labels_distance, labels_magnitude)
            running_loss += eval_loss.item()

            stats_pred_depth.append(pred_depth.detach().cpu().numpy())
            stats_pred_distance.append(pred_distance.detach().cpu().numpy())
            stats_pred_magnitude.append(pred_magnitude.detach().cpu().numpy())

            stats_lbl_depth.append(labels_depth.detach().cpu().numpy())
            stats_lbl_distance.append(labels_distance.detach().cpu().numpy())
            stats_lbl_magnitude.append(labels_magnitude.detach().cpu().numpy())

        mean_depth_err, mean_distance_err, mean_magnitude_err = \
            self.stats_manager.get_stats(pred_depth=stats_pred_depth, pred_distance=stats_pred_distance,
                                         pred_magnitude=stats_pred_magnitude,
                                         lbl_depth=stats_lbl_depth, lbl_distance=stats_lbl_distance,
                                         lbl_magnitude=stats_lbl_magnitude)
        running_eval_loss = running_loss / len(test_dataloader)

        stats_description = f'### Test loss = {running_eval_loss}, mean DEPTH error = {mean_depth_err}, \
                              mean DISTANCE error = {mean_distance_err}, mean MAGNITUDE error = {mean_magnitude_err}'

        print(stats_description)
        history = open(os.path.join(self.config['exp_path'], self.config['exp_name'], '__testStats__.txt'), "a")
        history.write(stats_description)
        history.close()
