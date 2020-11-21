import json
import os

import shutil
import torch
import torch.optim as optim
from networks.EarthNetComplex import EarthNetComplex

from data.data_manager import DataManager
from trainer import Trainer
from utils.data_logs import save_logs_about
import utils.losses as loss_functions


def main():
    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    model = EarthNetComplex().to(config['device'])
    model.apply(EarthNetComplex.init_weights)

    # Save info about experiment
    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))
    shutil.copy(model.get_path(), os.path.join(config['exp_path'], config['exp_name']))

    criterion = getattr(loss_functions, config['loss_function'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'], last_epoch=-1)

    data_manager = DataManager(config)
    train_loader, validation_loader, test_loader = data_manager.get_train_eval_test_dataloaders()

    trainer = Trainer(model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, config)
    trainer.train()

    trainer.test_net(test_loader)


def test_net():
    # Function made only to test a pretrained network.
    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EarthNetComplex().to(config['device'])
    checkpoint = torch.load(os.path.join(config['exp_path'], config['exp_name'], 'latest_checkpoint.pkl'),
                            map_location=config['device'])
    model.load_state_dict(checkpoint['model_weights'])

    criterion = getattr(loss_functions, config['loss_function'])

    data_manager = DataManager(config)
    _, _, test_loader = data_manager.get_train_eval_test_dataloaders()

    trainer = Trainer(model, None, None, criterion, None, None, config)
    trainer.test_net(test_loader)


if __name__ == "__main__":
    main()
