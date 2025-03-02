import os
import torch
import wandb
import logging
import shutil
import collections
from utils import AverageMeter
from model import RecurrentAttention
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd


class Trainer(object):
    def __init__(self, config, data_loader, used_channel):

        self.config = config
        self.used_channel = used_channel

        self.num_glimpses = config.num_glimpses
        self.rnn_hidden_size = config.rnn_hidden_size
        self.measure_embedding_hidden_size = list(map(lambda x: int(x), config.measure_embedding_hidden_size.split(',')))
        self.RIS_phase_power_embedding_hidden_size = list(map(lambda x: int(x), config.RIS_phase_power_embedding_hidden_size.split(',')))
        self.RIS_phase_customization_hidden_size = list(map(lambda x: int(x), config.RIS_phase_customization_hidden_size.split(',')))
        self.classify_hidden_size = list(map(lambda x: int(x), config.classify_hidden_size.split(',')))

        # verify network dims, which should match the present codes
        assert len(self.measure_embedding_hidden_size) == 2, \
            'network dims error! present codes only support len(measure_embedding_hidden_size) == 2'
        assert len(self.RIS_phase_power_embedding_hidden_size) == 2, \
            'network dims error! present codes only support len(RIS_phase_power_embedding_hidden_size) == 2'
        assert len(self.RIS_phase_customization_hidden_size) == 1, \
            'network dims error! present codes only support len(RIS_phase_customization_hidden_size) == 1'
        assert len(self.classify_hidden_size) == 1, \
            'network dims error! present codes only support len(classify_hidden_size) == 1'
        assert self.measure_embedding_hidden_size[-1] == self.RIS_phase_power_embedding_hidden_size[-1], \
            'network dims error! measure_embedding_hidden_size and RIS_phase_power_embedding_hidden_size must have the same output size'
        
        transmit_power = list(map(lambda x: int(x), config.transmit_power.split(',')))[0] # dBm
        self.transmit_scale = torch.tensor(10).pow((transmit_power - 30) / 10).pow(0.5)

        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_classes = data_loader[2]
            self.num_train = data_loader[3]
            self.num_valid = data_loader[4]
        else:
            self.test_loader = data_loader[0]
            self.num_classes = data_loader[1]
            self.num_test = data_loader[2]

        self.epochs = config.epochs
        self.start_epoch = 1
        self.lr = config.init_lr

        self.device = config.device
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.resume = config.resume

        if not config.learned_start:
            model_dir = '***' # a given RIS phase configuration for the first illumination
            para_dict = torch.load(model_dir, map_location='cpu')
            self.RIS_phase_power_initial = para_dict['model_state']['measure_embedding.first_RIS_phase_power']
            self.RIS_phase_power_initial = self.RIS_phase_power_initial.to(device=self.device)

        self.model = RecurrentAttention(self.measure_embedding_hidden_size, self.RIS_phase_power_embedding_hidden_size, 
                                        self.RIS_phase_customization_hidden_size, self.classify_hidden_size, 
                                        self.used_channel, self.rnn_hidden_size, self.num_classes, 
                                        self.config.learned_start, self.transmit_scale)
        self.model = self.model.to(device=self.device)
        for i in range(len(self.model.measure_embedding.physical.used_channel)):
            self.model.measure_embedding.physical.used_channel[i] = self.model.measure_embedding.physical.used_channel[i].to(device=self.device)

        if self.config.wandbflag:
            logging.info('********************\n' + str(self.model) + '\n********************')
            logging.info('[*] Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.lr_patience)
        self.experiment = wandb.init(project=config.wandb_project, resume='allow', anonymous='must', config=config) if config.wandbflag else 0


    def train(self):

        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(self.config.resume_load_model_dir)
        logging.info("\n[*] Train on {} samples, validate on {} samples".format(self.num_train, self.num_valid))

        for epoch in range(self.start_epoch, self.epochs + 1):
            
            self.curr_epoch = epoch
            logging.info('Epoch: {}/{} - LR: {:.6f}'.format(epoch, self.epochs, self.lr))

            self.model.train()
            train_loss, train_acc, glimpses, train_glimpse_counter = self.train_one_epoch(epoch) # train for 1 epoch
            self.model.eval()
            valid_loss, valid_acc, val_glimpses, val_glimpse_counter = self.validate(epoch) # evaluate on validation set
            self.scheduler.step(valid_loss)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} - train glm {:.3f}"
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val glm {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            logging.info(msg.format(train_loss, train_acc, glimpses, valid_loss, valid_acc, val_glimpses))
            logging.info(sorted(train_glimpse_counter.items()))
            logging.info(sorted(val_glimpse_counter.items()))
            if self.config.wandbflag:
                self.experiment.log({'valid_loss': valid_loss, 'train_loss': train_loss,
                                     'valid_accuracy': valid_acc, 'train_accuracy': train_acc,
                                     'valid_glimpses': val_glimpses, 'train_glimpses': glimpses,
                                     'learning_rate': self.optimizer.param_groups[0]['lr'],'epoch': epoch})

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                logging.info("[!] No improvement in {} epochs, stopping training.".format(self.train_patience))
                self.save_checkpoint({'epoch': epoch, 'model_state': self.model.state_dict(),
                                    'optim_state': self.optimizer.state_dict(), 'best_valid_acc': self.best_valid_acc,}, is_best)
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            if self.config.wandbflag and (is_best or epoch % self.config.ckpt_save_interval == 0):
                self.save_checkpoint({'epoch': epoch, 'model_state': self.model.state_dict(),
                                    'optim_state': self.optimizer.state_dict(), 'best_valid_acc': self.best_valid_acc,}, is_best)


    def train_one_epoch(self, epoch):

        losses = AverageMeter()
        accs = AverageMeter()
        glimpses = AverageMeter()
        glimpse_counter = collections.Counter()
        for batch in self.train_loader:
            x, y = batch
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            loss, glm, acc = self.rollout(x, y)
            glimpses.update(glm, x.shape[0])
            losses.update(loss.data.item(), x.shape[0])
            accs.update(acc.data.item(), x.shape[0])

            # compute gradients and update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            # measure elapsed time
            if self.config.wandbflag:
                self.experiment.log({'train_loss': losses.avg, 'train_accuracy': accs.avg, 'train_glimpses': glimpses.avg, 'epoch': epoch,})

        return losses.avg, accs.avg, glimpses.avg, glimpse_counter


    def rollout(self, x, y):

        num_glimpses = self.config.num_glimpses
        batch_size = x.shape[0]
        
        if self.config.learned_start:
            RIS_phase_power_t = None
        else:
            RIS_phase_power_t = self.RIS_phase_power_initial.repeat(batch_size, 1).unsqueeze(dim=1)
        state_t = None
        used_forward_time = 0
        for t in range(num_glimpses):
            # forward pass through model
            state_t, RIS_phase_power_t, classifier_log_probs = self.model(x, RIS_phase_power_t, state_t)

        # calculate reward
        predicted = torch.max(classifier_log_probs, 1)[1]
        correct = (predicted.detach() == y.long()).float()

        # compute losses for differentiable modules
        loss_action = F.nll_loss(classifier_log_probs, y)
        loss = loss_action
        # compute accuracy
        acc = 100 * (correct.sum() / len(y))

        return loss, num_glimpses, acc


    def validate(self, epoch):

        losses = AverageMeter()
        accs = AverageMeter()
        glimpses = AverageMeter()
        glimpse_counter = collections.Counter()
        for batch in self.valid_loader:
            x, y = batch
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            loss, glm, acc = self.rollout(x,y)
            glimpses.update(glm, x.shape[0])
            losses.update(loss.data.item(), x.shape[0])
            accs.update(acc.data.item(), x.shape[0])

        return losses.avg, accs.avg, glimpses.avg, glimpse_counter


    def test(self):
        wandb_data_csv = pd.read_csv(self.config.test_wandb_data)
        test_index = self.config.test_index - 1
        num_glimpses = wandb_data_csv.loc[test_index, 'num_glimpses']

        ## use the best model
        correct = 0
        num_test = 0
        test_model_dir = wandb_data_csv.loc[test_index, 'ckpt_dir'] + '/model_best.pth'
        self.load_checkpoint(test_model_dir)

        error_predict_counter = collections.Counter()
        correct_predict_counter = collections.Counter()
        error_prob = [[] for _ in range(self.num_classes)]
        correct_prob = [[] for _ in range(self.num_classes)]

        for batch in self.test_loader:
            x, y = batch
            x = x.to(device=self.device)
            y = y.to(device=self.device)
            batch_size = x.shape[0]
            num_test += batch_size

            if self.config.learned_start:
                RIS_phase_power_t = None
            else:
                RIS_phase_power_t = self.RIS_phase_power_initial.repeat(batch_size, 1).unsqueeze(dim=1)
            state_t = None
            for t in range(num_glimpses):
                # forward pass through model
                state_t, RIS_phase_power_t, classifier_log_probs = self.model(x, RIS_phase_power_t, state_t)

            predicted = torch.max(classifier_log_probs, 1)[1]
            correct += (predicted.detach() == y.long()).float().sum()
            error_predict_counter.update(y[predicted.detach() != y.long()].type(torch.long).tolist())
            correct_predict_counter.update(y[predicted.detach() == y.long()].type(torch.long).tolist())
            value, classnum = torch.max(torch.exp(classifier_log_probs[predicted.detach() != y.long(), :]), dim=1)
            for i in range(self.num_classes):
                error_prob[i] += list(value[classnum == i].data)
            value, classnum = torch.max(torch.exp(classifier_log_probs[predicted.detach() == y.long(), :]), dim=1)
            for i in range(self.num_classes):
                correct_prob[i] += list(value[classnum == i].data)

        error_prob = [torch.tensor(error_prob[i]).mean() for i in range(self.num_classes)]
        correct_prob = [torch.tensor(correct_prob[i]).mean() for i in range(self.num_classes)]
        perc = (100. * correct) / num_test
        error = 100 - perc
        glimpse_ave = num_glimpses
        wandb_data_csv.loc[test_index, 'predict_correct_rate_best_model'] = perc.data.cpu().detach().numpy()
        wandb_data_csv.loc[test_index, 'predict_glimpse_best_model'] = glimpse_ave
        logging.info('[*] Test Accuracy: {}/{} ({:.2f}% - {:.2f}%), average glimpses {:.2f}'.format(int(correct), num_test, perc, error, glimpse_ave))
        logging.info('number of error prediction instances for each class: {}'.format(sorted(error_predict_counter.items())))
        logging.info('number of correct prediction instances for each class: {}'.format(sorted(correct_predict_counter.items())))
        logging.info('average confidence of error prediction for each class: {}'.format(error_prob))
        logging.info('average confidence of correct prediction for each class: {}'.format(correct_prob))

        wandb_data_csv.to_csv(self.config.test_wandb_data, sep=',', index=False, header=True)


    def save_checkpoint(self, state, is_best):

        filename = 'epoch' + str(state['epoch']) + '_ckpt.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        logging.info("[*] Saving model to {}".format(self.ckpt_dir))

        if is_best:
            filename = 'model_best.pth'
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))


    def load_checkpoint(self, filename):

        ckpt = torch.load(filename, map_location=self.device)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        logging.info("[*] Loaded {} checkpoint @ epoch {} with valid acc of {:.3f}".format(filename, ckpt['epoch'], ckpt['best_valid_acc']))
