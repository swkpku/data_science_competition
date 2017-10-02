import time, os
import torch
import shutil
import visdom
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class Trainer():
    def __init__(self, train_dataloader, val_dataloader, model, criterion, config):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.criterion = criterion
        self.config = config
        
        self.viz = visdom.Visdom()
        self.loss_viz = self.viz.line(Y=np.array([9]), X=np.array([0]))
        self.acc_viz = self.viz.line(Y=np.array([0]), X=np.array([0]))
        
        if (self.config['optimizer'] == 'Adam'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr = self.config['learning_rate'],
                                          weight_decay=self.config['weight_decay'])
        else:
            raise ValueError("optimizer not support.")
            
        self.decay_rate = 0

    def run(self):
        # optionally resume from a checkpoint
        resume = self.config['resume']
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                self.config['start_epoch'] = checkpoint['epoch']
                self.config['best_val_prec1'] = checkpoint['best_val_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))

        torch.backends.cudnn.benchmark = True # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                                              # If this is set to false, uses some in-built heuristics that might not always be fastest.

        # train
        print("start training")
        for epoch in range(self.config['start_epoch'], self.config['epochs']):
            self._train(epoch)

        
    def _train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (img, target, _) in enumerate(self.train_dataloader):
            # calculate the iteration number throughout whole training process
            total_iter = epoch * len(self.train_dataloader) + i
            
            # learning rate schedule
            self._adjust_learning_rate(total_iter, self.config['decay_lr_freq'])
            
            # measure data loading time
            data_time.update(time.time() - end)
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(img)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)
        
            # measure accuracy and record loss
            prec1, prec5 = self._accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], img.size(0))
            top1.update(prec1[0], img.size(0))
            top5.update(prec5[0], img.size(0))

            # compute gradient and do one optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # visulization
            self.viz.line(
                Y=np.array([losses.val]),
                X=np.array([total_iter]),
                win=self.loss_viz,
                update="append"
            )
            
            self.viz.line(
                Y=np.array([top1.val]),
                X=np.array([total_iter]),
                win=self.acc_viz,
                update="append"
            )

            if i % self.config['print_freq'] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(self.train_dataloader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            if i != 0 and i % self.config['validate_freq'] == 0:
                prec1 = self._validate(self.val_dataloader)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > self.config['best_val_prec1']
                self.config['best_val_prec1'] = max(prec1, self.config['best_val_prec1'])
                self._save_checkpoint({
                    'epoch': epoch,
                    'iter' : i,
                    'state_dict': self.model.state_dict(),
                    'best_val_prec1': self.config['best_val_prec1'],
                    'optimizer' : self.optimizer.state_dict(),
                    'loss' : losses,
                    'top1' : top1,
                    'top5' : top5,
                    'config': self.config
                }, is_best)
                
            if i != 0 and i % self.config['save_freq'] == 0:
                self._save_checkpoint({
                    'epoch': epoch,
                    'iter' : i,
                    'state_dict': self.model.state_dict(),
                    'best_val_prec1': self.config['best_val_prec1'],
                    'optimizer' : self.optimizer.state_dict(),
                    'loss' : losses,
                    'top1' : top1,
                    'top5' : top5,
                    'config': self.config
                }, is_best=False)

    def _validate(self, val_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (img, target) in enumerate(self.val_dataloader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(img, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self._accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], img.size(0))
            top1.update(prec1[0], img.size(0))
            top5.update(prec5[0], img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.config['print_freq'] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg


    def _save_checkpoint(self, state, is_best):
        epoch = state['epoch']
        iteration = state['iter']
        filename = self.config['arch'] + '_checkpoint_epoch_' + str(epoch) + '_iter_' + str(iteration) + '.pth.tar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.config['arch'] + '_checkpoint_epoch_' + str(epoch) + '_iter_' + str(iteration) + '_model_best.pth.tar')

        
    def _adjust_learning_rate(self, total_iter, decay_lr_freq):
        """Sets the learning rate to the initial LR decayed by 10 every 1e4 iterations"""
        decay_rate = total_iter // decay_lr_freq
        if (decay_rate != self.decay_rate):
            self.decay_rate = decay_rate
            print('adjust learning rate, decay_rate = %d' % self.decay_rate)
            new_learning_rate = self.config['learning_rate'] * (0.1 ** decay_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_learning_rate
        
        
    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_trainer(train_dataloader, val_dataloader, model, criterion, config):
    return Trainer(train_dataloader, val_dataloader, model, criterion, config)