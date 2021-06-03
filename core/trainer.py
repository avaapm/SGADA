import os
import sys
sys.path.append(os.path.abspath('.'))
from logging import getLogger
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from utils.utils import AverageMeter, save

logger = getLogger('sgada.trainer')


def train_target_cnnP_domain(
    source_cnn,
    target_cnn,
    discriminator,
    criterion,
    optimizer,
    d_optimizer,
    source_train_loader,
    target_train_loader,
    target_test_loader,
    args=None
):
    validation = validate(source_cnn, target_test_loader, criterion, args=args)
    log_source = 'Source/Acc {:.3f} '.format(validation['avgAcc'])

    best_score = None
    best_class_score = None
    for epoch_i in range(1, 1 + args.epochs):
        start_time = time()
        training = adversarial_domain(
            source_cnn, target_cnn, discriminator,
            source_train_loader, target_train_loader, target_test_loader,
            criterion, criterion,
            optimizer, d_optimizer,trial, best_score, best_class_score, epoch_i, args=args
        )
        best_score = training['best_score']
        best_class_score = training['best_class_score']
        n_iters = training['n_iters']
        validation = validate(
            target_cnn, target_test_loader, criterion, args=args)
        clsNames = validation['classNames']
        log = 'Epoch {}/{} '.format(epoch_i, args.epochs)
        log += 'D/Loss {:.3f} Target/Loss {:.3f} '.format(
            training['d/loss'], training['target/loss'])
        log += '[Val] Target/Loss {:.3f} Target/Acc {:.3f} '.format(
            validation['loss'], validation['acc'])
        log += log_source
        log += 'Time {:.2f}s'.format(time() - start_time)
        logger.info(log)

        # save
        is_best = (best_score is None or validation['avgAcc'] > best_score)
        best_score = validation['avgAcc'] if is_best else best_score
        best_class_score = validation['classAcc'] if is_best else best_class_score
        state_dict = {
            'model': target_cnn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_i,
            'val/acc': best_score,
        }
        save(args.logdir, state_dict, is_best)
        for cls_idx, clss in enumerate(clsNames):
            logger.info('{}: {}'.format(clss, validation['classAcc'][cls_idx]))
        logger.info('Current val. acc.: {}'.format(validation['avgAcc']))
        logger.info('Best val. acc.: {}'.format(best_score))
        classWiseDict = {}
        for cls_idx, clss in enumerate(clsNames):
            classWiseDict[clss] = validation['classAcc'][cls_idx].item()
    
    return best_score, best_class_score, clsNames


def adversarial_domain(
    source_cnn, target_cnn, discriminator,
    source_loader, target_loader, target_test_loader,
    criterion, d_criterion,
    optimizer, d_optimizer, best_score, best_class_score, epoch_i, args=None
):
    source_cnn.eval()
    target_cnn.encoder.train()
    discriminator.train()

    best_score = best_score
    best_class_score = best_class_score

    losses, d_losses = AverageMeter(), AverageMeter()
    n_iters = min(len(source_loader), len(target_loader))
    valSteps = n_iters//args.num_val
    valStepsList = [valSteps+(x*valSteps) for x in range(args.num_val)]
    vals = valStepsList[:-1]
    source_iter, target_iter = iter(source_loader), iter(target_loader)
    for iter_i in range(n_iters):
        source_data, source_target = source_iter.next()
        target_data, target_target, target_conf, target_domain, target_domain_conf = target_iter.next()
        source_data = source_data.to(args.device)
        target_data = target_data.to(args.device)
        target_target = target_target.to(args.device)
        target_conf = target_conf.to(args.device)
        target_domain = target_domain.to(args.device)
        target_domain_conf = target_domain_conf.to(args.device)
        bs = source_data.size(0)

        D_input_source = source_cnn.encoder(source_data)
        D_input_target = target_cnn.encoder(target_data)
        D_target_source = torch.tensor(
            [0] * bs, dtype=torch.long).to(args.device)
        D_target_target = torch.tensor(
            [1] * bs, dtype=torch.long).to(args.device)

        # train Discriminator
        D_output_source = discriminator(D_input_source)
        D_output_target = discriminator(D_input_target)
        D_output = torch.cat([D_output_source, D_output_target], dim=0)
        D_target = torch.cat([D_target_source, D_target_target], dim=0)
        d_loss = criterion(D_output, D_target)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        d_losses.update(d_loss.item(), bs)

        # train Target
        D_input_target = target_cnn.encoder(target_data)
        D_output_target = discriminator(D_input_target)
        D_output_target_P = target_cnn.classifier(D_input_target)
        lossT = criterion(D_output_target, D_target_source)
        validSource = (target_domain == 0) & (target_conf >= args.thr)
        validMaskSource = validSource.nonzero(as_tuple=False)[:, 0]
        validTarget = (target_domain == 1) & (target_domain_conf <= args.thr_domain) & (target_conf >= args.thr)
        validMaskTarget = validTarget.nonzero(as_tuple=False)[:, 0]
        validIndexes = torch.cat((validMaskSource, validMaskTarget), 0)
        lossP = criterion(D_output_target_P[validIndexes], target_target[validIndexes])
        loss = lossT + args.lam*lossP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), bs)
        if iter_i in vals:
            validation = validate(
                target_cnn, target_test_loader, 
                criterion, args=args)
            clsNames = validation['classNames']
            is_best = (best_score is None or validation['avgAcc'] > best_score)
            best_score = validation['avgAcc'] if is_best else best_score
            best_class_score = validation['classAcc'] if is_best else best_class_score
            state_dict = {
                'model': target_cnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_i,
                'val/acc': best_score,
            }
            save(args.logdir, state_dict, is_best)
            logger.info('Epoch_{} Iter_{} nstep: {}'.format(epoch_i, iter_i, nstep))
            for cls_idx, clss in enumerate(clsNames):
                logger.info('{}: {}'.format(clss, validation['classAcc'][cls_idx]))
            logger.info('Current val. acc.: {}'.format(validation['avgAcc']))
            logger.info('Best val. acc.: {}'.format(best_score))
            classWiseDict = {}
            for cls_idx, clss in enumerate(clsNames):
                classWiseDict[clss] = validation['classAcc'][cls_idx].item()
            source_cnn.eval()
            target_cnn.encoder.train()
            discriminator.train()

    return {'d/loss': d_losses.avg, 'target/loss': losses.avg, 'best_score': best_score, 'best_class_score': best_class_score, 'n_iters': n_iters}


def step(model, data, target, criterion, args):
    data, target = data.to(args.device), target.to(args.device)
    output = model(data)
    loss = criterion(output, target)
    return output, loss


def validate(model, dataloader, criterion, args=None):
    model.eval()
    losses = AverageMeter()
    targets, probas = [], []
    if args.classInfo == None:
        classes = torch.unique(torch.tensor(dataloader.dataset.targets))
        classNames = dataloader.dataset.classes
    else:
        classes = args.classInfo['classes']
        classNames = args.classInfo['classNames']
    class_acc = torch.zeros(len(classes))
    class_len = torch.zeros(len(classes))
    acc_ev = 0
    with torch.no_grad():
        for iter_i, (data, target) in enumerate(dataloader):
            bs = target.size(0)
            output, loss = step(model, data, target, criterion, args)
            pred_cls = output.data.max(1)[1]
            acc_ev += pred_cls.cpu().eq(target.data).cpu().sum()
            for class_idx, class_id in enumerate(classes):
                idxes = torch.nonzero(target==class_id.to(target.device), as_tuple=False)
                class_acc[class_idx] += pred_cls[idxes].cpu().eq(target[idxes].data).cpu().sum()
                class_len[class_idx] += len(idxes)
            output = torch.softmax(output, dim=1)
            losses.update(loss.item(), bs)
            targets.extend(target.cpu().numpy().tolist())
            probas.extend(output.cpu().numpy().tolist())
    probas = np.asarray(probas)
    preds = np.argmax(probas, axis=1)
    acc = accuracy_score(targets, preds)
    class_acc /= class_len
    avgAcc = 0.0
    for i in range(len(class_acc)):
        avgAcc += class_acc[i]
    avgAcc = avgAcc / len(class_acc)
    return {
        'loss': losses.avg, 'acc': acc, 'avgAcc': avgAcc, 'classAcc': class_acc, 'classNames': classNames,
    }