import torch as ch
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchvision.utils import make_grid

from .tools import helpers
from .tools.helpers import AverageMeter, ckpt_at_epoch, has_attr
from .tools import constants as consts
import helper as hp
import dill 
import os
import time
import warnings

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

try:
    from apex import amp
except Exception as e:
    warnings.warn('Could not import amp.')

def check_required_args(args, eval_only=False):
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
    """
    required_args_eval = ["adv_eval"]
    required_args_train = ["epochs", "out_dir", "adv_train",
        "log_iters", "lr", "momentum", "weight_decay"]
    adv_required_args = ["attack_steps", "eps", "constraint", 
            "use_best", "attack_lr", "random_restarts"]

    # Generic function for checking all arguments in a list
    def check_args(args_list):
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only: check_args(required_args_train)
    else: check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, 'custom_train_loss')
    has_custom_adv = has_attr(args, 'custom_adv_loss')
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError("Cannot use custom train loss \
            without a custom adversarial loss (see docs)")


def make_optimizer_and_schedule(args, model, checkpoint, params):
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`
        params (list|None) : a list of parameters that should be updatable, all
            other params will not update. If ``None``, update all params 

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    param_list = model.parameters() if params is None else params
    optimizer = SGD(param_list, args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.mixed_precision:
        model.to('cuda')
        model, optimizer = amp.initialize(model, optimizer, 'O1')

    # Make schedule
    schedule = None
    if args.custom_lr_multiplier == 'cyclic':
        eps = args.epochs
        lr_func = lambda t: np.interp([t], [0, eps*4//15, eps], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_multiplier:
        cs = args.custom_lr_multiplier
        periods = eval(cs) if type(cs) is str else cs
        if args.lr_interpolation == 'linear':
            lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep):
                for (milestone, lr) in reversed(periods):
                    if ep >= milestone: return lr
                return 1.0
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()
        
        if 'amp' in checkpoint and checkpoint['amp'] not in [None, 'N/A']:
            amp.load_state_dict(checkpoint['amp'])

        # TODO: see if there's a smarter way to do this
        # TODO: see what's up with loading fp32 weights and then MP training
        if args.mixed_precision:
            model.load_state_dict(checkpoint['model'])

    return optimizer, schedule

def eval_model(args, model, loader, store, dp_device_ids=[0], extra_model_kwargs={}, with_images=False):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object 
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
        extra_model_kwargs : dict, passed to model call in _model_loop
    """
    check_required_args(args, eval_only=True)
    start_time = time.time()

    if store is not None: 
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    writer = store.tensorboard if store else None

    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model, device_ids=dp_device_ids).to(ch.device(f'cuda:{dp_device_ids[0]}'))
    # model = ch.nn.DataParallel(model)

    prec1, _, nat_loss, _ = _model_loop(args, 'val', loader, 
                                        model, None, 0, False, writer, extra_model_kwargs, devices=dp_device_ids)

    adv_prec1_og_preds, adv_prec1, adv_loss, perception_distance = float('nan'), float('nan'), float('nan'), float('nan')
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, 'attack_lr') else None
        if with_images:
            adv_prec1, adv_prec1_og_preds, adv_loss, og_inputs, adversarial_inputs, original_preds, \
                adv_preds, true_labels, perception_distance = \
                    _model_loop(args, 'val', loader, model, None, 0, 
                        True, writer, extra_model_kwargs, return_images=True, devices=dp_device_ids)
        else:
            adv_prec1, adv_prec1_og_preds, adv_loss, perception_distance = _model_loop(args, 'val', loader, 
                                        model, None, 0, True, writer, extra_model_kwargs, devices=dp_device_ids)
    log_info = {
        'epoch':0,
        'nat_prec1':prec1,
        'adv_prec1':adv_prec1,
        'adv_prec1_og_preds':adv_prec1_og_preds,
        'perception_distance':perception_distance,
        'nat_loss':nat_loss,
        'adv_loss':adv_loss,
        'train_prec1':float('nan'),
        'train_loss':float('nan'),
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store: store[consts.LOGS_TABLE].append_row(log_info)
    if with_images:
        return log_info, og_inputs, adversarial_inputs, original_preds, adv_preds, true_labels
    else:
        return log_info

def train_model(args, model, loaders, *, checkpoint=None, dp_device_ids=None,
            store=None, update_params=None, disable_no_grad=False, extra_model_kwargs={}):
    """
    Main function for training a model. 

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a 
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do 
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_lr_multplier (str)
                If given, use a custom LR schedule, formed by multiplying the
                    original ``lr`` (format: [(epoch, LR_MULTIPLIER),...])
            lr_interpolation (str)
                How to drop the learning rate, either ``step`` or ``linear``,
                    ignored unless ``custom_lr_multiplier`` is provided.
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            custom_eps_multiplier (str, *required if adv_train or adv_eval*)
                If given, then set epsilon according to a schedule by
                multiplying the given eps value by a factor at each epoch. Given
                in the same format as ``custom_lr_multiplier``, ``[(epoch,
                MULTIPLIER)..]``
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            custom_accuracy (function)
                If given, should be a function that takes in model outputs
                and model targets and outputs a top1 and top5 accuracy, will 
                displayed instead of conventional accuracies
            regularizer (function, optional) 
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)` 
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        dp_device_ids (list|None) : if not ``None``, a list of device ids to
            use for DataParallel.
        store (cox.Store) : a cox store for logging training progress
        update_params (list) : list of parameters to use for training, if None
            then all parameters in the model are used (useful for transfer
            learning)
        disable_no_grad (bool) : if True, then even model evaluation will be
            run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
        extra_model_kwargs : dict
            passed to model call in _model_loop, useful for relative adversarial training
    """
    # Logging setup
    writer = store.tensorboard if store else None
    prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
    if store is not None: 
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    
    # Reformat and read arguments
    check_required_args(args) # Argument sanity check
    for p in ['eps', 'attack_lr', 'custom_eps_multiplier']:
        setattr(args, p, eval(str(getattr(args, p))) if has_attr(args, p) else None)
    if args.custom_eps_multiplier is not None: 
        eps_periods = args.custom_eps_multiplier
        args.custom_eps_multiplier = lambda t: np.interp([t], *zip(*eps_periods))[0]

    # Initial setup
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    # Put the model into parallel mode
    if not hasattr(model, "module"):
        model = ch.nn.DataParallel(model, device_ids=dp_device_ids).to(
            ch.device(f'cuda:{dp_device_ids[0]}'))

    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[prec1_key] if prec1_key in checkpoint \
            else _model_loop(args, 'val', val_loader, model, None, start_epoch-1, 
                args.adv_train, writer=None, extra_model_kwargs=extra_model_kwargs, 
                devices=dp_device_ids)[0]

    # Timestamp for training start time
    start_time = time.time()

    train_accs, train_losses, test_accs, test_accs_adv, test_losses = [], [], [], [], []

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        ## _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer, extra_model_kwargs=extra_model_kwargs)

        train_acc, test_acc = hp.eval_model(model, train_loader, dp_device_ids), \
            hp.eval_model(model, val_loader, dp_device_ids)

        train_prec1, train_loss = _model_loop(args, 'train', train_loader, 
                model, opt, epoch, args.adv_train, writer, 
                extra_model_kwargs=extra_model_kwargs, devices=dp_device_ids)
        last_epoch = (epoch == (args.epochs - 1))


        # evaluate on validation set
        sd_info = {
            'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1,
            'amp': amp.state_dict() if args.mixed_precision else None,
        }


        def save_checkpoint(filename):
            # ckpt_save_path = os.path.join(args.out_dir if not store else \
            #                               store.path, filename)
            ckpt_save_path = os.path.join(args.out_dir, filename)
            print (f'Checkpoint saved to: {ckpt_save_path}')
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad() 
            with ctx:
                prec1, _, nat_loss, _ = _model_loop(args, 'val', val_loader, model, 
                        None, epoch, False, writer, extra_model_kwargs=extra_model_kwargs, 
                        devices=dp_device_ids)

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and _model_loop(args, 'val', val_loader,
                    model, None, epoch, True, writer, extra_model_kwargs=extra_model_kwargs, 
                    devices=dp_device_ids)
            adv_prec1, _, adv_loss, _ = adv_val or (-1.0, -1.0, -1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)
            sd_info[prec1_key] = our_prec1

            if should_log or should_save_ckpt:
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                test_accs_adv.append(adv_prec1)
                train_losses.append(train_loss)
                test_losses.append(nat_loss)

            # log every checkpoint
            log_info = {
                'epoch':epoch + 1,
                'nat_prec1':prec1,
                'adv_prec1':adv_prec1,
                'nat_loss':nat_loss,
                'adv_loss':adv_loss,
                'train_prec1':train_prec1,
                'train_loss':train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch).format(args.random_seed))

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST.format(args.random_seed))
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST.format(args.random_seed))

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

    return model, train_accs, train_losses, test_accs, test_accs_adv, test_losses

def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer, extra_model_kwargs, 
                return_images=False, devices=[0]):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)
        extra_model_kwargs (dict) : dict of args to be passed in model call
        return_images (bool) : only useful when adv = True. 
                        Returns the original and adversarial images.
        eval_predicted_labels (bool) : only useful when adv = True. 
                        Evaluates accuracy wrt predicted labels on original input

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    losses = AverageMeter()
    top1_og_preds = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    og_inputs, adversarial_inputs, perception_distance = None, None, None
    original_predictions, adversarial_predictions, ground_truth = None, None, None

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps \
                if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    has_custom_val_loss = has_attr(args, 'custom_val_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
            else ch.nn.CrossEntropyLoss()
    val_criterion = args.custom_val_loss if has_custom_val_loss \
            else ch.nn.CrossEntropyLoss()
    
    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        attack_kwargs = hp.consolidate_attack_kwargs(args, adv_criterion, random_restarts, 
            eps, is_train, devices)

    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
       # measure data loading time
        # 2 cases: the usual cross entropy loss or the contrastive loss
        target = target.cuda(device=ch.device(f'cuda:{args.devices[0]}'), non_blocking=True)
        # target = target.to(ch.device(f'cuda:{args.devices[0]}'))
        
        inp, loss, output_original, output, final_inp, *extra = hp.consolidate_loss(inp, target, model, adv, 
            train_criterion if is_train else val_criterion, is_train, attack_kwargs, 
            extra_model_kwargs, devices=devices)

        if len(loss.shape) > 0: loss = loss.mean()

        model_logits = output[0] if (type(output) is tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top1_acc_og_preds = float('nan')
        top5_acc = float('nan')
        try:
            maxk = min(5, model_logits.shape[-1])
            if has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, target)
                # Accuracy wrt originally predcted labels
                prec_og_preds = args.custom_accuracy(model_logits, ch.argmax(output_original, 1))
            else:
                prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
                prec1, prec5 = prec1[0], prec5[0]
                # Accuracy wrt originally predcted labels
                prec_og_preds, = helpers.accuracy(model_logits, ch.argmax(output_original, 1), topk=(1,))
                prec_og_preds = prec_og_preds[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1, inp.size(0))
            top1_og_preds.update(prec_og_preds, inp.size(0))
            top5.update(prec5, inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
            top1_acc_og_preds = top1_og_preds.avg
        except:
            warnings.warn('Failed to calculate the accuracy.')

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term =  args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()
            opt.step()
        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR
        if is_train:
            desc = ('{2} Epoch:{0} ({loop_type}) | Loss {loss.avg:.4f} | '
                    '{1}1 {top1_acc:.3f} | {1}OG {top1_acc_og_preds:.3f} | Loss breakdown {loss_str} ||'.format(
                    epoch, prec, loop_msg, loop_type=loop_type,
                    loss=losses, top1_acc=top1_acc, top1_acc_og_preds=top1_acc_og_preds,
                    loss_str=str(train_criterion)))
        else:
            desc = ('{2} Epoch:{0} ({loop_type}) | Loss {loss.avg:.4f} | '
                    '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | {1}OG {top1_acc_og_preds:.3f}'.format(
                    epoch, prec, loop_msg, loop_type=loop_type, loss=losses, 
                    top1_acc=top1_acc, top5_acc=top5_acc, top1_acc_og_preds=top1_acc_og_preds))

        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)
        
        if return_images:
            og_inputs = ch.cat((og_inputs, inp.detach().cpu())) if og_inputs is not None else inp.detach().cpu()
            adversarial_inputs = ch.cat((adversarial_inputs, final_inp.detach().cpu())) \
                if adversarial_inputs is not None else final_inp.detach().cpu()
            original_predictions = ch.cat((original_predictions, ch.argmax(output_original, 1).detach().cpu())) \
                if original_predictions is not None else ch.argmax(output_original, 1).detach().cpu()
            adversarial_predictions = ch.cat((adversarial_predictions, ch.argmax(output, 1).detach().cpu())) \
                if adversarial_predictions is not None else ch.argmax(output, 1).detach().cpu()
            ground_truth = ch.cat((ground_truth, target.detach().cpu())) if ground_truth is not None else \
                target.detach().cpu()
            if args.perception_loss:
                assert len(extra) == 2
                perception_distance = ch.cat((perception_distance, 
                                              args.perception_loss(extra[0].detach().cpu(), extra[1].detach().cpu()))) \
                    if perception_distance is not None else args.perception_loss(extra[0].detach().cpu(), 
                                                                                 extra[1].detach().cpu())
            del extra
            ch.cuda.empty_cache()


        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1_wrt_preds', 'top1', 'top5']
        vals = [losses, top1_og_preds, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)

    if return_images:
        return top1.avg, top1_og_preds.avg, losses.avg, og_inputs, adversarial_inputs, \
            original_predictions, adversarial_predictions, ground_truth, perception_distance
    elif not is_train:
        return top1.avg, top1_og_preds.avg, losses.avg, perception_distance
    else:
        return top1.avg, losses.avg

