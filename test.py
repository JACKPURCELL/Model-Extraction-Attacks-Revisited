import numpy as np
import torch
import torch.nn as nn

def calc_entropy(input_tensor):
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy

def calc_entropy_2(input_tensor):
    lsm = nn.Softmax()
    input_tensor = lsm(input_tensor)
    criterion  = nn.CrossEntropyLoss()
    return criterion(input_tensor,input_tensor)


def entropy(y_pred_prob,indices,n_samples):

    origin_index = torch.tensor(indices)
    entropy= -torch.nansum(torch.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = torch.argmax(y_pred_prob, axis=1)
    eni = torch.column_stack((origin_index,
                           entropy,
                           pred_label))

    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:, 0].astype(int)[:n_samples]



unlabel_dataset_indices=_unlabel_dataset.indices
def update_unlabel(_unlabel_dataset.indices):
    # unlabel_dataset_indices:input
    _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),
                                    hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api,transform = 'mixmatch'),
                                unlabel_dataset_indices)

    _unlabel_dataloader = DataLoader(dataset=_unlabel_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,drop_last=True)
    unlabel_iterator = itertools.cycle(_unlabel_dataloader)
    
    for i in range(len(unlabel_iterator)):
        inputs_u, _, _, _ = next(unlabel_iterator)
        
        with torch.no_grad():
            outputs_u = forward_fn(inputs_u)
            outputs_u_total = torch.cat(outputs_u_total,outputs_u)
        
    new_label_indices = entropy(outputs_u_total,args.n_samples)

    for i in indices:
        unlabel_dataset_indices.remove(i)
        
    _new_label_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api),
                                new_label_indices)
    _unlabel_dataset = Subset(RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir=args.hapi_data_dir,hapi_info=args.hapi_info,api=args.api),
                                unlabel_dataset_indices)
    _new_label_dataloader = DataLoader(dataset=_label_dataset,
                    batch_size=args.batch_size,
                    shuffle=shuffle,sampler=sampler,
                    num_workers=args.num_workers,drop_last=True)
    new_label_iterator = itertools.cycle(_new_label_dataloader)
    
    for i in range(len(new_label_iterator)):
        _input, _label, _soft_label, hapi_label  = next(new_label_iterator)
        _input = _input.cuda()
        _soft_label = _soft_label.cuda()
        _label = _label.cuda()
        hapi_label = hapi_label.cuda()
        _output = forward_fn(_input)
        if label_train:
            loss = loss_fn( _label=hapi_label, _output=_output)
        else:
            loss = loss_fn( _soft_label=_soft_label, _output=_output)
        if backward_and_step:
            optimizer.zero_grad() 
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            
    
                            
                            
 

    
    
    
    
    
    
    
    
    
    
    
    len_loader_train = len(loader_train)
    total_iter = (epochs - resume) * len_loader_train

    logger = MetricLogger()
    if mixmatch:
        logger.create_meters(loss=None)
    else:
        logger.create_meters(   gt_acc1=None, 
                          hapi_loss=None, hapi_acc1=None)
    if resume and lr_scheduler:
        for _ in range(resume):
            lr_scheduler.step()
    iterator = range(resume, epochs)
    if verbose and output_freq == 'epoch':
        header: str = '{blue_light}{0}: {reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(header), 30) + get_ansi_len(header))
        iterator = logger.log_every(range(resume, epochs),
                                    header=print_prefix,
                                    tqdm_header='Epoch',
                                    indent=indent)
    for _epoch in iterator:
        _epoch += 1
        logger.reset()
        loader_epoch = loader_train
        if verbose and output_freq == 'iter':
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            loader_epoch = logger.log_every(loader_train, header=header,
                                            tqdm_header='Batch',
                                            indent=indent)
# a = torch.tensor([1,2,3,4,5,6,7,8,9,10],dtype=torch.float32)
# print(calc_entropy(a))
# print(calc_entropy_2(a))
# print(entropy(a))

b = torch.tensor(([1,1,1,1,1,1,1,1,1,1],[1,2,3,4,5,6,7,8,9,10]),dtype=torch.float32)
# print(calc_entropy(b))
# print(calc_entropy_2(b))
print(entropy(b,10))

# b = torch.tensor([1,0,0],dtype=torch.float32)
# print(calc_entropy(b))
# print(calc_entropy_2(b))
# print(entropy(b))



def distillation(module: nn.Module, num_classes: int,
          epochs: int, optimizer, lr_scheduler,
        log_dir:str = 'runs/test', 
          grad_clip: float = 5.0, 
          print_prefix: str = 'Distill', start_epoch: int = 0, resume: int = 0,
          validate_interval: int = 1, save: bool = True,
          loader_train: torch.utils.data.DataLoader = None,
          loader_valid: torch.utils.data.DataLoader = None,
          unlabel_iterator = None,
        file_path: str = None,
          folder_path: str = None, suffix: str = None,
           main_tag: str = 'train', tag: str = '',

          verbose: bool = True, output_freq: str = 'iter', indent: int = 0,
          change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch',
          backward_and_step: bool = True, 
          mixmatch: bool = False,label_train: bool=False,api=False,task='sentiment',
          **kwargs):
    r"""Train the model"""
    if epochs <= 0:
        return
   
    forward_fn =  module.__call__



    writer = SummaryWriter(log_dir=log_dir)
    validate_fn = dis_validate 


    scaler: torch.cuda.amp.GradScaler = None

    best_validate_result = (0.0, float('inf'))
    best_acc = 0.0
    # if validate_interval != 0:
    #     best_validate_result = validate_fn(module=module,loader=loader_valid, 
    #                                        writer=None, tag=tag, _epoch=start_epoch,
    #                                        verbose=verbose, indent=indent, num_classes=num_classes,
    #                                        label_train=label_train,api=api,**kwargs)
    #     best_acc = best_validate_result[0]

    params: list[nn.Parameter] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group['params'])
    len_loader_train = len(loader_train)
    total_iter = (epochs - resume) * len_loader_train

    logger = MetricLogger()
    if mixmatch:
        logger.create_meters(loss=None)
    else:
        logger.create_meters(   gt_acc1=None, 
                          hapi_loss=None, hapi_acc1=None)
    if resume and lr_scheduler:
        for _ in range(resume):
            lr_scheduler.step()
    iterator = range(resume, epochs)
    if verbose and output_freq == 'epoch':
        header: str = '{blue_light}{0}: {reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(header), 30) + get_ansi_len(header))
        iterator = logger.log_every(range(resume, epochs),
                                    header=print_prefix,
                                    tqdm_header='Epoch',
                                    indent=indent)
    for _epoch in iterator:
        _epoch += 1
        logger.reset()
        loader_epoch = loader_train
        if verbose and output_freq == 'iter':
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            loader_epoch = logger.log_every(loader_train, header=header,
                                            tqdm_header='Batch',
                                            indent=indent)
        if change_train_eval:
            module.train()
        activate_params(module, params)




        # if _epoch < 10000:
        #     mode = 'train_STU' #kl loss / return raw data
        #     print(_epoch,mode)
        # elif _epoch >= 10000:
        #     mode = 'train_ADV_STU'  #kl loss / return adv data
        #     print(_epoch,mode)


        for i, data in enumerate(loader_epoch):
            _iter = _epoch * len_loader_train + i
            match task:
                case 'emotion':
                    if mixmatch:
                        mixed_input, mixed_target, batch_size = mixmatch_get_data(data,forward_fn,unlabel_iterator)

                        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
                        mixed_input = list(torch.split(mixed_input, batch_size))
                        mixed_input = interleave_fn(mixed_input, batch_size)

                        logits = [forward_fn(mixed_input[0])]
                        for input in mixed_input[1:]:
                            logits.append(forward_fn(input))

                        # put interleaved samples back
                        logits = interleave_fn(logits, batch_size)
                        logits_x = logits[0]
                        logits_u = torch.cat(logits[1:], dim=0)

                        loss = loss_fn(outputs_x = logits_x, targets_x = mixed_target[:batch_size], outputs_u = logits_u, targets_u = mixed_target[batch_size:], iter = _iter)

                    # elif adaptive:
                    #     _input, _label, _soft_label, hapi_label  = data
                    #     _input = _input.cuda()
                    #     _soft_label = _soft_label.cuda()
                    #     _label = _label.cuda()
                    #     hapi_label = hapi_label.cuda()
                    #     _output = forward_fn(_input)
                    #     if label_train:
                    #         loss = loss_fn( _label=hapi_label, _output=_output)
                    #     else:
                    #         loss = loss_fn( _soft_label=_soft_label, _output=_output)
        
                    else:
                        _input, _label, _soft_label, hapi_label  = data
                        _input = _input.cuda()
                        _soft_label = _soft_label.cuda()
                        _label = _label.cuda()
                        hapi_label = hapi_label.cuda()
                        _output = forward_fn(_input)
                        _output
                        if label_train:
                            loss = loss_fn( _label=hapi_label, _output=_output)
                        else:
                            loss = loss_fn( _soft_label=_soft_label, _output=_output)

                case 'sentiment':
                    input_ids, token_type_ids, attention_mask, _label, _soft_label, hapi_label  = data
                    input_ids = input_ids.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    _label = _label.cuda()
                    _soft_label = _soft_label.cuda()
                    hapi_label = hapi_label.cuda()

                    _output = forward_fn(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
                    
                    if label_train:
                        loss = loss_fn( _label=hapi_label, _output=_output)
                    else:
                        loss = loss_fn( _soft_label=_soft_label, _output=_output)
                    
            if backward_and_step:
                optimizer.zero_grad()
                #backward the weights 
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

          

            if lr_scheduler and lr_scheduler_freq == 'iter':
                lr_scheduler.step()
                
               
            if mixmatch:
                 logger.update(n=batch_size, loss=float(loss))
            else:    
                match task:
                    case 'sentiment':
                        _output = _output[:,:2]
                        new_num_classes = 2
                    case 'emotion':
                        new_num_classes = num_classes
                hapi_acc1, hapi_acc5 = accuracy_fn(
                    _output, hapi_label, num_classes=new_num_classes, topk=(1, 5))
                gt_acc1, gt_acc5 = accuracy_fn(
                    _output, _label, num_classes=new_num_classes, topk=(1, 5))
                batch_size = int(_label.size(0)) 
                logger.update(n=batch_size, gt_acc1=gt_acc1,  
                            hapi_loss=float(loss), hapi_acc1=hapi_acc1)
        optimizer.zero_grad()
        if lr_scheduler and lr_scheduler_freq == 'epoch':
            lr_scheduler.step()
        if change_train_eval:
            module.eval()
        activate_params(module, [])
        if mixmatch:
            loss=(logger.meters['loss'].global_avg)
            if writer is not None:
                writer.add_scalars(main_tag='loss/' + main_tag,
                            tag_scalar_dict={tag: loss}, global_step=_epoch + start_epoch)        
        else:
            gt_acc1, hapi_loss, hapi_acc1 = (
                    logger.meters['gt_acc1'].global_avg,
                    logger.meters['hapi_loss'].global_avg,
                    logger.meters['hapi_acc1'].global_avg)
            if writer is not None:
                writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                            tag_scalar_dict={tag: gt_acc1}, global_step=_epoch + start_epoch)        
                writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                            tag_scalar_dict={tag: hapi_loss}, global_step=_epoch + start_epoch)
                writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                        tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch + start_epoch)
            
        if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
            validate_result = validate_fn(module=module,
                                          num_classes=num_classes,
                                          loader=loader_valid,
                                          writer=writer, tag=tag,
                                          _epoch=_epoch + start_epoch,
                                          verbose=verbose, indent=indent,
                                          label_train=label_train,
                                          api=api,task=task,
                                          **kwargs)
            cur_acc = validate_result[0]
            if cur_acc >= best_acc:
                best_validate_result = validate_result
                if verbose:
                    prints('{purple}best result update!{reset}'.format(
                        **ansi), indent=indent)
                    prints(f'Current Acc: {cur_acc:.3f}    '
                           f'Previous Best Acc: {best_acc:.3f}',
                           indent=indent)
                best_acc = cur_acc
                if save:
                    save_fn(file_path=file_path, folder_path=folder_path,
                            suffix=suffix, verbose=verbose)
            if verbose:
                prints('-' * 50, indent=indent)
    module.zero_grad()
    return best_validate_result