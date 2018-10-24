import pandas as pd
import numpy as np
import torch
import time
import os

from tqdm import tqdm_notebook


class CSVLogger:
    """Log data to file in csv format
    
    Parameters:
        path: str
            filename to write data to
        columns: list
            columns names to write
        log_time: bool
            if True add 'time' as first column
    """
    
    def __init__(self, path, columns, log_time=False):
        self.path = path
        self.columns = columns
        self.log_time = log_time
        self._init()
    
    def write(self, **kwargs):
        line = self._constructLineFromDict(kwargs)
        self._writeLine(line)
    
    def resetClock(self):
        self.start_time = self._getAbsoluteTime()
        
    def _readHistory(self):
        return pd.read_csv(self.path)
    
    def _init(self):
        header = self._constructHeader()
        self._writeLine(header, "w")
    
    def _writeLine(self, line, mode="a"):
        with open(self.path, mode) as f:
            f.write(line + "\n")
            
    def _constructLineFromDict(self, kwargs):
        line = ""
        for field in self.columns:
            if field in kwargs:
                line += str(kwargs[field])
            line += ","
        line = line[:-1]  # remove last comma
        line = self._addTimeIfNedeed(line)
        return line
    
    def _constructHeader(self):
        header = ",".join(self.columns)
        if self.log_time:
            header = "time," + header
        return header
    
    def _getAbsoluteTime(self):
        return int(time.clock_gettime(time.CLOCK_MONOTONIC))
    
    def _getRelativeTime(self):
        try:
            return self._getAbsoluteTime() - self.start_time
        except AttributeError:
            raise Exception("Call 'resetClock' before logging with time tracking")
        
    def _addTimeIfNedeed(self, line):
        if self.log_time:
            line = str(self._getRelativeTime()) + "," + line
        return line


class BestLastCheckpointer:
    """Save best and last versions of dict passed to update
    method based on loss value of current iteration
    
    Parameters:
        path: str
    """
    
    def __init__(self, path):
        self.path = path
        self.loss_best_value = None
        
    def update(self, value, **kwargs):
        self.save("last", **kwargs)
        if self.loss_best_value is None or self.loss_best_value > value:
            self.loss_best_value = value
            self.save("best", **kwargs)
        
    def save(self, which, **kwargs):
        path = self._makePath(which)
        save_checkpoint(path, **kwargs)
    
    def load(self, which, **kwargs):
        path = self._makePath(which)
        return load_checkpoint(path, **kwargs)
        
    def _makePath(self, which):
        if which not in ("best", "last"):
            raise NotImplementedError("only 'best' and 'last' are supported")
        return self.path + "_" + which + ".pth"


def train_epoch_fn(model, dataloader, optim, loss_fn, verbose=0, loss_file=None):
    """Perform training for one epoch
    
    Parameters:
        model: object
            has a '__call__' method, which accept all keyword arguments 
            in a batch and returns a dict.
        
        dataloader: iterable
            yields a single batch in each iteration. each batch is a dict 
            with at least one field 'N' - number of elements in a batch.
            
        optim: pytorch optimizer
            has 'step' and 'zero_grad' methods with no parameters.
            logic on scheduling of learning rate, etc, must be implemented
            outside this function
            
        loss_fn: function
            returns a zero dimensional pytorch tensor
            
        verbose: int
            0 - no progress output
            1 - output progress bar and current epoch
            
        loss_file: str or None
            filename to write loss after each iteration
    
    """
    model.train()
    dataloader = wrap_dataloader(dataloader, verbose)
    epoch_loss = 0.0
    N = 0
    
    if loss_file is not None:
        f = open(loss_file, "a")
    
    for inputs in dataloader:
        inputs = place_to(model.device, inputs)
        inputs_size = len(inputs[list(inputs.keys())[0]])
        
        def closure():
            optim.zero_grad()
            
            outputs = model(**inputs)
            
            loss = loss_fn(**outputs, **inputs)
            loss.backward()
            
            free_tensors(outputs)
            return loss
        
        loss = optim.step(closure)
        
        if loss_file is not None:
            f.write("{}\n".format(loss.item()))
        
        epoch_loss += loss.item() * inputs_size
        N += inputs_size
        
        free_tensors(inputs)
        del loss
        
    if loss_file is not None:
        f.close()   
    
    torch.cuda.empty_cache()
    return epoch_loss / N


def eval_fn(model, dataloader, metrics, mean=True, verbose=0):
    """Perform evaluation on dataset. Usually used during training.
    
    Parameters:
        model: object
            has a '__call__' method, which accept all keyword arguments 
            in a batch and returns a dict. model must have a device attribute
        
        dataloader: iterable
            yields a single batch in each iteration. each batch is a dict 
            with at least one field 'N' - number of elements in a batch.
            
        metrics: dict of functions
            each function must return a zero dimensional pytorch tensor if mean is True
            or a one dimensional tensor with length of batchsize if mean is False
            
        mean: bool, True by default
            see parameter 'metrics' for more information
            
        verbose: int
            0 - no progress output
            1 - output progress bar and current epoch
    
    """
    model.eval()
    dataloader = wrap_dataloader(dataloader, verbose)
    if mean:
        epoch_metrics = { metric_name: 0.0 for metric_name in metrics }
        N = { metric_name: 0 for metric_name in metrics }
    else:
        epoch_metrics = { metric_name:[] for metric_name in metrics }
    
    with torch.no_grad():
        for inputs in dataloader:
            inputs = place_to(model.device, inputs)
            inputs_size = len(inputs[list(inputs.keys())[0]])
            outputs = model(**inputs)

            for metric_name, metric in metrics.items():
                value = metric(**outputs, **inputs)
                
                if mean:
                    epoch_metrics[metric_name] += value * inputs_size
                    N[metric_name] += inputs_size
                else:
                    epoch_metrics[metric_name].append(value)

            free_tensors(outputs)
            free_tensors(inputs)
        
    if mean:
        for metric_name in metrics:
            epoch_metrics[metric_name] /= N[metric_name]
    else:
        for metric_name in metrics:
            epoch_metrics[metric_name] = np.stack(epoch_metrics[metric_name])
    torch.cuda.empty_cache()
    return epoch_metrics


def wrap_dataloader(dataloader, verbose):
    """Wrap pytorch dataloader in tqdm progress bar if verbose > 0
    """
    if verbose > 0:
        dataloader = tqdm_notebook(dataloader)
    return dataloader


def freeze(module, include="all"):
    """Freeze parameters of model by setting requires_grad to False
    
    Parameters:
        model: pytorch Module
        include: str or tuple
            determine which weights to freeze
            if str it must be 'all' or 'none', if tuple it must contains types
            of modules what to freeze.
    """
    set_requires_grad(module, include, False)


def unfreeze(module, include="all"):
    """Unfreeze parameters of model by setting requires_grad to False
    
    Parameters:
        model: pytorch Module
        include: str or tuple
            determine which weights to freeze
            if str it must be 'all' or 'none', if tuple it must contains types
            of modules what to unfreeze.
    """
    set_requires_grad(module, include, True)


def set_requires_grad(module, include, value):
    if not isinstance(include, (str, tuple)):
        raise TypeError("'include' argument must be str or tuple")
    if include == "none":
        return
    for m in module.modules():
        if include == "all" or isinstance(m, include):
            for p in m.parameters():
                p.requires_grad = value


def save_checkpoint(path, **kwargs):
    """Save keyword arguments as pytorch state dict
    
    Arguments 'model' and 'optim' must have a 'state_dict' method.
    """
    state_dict = {}
    for key in kwargs:
        if key in ("model", "optim"):
            state_dict[key] = kwargs[key].state_dict()
        else:
            state_dict[key] = kwargs[key]
    torch.save(state_dict, path)


def load_checkpoint(path, **kwargs):
    """If 'model' or/and 'optim' in kwargs, then load state dict from file
    Returns not matched keys as a dict(may be empty).
    """
    map_location = None
    if "model" in kwargs:
        map_location = kwargs["model"].device.type
        
    state_dict = torch.load(path, map_location=map_location)
    res_state_dict = {}
    for key in state_dict:
        if key in kwargs and key in ("model", "optim"):
            kwargs[key].load_state_dict(state_dict[key])
        else:
            res_state_dict[key] = state_dict[key]
    return res_state_dict


def get_learning_rate(optim, groups=None):
    """Get current learning rate of pytorch optimizer
    
    Parameters:
        oprim: pytorch optimizer
        groups: None, int, list of int
            if None, then return learninng rates of all param groups
            if int, then return learning rate of that group
            if list, then return list of learning rates of these groups
    
    Returns: float or list of float
    """
    state_dict = optim.state_dict()
    param_groups = state_dict["param_groups"]
    if groups is None:
        return [pg["lr"] for pg in param_groups]
    elif isinstance(groups, int):
        return param_groups[groups]["lr"]
    elif isinstance(groups, list):
        return [param_groups[g]["lr"] for g in groups]
    else:
        raise NotImplementedError()


def set_learning_rate(optim, lr, groups=None):
    """Set learning rate of pytorch optimizer
    
    Parameters:
        oprim: pytorch optimizer
        lr: float or list of float
            if float, then learning rates of all param groups would be set to
            that value
            if list, then the length must be equal to length of groups
        groups: None or int or list of int
            if None, then return learninng rates of all param groups
            if int, then return learning rate of that group
            if list, then return list of learning rates of these groups
    """
    state_dict = optim.state_dict()
    param_groups = state_dict["param_groups"]
    
    if isinstance(groups, int):
        param_groups[groups]["lr"] = lr
    else:
        if groups is None:
            groups = list(range(len(param_groups)))
        if isinstance(lr, float):
            lr = [lr] * len(groups)
    
        for lr_, pg in zip(lr, groups):
            param_groups[pg]["lr"] = lr_

    optim.load_state_dict(state_dict)


def interruptible(iterable, iterablename="auto"):
    """Wrap an iterable so that it can be stopped from any other script.
    
    Parameters:
        iterable: 
            objects that supports 'for' iteration
        iterablename: str
            name which may be used to stop iteration for certain loop
            if 'auto', then name will be generated and printed at start
            of each iteration
            
    Returns: wrapped iterable object
    """
    if iterablename == "auto":
        letters = list("qwertyuiopasdfghjklzxcvbnm0123456789")
        iterablename = "".join(np.random.choice(letters, size=5))
        
    filename = os.path.join("/tmp", iterablename)
            
    with open(filename, "w") as f:
        f.write("init\n")
        
    for item in iterable:
        print("Use this name to stop iteration:", iterablename)
        
        yield item
        
        with open(filename, "r") as f:
            cmd = f.readlines()[-1].strip()
        if cmd == "interrupt":
            print("Stop iteration")
            break
    
    os.system("rm {}".format(filename))


def interrupt(iterablename):
    """Interrupt a loop by its name
    """
    path = os.path.join("/tmp", iterablename)
    
    if not os.path.exists(path):
        raise Exception("Loop not found")
        
    with open(path, "a") as f:
        f.write("interrupt\n")


def place_to(device, tensors):
    """Place dict of tensors to specified device
    
    Parameters:
        device: pytorch device object
        tensors: dict
            tensors to place to device
    
    Returns:
        tensors: dict
            keys are the same as in argument except moved to device
    """
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            tensors[name] = tensor.to(device)
    return tensors


def free_tensors(tensors):
    """Delete tensors from memory
    
    Parameters:
        tensors: dict
            all tensors would be deleted with del operator
    """
    to_delete = []
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            to_delete.append(name)
    for name in to_delete:
        del tensors[name]
