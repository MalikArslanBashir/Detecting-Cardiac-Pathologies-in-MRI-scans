from __future__                                import absolute_import, print_function

import numpy                                                           as np
try:
    from tensorflow.python.keras               import backend          as K
    from tensorflow.python.keras.callbacks     import Callback
except:
    import keras.backend                                               as K
    from keras.callbacks                       import Callback


class Stable(object):
    """
    learning rate updated as:
    lr(t) = lr(0)
    """
    def __init__(self, init_lr):
        self.init_lr            = init_lr
        
    def __call__(self, step):
        updated_lr              = self.init_lr
        
        return float(updated_lr)
    

class StepDecay(object):
    """
    learning rate updated as:
    lr(t) = lr(0) * gamma^power
    """
    def __init__(self, init_lr, gamma, step_size):        
        self.init_lr            = init_lr
        self.gamma              = gamma
        self.step_size          = step_size

    def __call__(self, step):
        power                   = np.floor((step) / self.step_size)
        updated_lr              = self.init_lr * (self.gamma ** power)
        
        return float(updated_lr)
    

class ExpoDecay(object):
    """
    learning rate updated as:
    lr(t) = lr(0)/(1 + gamma*t)
    """
    def __init__(self, init_lr,gamma):
        self.init_lr                = init_lr
        self.gamma                  = gamma
        
    def __call__(self, step):
        updated_lr                  = self.init_lr * 1./(1. + self.gamma * step)
        
        return float(updated_lr)


class Poly(object):
    """
    Polynomial Learning Rate Schedule
    update learning rate after each step/batch as:
    lr(t) = (lr(N) - lr(0)) * (t * 1/N) ^ power + lr(0)
    * starts at lr(0) and ends at lr(N)
    """
    def __init__(self, init_lr, poly_end_lr, poly_total_steps, power):
        self.poly_total_steps       = poly_total_steps
        self.poly_end_lr            = poly_end_lr
        self.power                  = power
        self.init_lr                = init_lr

    def __call__(self, step):
        updated_lr                  = (self.poly_end_lr - self.init_lr) * \
                                        (step * 1./(1. * self.poly_total_steps)) \
                                        ** self.power + self.init_lr
        
        return float(updated_lr)
    

class Triangular(object):
    """
    Triangular Cyclic Learning Rate Schedule
    lr(t) = (lr(N) - lr(0)) * 2/pi * |arcsin(sin( (pi*t)/(2*l) ))| + lr(0)
    """
    def __init__(self,init_lr,maxLR,batch_tri_step_size):
        self.batch_tri_step_size= batch_tri_step_size
        self.maxLR                  = maxLR
        self.init_lr                = init_lr

    def __call__(self, step):
        updated_lr                  = (self.maxLR - self.init_lr) * 2./np.pi \
                                        * np.abs(np.arcsin(np.sin((np.pi \
                                        * step)/(2. * self.batch_tri_step_size)))) + \
                                        self.init_lr
                                        
        return float(updated_lr)
    

class Triangular2(object):
    """
    Triangular Cyclic Learning Rate Schedule, with decay of 2 after each cycle
    factor = @(t) 1/(2^floor(t/(2*l)))
    lr(t) - (lr(N) - lr(0)) * factor(t) * 2/pi * |arcsin(sin( (pi*t)/(2*l) ))| + lr(0)
    """
    def __init__(self, init_lr, maxLR, batch_tri2_step_size):
        self.batch_tri2_step_size   = batch_tri2_step_size
        self.maxLR                  = maxLR
        self.init_lr                = init_lr

    def __call__(self, step):        
        factor                      = 1./(2.**(np.floor(step*1./(2. * \
                                        self.batch_tri2_step_size))))
        updated_lr                  = (self.maxLR - self.init_lr) * factor \
                                        * 2./np.pi * np.abs(np.arcsin(np.sin((np.pi \
                                        * step)/(2. * self.batch_tri2_step_size)))) + \
                                        self.init_lr
                                        
        return float(updated_lr)


class TriangularExp(object):
    """
    Triangular Cyclic Learning Rate Schedule, with spcified decay after each cycle
    factor = @(t/(2*l)))
    lr(t) - (lr(N) - lr(0)) * factor(t) * 2/pi * |arcsin(sin( (pi*t)/(2*l) ))| + lr(0)
    """
    def __init__(self, init_lr, maxLR, batch_tri_exp_gamma, batch_tri_exp_step_size):
        self.batch_tri_exp_step_size   = batch_tri_exp_step_size
        self.batch_tri_exp_gamma       = batch_tri_exp_gamma
        self.maxLR                  = maxLR
        self.init_lr                = init_lr

    def __call__(self, step):        
        power                        = np.floor(step /(2*self.batch_tri_exp_step_size)) 
        factor                      = self.batch_tri_exp_gamma**power
        updated_lr                  = (self.maxLR - self.init_lr) * factor \
                                        * 2./np.pi * np.abs(np.arcsin(np.sin((np.pi \
                                        * step)/(2. * self.batch_tri_exp_step_size)))) + \
                                        self.init_lr
                                        
        return float(updated_lr)
    

class Cosine_annealing(object):
    """
    Cyclic Cosine Annealing function as given by Huang et al. (2017)
    TbyM = ceil(T/M)
    lr(t) = lr(0) * 0.5 * (cos((pi*mod(t-1, TbyM)) / (TbyM)) + 1)
    where   T -> Total Iterations
            M -> Number of Cycles
    """
    def __init__(self, init_lr, batch_cos_steps_per_cycle):
        self.batch_cos_steps_per_cycle  = batch_cos_steps_per_cycle
        self.init_lr                    = init_lr

    def __call__(self, step):        
        updated_lr                      = (np.cos(np.pi * np.mod(step-1, \
                                         self.batch_cos_steps_per_cycle) \
                                        /(1. * self.batch_cos_steps_per_cycle)) \
                                        + 1.) * self.init_lr * 0.5
        
        return float(updated_lr)


class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
    schedule: a function that takes an epoch or a batch index as input
              (integer, indexed from 0) and current learning rate
              and returns a new learning rate as output (float).
    verbose: int. 0: quiet, 1: update messages.
    """
    def __init__(self, schedule, train_on_batch, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.train_on_batch     = train_on_batch
        self.schedule           = schedule
        self.verbose            = verbose
        self.batch_count        = 0
        

    def on_batch_begin(self, batch, logs=None):
        if(self.train_on_batch):
            lr                  = float(K.get_value(self.model.optimizer.lr))
            lr                  = self.schedule(self.batch_count)
            
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: LearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

    def on_batch_end(self, batch, logs=None):
        if(self.train_on_batch):
            logs                = logs or {}
            self.batch_count   +=1
            logs['lr']          = K.get_value(self.model.optimizer.lr)


    def on_epoch_begin(self, epoch, logs=None):
        if not (self.train_on_batch):
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a super(LearningRateScheduler, self).__init__())')
                    
            lr                  = float(K.get_value(self.model.optimizer.lr))
            
            try:                                                    # new API
                lr              = self.schedule(epoch, lr)
            except TypeError:                                       # old API for backward compatibility
                lr              = self.schedule(epoch)
            
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nEpoch %05d: LearningRateScheduler setting learning '
                      'rate to %s.' % (epoch + 1, lr))
                

    def on_epoch_end(self, epoch, logs=None):
        if not (self.train_on_batch):
            logs                = logs or {}
            logs['lr']          = K.get_value(self.model.optimizer.lr)


def get_lr_scheduler(config, schedule_choice):
    
    config_lr                   = config['LR']
    init_lr                     = config_lr.get('init_lr')
    maxLR                       = config_lr.get('max_lr')
    train_on_batch              = config_lr.get('train_on_batch')
    batch_expo_gamma            = config_lr.get('batch_expo_gamma')
    epoch_expo_gamma            = config_lr.get('epoch_expo_gamma')

    batch_step_gamma            = config_lr.get('batch_step_gamma')
    epoch_step_gamma            = config_lr.get('epoch_step_gamma')

    batch_step_step_size        = config_lr.get('batch_step_step_size')
    epoch_step_step_size        = config_lr.get('epoch_step_step_size')

    batch_poly_total_steps      = config_lr.get('batch_poly_total_steps')
    epoch_poly_total_steps      = config_lr.get('epoch_poly_total_steps')
    poly_power                  = config_lr.get('poly_power')

    batch_tri_step_size         = config_lr.get('batch_tri_step_size')
    batch_tri_exp_gamma         = config_lr.get('batch_tri_exp_gamma')
    batch_cos_steps_per_cycle   = config_lr.get('batch_cos_steps_per_cycle')

    
    if schedule_choice == 1:
        print('LR Schedule: Stable')        
        
        lr      = Stable(init_lr)
        
        return lr
    
    elif schedule_choice == 2:
        print('LR Schedule: Step Decay')
        
        if (train_on_batch):
            lr  = StepDecay(init_lr, batch_step_gamma, batch_step_step_size)            
        else:
            lr  = StepDecay(init_lr, epoch_step_gamma, epoch_step_step_size)
        
        return lr
    
    elif schedule_choice == 3:
        print('LR Schedule: Exponential Decay')

        if (train_on_batch):
            lr  = ExpoDecay(init_lr, batch_expo_gamma)
        else:
            lr  = ExpoDecay(init_lr, epoch_expo_gamma)
        
        return lr
    
    elif schedule_choice == 4:
        print('LR Schedule: Poly')

        if (train_on_batch):
            lr  = Poly(init_lr, maxLR, batch_poly_total_steps, poly_power)
        else:
            lr  = Poly(init_lr, maxLR, epoch_poly_total_steps, poly_power)
        
        return lr
    
    elif schedule_choice == 5:
        print('LR Schedule: Triangular')
        
        lr      = Triangular(init_lr, maxLR, batch_tri_step_size)
        
        return lr
    
    elif schedule_choice == 6:
        print('LR Schedule: Triangular Exp')
        
        lr      = TriangularExp(init_lr, maxLR, batch_tri_exp_gamma, batch_tri_step_size)
        
        return lr
    
    elif schedule_choice==7:
        print('LR Schedule: Cosine annealing')

        lr      = Cosine_annealing(init_lr, batch_cos_steps_per_cycle)
        
        return lr
    
    elif schedule_choice==8:
        print('LR Schedule: Triangular 2')

        lr      = Triangular2(init_lr, maxLR, batch_tri_step_size)
        
        return lr

    
    else:
        print('[WARNING]: Invalid LR_Scheduler choice')
        
        return None

###############################################################################
