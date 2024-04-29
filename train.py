import os, time, datetime
import numpy                                                as np
from shutil                             import copyfile
from parse_config                       import parse_config
#from model                              import get_model
from nnunet                             import nnunet_model
from BatchGen                           import DataGenerator
from lr_schedulers                      import get_lr_scheduler, LearningRateScheduler
from tensorflow.python.keras.callbacks  import TensorBoard
from losses_and_metrics                 import get_loss_function, dice_LV_tf, dice_RV_tf, dice_Myo_tf
from optimizers                         import get_optimizer
try:
    from tensorflow.python.keras        import backend      as K
    from tensorflow.python.keras        import callbacks
except:
    import keras.backend                                    as K
    from keras                          import callbacks

# Uncomment below if the random seed needs to be fixed
# K.set_random_seed(237)

class LRTensorBoard(callbacks.TensorBoard):
	"""
	modified tensorboard for logging learning rate, batch-wise & epoch-wise
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def on_train_batch_end(self, step, logs=None):
		logs.update({'lr': K.eval(self.model.optimizer.lr)})
		super().on_train_batch_end(step, logs)

class Training(object):
    def __init__(self, config_file):        
        self.config                   = parse_config(config_file)
        self.config_file              = config_file
        
        # Reading training parameters
        self.config_train             = self.config['Train']
        self.model_choice             = self.config_train.get('model_choice')
        self.epochs                   = self.config_train.get('epochs')
        self.epoch_gap_model_save     = self.config_train.get('epoch_gap_model_save')
        self.model_log_dir            = self.config_train.get('model_save_dir')
        self.num_classes              = self.config_train.get('num_classes')
        self.num_channels             = self.config_train.get('num_channels')
        
        self.train_batch_size         = self.config_train.get('train_batch_size')
        self.crop_size                = self.config_train.get('crop_size')

        # Reading learning rate parameters
        self.config_lr                = self.config['LR']
        self.choice_lr_scheduler      = self.config_lr.get('choice_lr_scheduler')
        self.train_on_batch           = self.config_lr.get('train_on_batch')
        self.AutoReduce_flag          = self.config_lr.get('autoreduce_flag')
        self.AutoReduce_factor        = self.config_lr.get('autoreduce_factor')
        self.patience                 = self.config_lr.get('patience')
        
        # Fine Tuning parameters
        self.config_ft                = self.config['Fine Tuning']
        self.FineTune_flag            = self.config_ft.get('ft_flag')
        self.model_weights            = self.config_ft.get('model_weights')

        # Optimizier Choice
        self.config_opt               = self.config['Optimizer']
        self.optimizer_choice         = self.config_opt.get('optimizer_choice')
        
        # Loss and metrics Choice
        self.config_losses            = self.config['Losses']
        self.loss_choice              = self.config_losses.get('loss_choice')

    def __call__(self, volume_train, label_train, volume_val, label_val):        

        # Fetch time
        run_date_time                 = time.strftime('%Y-%m-%d_%H.%M.%S')

        # Creating directory for tensorboard logs
        if not os.path.exists(self.model_log_dir):
            os.makedirs(self.model_log_dir)
            
        tensorboard_log_path          = os.path.join(self.model_log_dir, run_date_time)
        csv_file_name                 = os.path.join(self.model_log_dir, run_date_time, 'training_log_{}.csv'.format(run_date_time))

        # Creating paths for model file
        model_checkpoint_format       = 'epoch_{epoch:02d}_val_loss_{val_loss:.2f}_' + run_date_time + '.hdf5'
        model_checkpoint_filepath     = os.path.join(self.model_log_dir, run_date_time, model_checkpoint_format)

        # Calling training and validation batch generators
        train_generator               = DataGenerator(volume_train,
                                                      label_train,
                                                      self.crop_size,
                                                      self.num_classes,
                                                      self.num_channels,
                                                      self.config,
                                                      'train')

        val_generator                 = DataGenerator(volume_val,
                                                      label_val,
                                                      self.crop_size,
                                                      self.num_classes,
                                                      self.num_channels,
                                                      self.config,
                                                      'val')

        # Picking up the desired model i.e. Unet or FCN
        if self.num_classes == 2:
            accuracy='binary_accuracy'
        elif self.num_classes > 2:
            accuracy='categorical_accuracy'
            
        if (self.FineTune_flag):
            model                    = nnunet_model(self.config, weights=None)
            model.load_weights(self.model_weights)
            print("Loaded model from disk")

            for i, layer in enumerate(model.layers):
                print(i, layer.name)
                
            '''
            print ("these layers wont be trained")
            for layer in model.layers[:8]:
                layer.trainable=False
                print (layer)
            '''
        else:
            model                   = nnunet_model(self.config, weights=None)
        model.summary()
        
        losses = {"pred1": get_loss_function(self.loss_choice) ,
	             "pred2": get_loss_function(self.loss_choice)}
        
        lossWeights = {"pred1":0.67 , "pred2": 0.33}
        #loss                        = get_loss_function(self.loss_choice) 
        optimizer_instance          = get_optimizer(self.config, self.optimizer_choice)
        metrics_list                = [accuracy, dice_LV_tf, dice_RV_tf, dice_Myo_tf]
        
        model.compile(optimizer=optimizer_instance, loss=losses, loss_weights=lossWeights, metrics=metrics_list)
       
        
        print('<< Metrics>>:', model.metrics_names)
        
        # Defining various callBacks
        checkpoints                 = callbacks.ModelCheckpoint(filepath=model_checkpoint_filepath,
                                                                monitor='val_loss',
                                                                verbose=1,
                                                                save_weights_only=True,
                                                                save_best_only=False,
                                                                period=self.epoch_gap_model_save)
        
        tensorboard                 = LRTensorBoard(log_dir=tensorboard_log_path, update_freq=1)#self.train_batch_size)
        csv_logger                  = callbacks.CSVLogger(filename=csv_file_name)
        
        if self.AutoReduce_flag:
            lr_scheduler            = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                                  factor=self.AutoReduce_factor,
                                                                  patience=self.patience, 
                                                                  min_lr=1e-7)

            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Auto Reduce is Activated>>>>>>>>>>>>>>>>>>>>>>>')



        else:
            schedule                = get_lr_scheduler(self.config, self.choice_lr_scheduler)
            lr_scheduler            = LearningRateScheduler(schedule, self.train_on_batch, verbose=0)
            
        callbacks_list              = [checkpoints, tensorboard, csv_logger, lr_scheduler]

        # Start Training
        t0                          = datetime.datetime.now()

        print('\n> Starting Clock: {}\n'.format(t0))

        train_hist                  = model.fit_generator(generator=train_generator,
                                                          validation_data=val_generator,
                                                          epochs=self.epochs,
                                                          callbacks=callbacks_list,
                                                          verbose=1,
                                                          initial_epoch=0,
                                                          shuffle=False,
                                                          use_multiprocessing=False
                                                          )
        # Backup config file for future reference
        copyfile(self.config_file, os.path.join(self.model_log_dir, run_date_time, 'config.txt'))

        # Print status
        print('\nTraining Finished Successfully!\n')
        tn                            = datetime.datetime.now()
        print('\n> Ending Clock: {}\n'.format(tn))
        print('--- Total Training Time: {} ---\n'.format(tn-t0))

###############################################################################
