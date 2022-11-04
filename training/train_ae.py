
import math
import os
import pickle
import sys
import pytorch_lightning as pl
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.autoencoder import *  #Lit_AE, AutoEncoderReLU, AutoEncoderTanh#, run_training
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import RichProgressBar


########################PARAMETERS#################################

#input data
EXP_NAME = "experiment4"

#model
LATENT_SPACE_DIM = 3
MODEL = AutoEncoderSig

#training
TEST_SIZE = 0.2
VAL_SIZE = 0.2
EPOCHS = 200
TRAIN_BATCH_SIZE = 256

#saving
MODEL_NAME = "AE_eucl_exp4_b"

####################################################################
def preprocessing(X, placeCells): #Todo add in class
    # X = X/placeCells.nu_max

    X_pp = X / placeCells.nu_max * (math.e - 1) + 1  # input between 0 and 1
    X_pp = np.log(X_pp)

    return X_pp

def run_training_process(run_params):
    #check path
    if os.path.exists('./saved_models/' + MODEL_NAME):
        print("ERROR: this model name already exist. Please provide a new model name")
        sys.exit()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "./data_generation/generated_data/" + EXP_NAME
    #with open(path + "/maze.pkl", 'rb') as file:
        #maze = pickle.load(file)
    with open(path + "/trajectory.pkl", 'rb') as file:
        traj = pickle.load(file)
    with open(path + "/placeCells.pkl", 'rb') as file:
        placeCells = pickle.load(file)

    X = placeCells.firingRates
    X = preprocessing(X, placeCells)
    X_nonoise = preprocessing(placeCells.noNoiseFirinRates, placeCells)

    #build useful idx arrays
    trajectory = np.array([traj.x_traj, traj.y_traj])
    long_traj_maze_config = np.empty(trajectory.shape[1])
    long_traj_maze_config[0:traj.traj_cut_idx[0]] = traj.corr_maze_config[0]
    for i in range(len(traj.corr_maze_config)):
        cur_conf = traj.corr_maze_config[i]
        #nb_goals = (len(maze.goals[cur_conf]))
        long_traj_maze_config[traj.traj_cut_idx[i]:traj.traj_cut_idx[i + 1]] = cur_conf
    long_edge_pos = np.empty(trajectory.shape[1])
    long_edge_pos[0:traj.traj_cut_idx[0]] = traj.corr_maze_config[0]
    for i in range(len(traj.edge_position)):
        # nb_goals = (len(maze.goals[cur_conf]))
        long_edge_pos[traj.traj_cut_idx[i]:traj.traj_cut_idx[i + 1]] = traj.edge_position[i]

    #Train, valid, test split
    X_train, X_test, X_n_train, X_n_test, traj_train, traj_test, maze_config_train, maze_config_test, edge_train, edge_test\
        = train_test_split(X, X_nonoise, trajectory.T, long_traj_maze_config, long_edge_pos, test_size=TEST_SIZE)

    X_train, X_val, X_n_train, X_n_val, traj_train, traj_val, maze_config_train, maze_config_val, edge_train, edge_val \
        = train_test_split(X_train, X_n_train, traj_train, maze_config_train, edge_train, test_size=VAL_SIZE)

    X_train = torch.tensor(X_train, dtype= torch.float32)
    X_val   = torch.tensor(X_val, dtype= torch.float32)
    X_test  = torch.tensor(X_test,  dtype= torch.float32)

    #build datasets
    train_loader  = torch.utils.data.DataLoader(X_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader    = torch.utils.data.DataLoader(X_val, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader   = torch.utils.data.DataLoader(X_test, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    #build model
    model = Lit_AE(MODEL(input_shape=X.shape[1], latent_shape=LATENT_SPACE_DIM))

    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return test_loader

    ##CALLBACKS
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min')

    callbacks = [checkpoint_callback, early_stop_callback, RichProgressBar()]

    #Training
    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(run_params, logger=logger,
                                            callbacks=callbacks)
    trainer.fit(model, datamodule=MyDataModule())
    print('Training process has finished.')

    # SAVE
    os.mkdir('./saved_models/' + MODEL_NAME)


    with open("./saved_models/" + MODEL_NAME + "/train_data", 'wb') as outp:
        pickle.dump(X_train, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/test_data", 'wb') as outp:
        pickle.dump(X_test, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/nonoise_train_data", 'wb') as outp:
        pickle.dump(X_n_train, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/nonoise_test_data", 'wb') as outp:
        pickle.dump(X_n_test, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/test_data_traj", 'wb') as outp:
        pickle.dump(traj_test, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/train_data_traj", 'wb') as outp:
        pickle.dump(traj_train, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/test_maze_config", 'wb') as outp:
        pickle.dump(maze_config_test, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/train_maze_config", 'wb') as outp:
        pickle.dump(maze_config_train, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/train_edge", 'wb') as outp:
        pickle.dump(edge_train, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/test_edge", 'wb') as outp:
        pickle.dump(edge_test, outp, pickle.HIGHEST_PROTOCOL)
    with open("./saved_models/" + MODEL_NAME + "/loss_logs", 'wb') as outp:
        pickle.dump(model.auto_encoder.lossLogs, outp, pickle.HIGHEST_PROTOCOL)
        # torch.save(model.state_dict(), "./saved_models/" + MODEL_NAME + "/model")
    with open("./saved_models/" + MODEL_NAME + "/model", 'wb') as outp:
        pickle.dump(model.auto_encoder, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    #TODO build params class for training
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args(['--log_every_n_steps', '500',
                                '--max_epochs', str(EPOCHS),
                                '--check_val_every_n_epoch', '1'])

    params = parser.parse_args(namespace=params)
    run_training_process(params)



