import datetime
from pathlib import Path

import argparse
import itertools

import numpy as np

import pytorch_lightning as pl

from neural_lifetimes import run_model
from neural_lifetimes.data.datamodules import SequenceDataModule
from neural_lifetimes.data.datasets.btyd import BTYD, GenMode
from neural_lifetimes.models.modules import InformationBottleneckEventModel
from neural_lifetimes.utils.data import FeatureDictionaryEncoder, Tokenizer, TargetCreator
from examples import eventsprofiles_datamodel


LOG_DIR = str(Path(__file__).parent / "logs")
data_dir = str(Path(__file__).parent.absolute())

START_TOKEN_DISCR = "<StartToken>"
COLS = eventsprofiles_datamodel.target_cols + eventsprofiles_datamodel.cont_feat + eventsprofiles_datamodel.discr_feat


def prepare_everything():
    btyd_dataset = BTYD.from_modes(
        modes=[
            GenMode(a=2, b=10, r=5, alpha=10),
            GenMode(a=2, b=10, r=10, alpha=600),
        ],
        num_customers=10000,
        mode_ratios=[2.5, 1],  # generate equal number of transactions from each mode
        seq_gen_dynamic=False,
        start_date=datetime.datetime(2019, 1, 1, 0, 0, 0),
        start_limit_date=datetime.datetime(2019, 6, 15, 0, 0, 0),
        end_date=datetime.datetime(2021, 1, 1, 0, 0, 0),
        data_dir=data_dir,
        continuous_features=eventsprofiles_datamodel.cont_feat,
        discrete_features=eventsprofiles_datamodel.discr_feat,
        track_statistics=True,
    )

    btyd_dataset[:]
    print(f"Expected Num Transactions per mode: {btyd_dataset.expected_num_transactions_from_priors()}")
    print(f"Expected p churn per mode: {btyd_dataset.expected_p_churn_from_priors()}")
    print(f"Expected time interval per mode: {btyd_dataset.expected_time_interval_from_priors()}")
    print(f"Truncated sequences: {btyd_dataset.truncated_sequences}")

    btyd_dataset.plot_tracked_statistics().show()

    discrete_values = btyd_dataset.get_discrete_feature_values(
        start_token=START_TOKEN_DISCR,
    )

    encoder = FeatureDictionaryEncoder(eventsprofiles_datamodel.cont_feat, discrete_values)

    tokenizer = Tokenizer(
        continuous_features=eventsprofiles_datamodel.cont_feat,
        discrete_features=discrete_values,
        start_token_continuous=np.nan,
        start_token_discrete=START_TOKEN_DISCR,
        start_token_other=np.nan,
        max_item_len=100,
        start_token_timestamp=datetime.datetime(1970, 1, 1, 1, 0),
    )

    target_transform = TargetCreator(cols=COLS)

    datamodule = SequenceDataModule(
        dataset=btyd_dataset,
        tokenizer=tokenizer,
        transform=encoder,
        target_transform=target_transform,
        test_size=0.2,
        batch_points=1024,
        min_points=1,
    )
    return encoder, datamodule

def define_model_and_run(encoder, datamodule, step_scale, target_weight, n_eigen, n_eigen_threshold, encoder_noise):

    loss_cfg = {
        "n_cold_steps": step_scale*1,
        "n_warmup_steps": step_scale*1,
        "target_weight": target_weight,
        "n_eigen": n_eigen,
        "n_eigen_threshold": n_eigen_threshold,
    }

    net = InformationBottleneckEventModel(
        feature_encoder_config=encoder.config_dict(),
        rnn_dim=256,
        emb_dim=256,
        drop_rate=0.5,
        bottleneck_dim=32,
        lr=0.0001,
        target_cols=COLS,
        encoder_noise=encoder_noise,
        loss_cfg=loss_cfg,
    )

    trainer = run_model(
        datamodule,
        net,
        log_dir=LOG_DIR,
        num_epochs=20,
        val_check_interval=50,
        limit_val_batches=20,
        gradient_clipping=0.0000001,
        trainer_kwargs={'accelerator': 'gpu'}
    )
    
    return 



if __name__ == "__main__":
    pl.seed_everything(9473)

    # parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('--ib_target_weight','-itw', help='Target weight for the information bottleneck', type=float, default=1e-6)
    # parser.add_argument('--step_scale','-itw', help='scale for weight warmup', type=int, default=100)
    # parser.add_argument('--n_eigen','-itw', help='number of eigenvalues', type=int, default=None)
    # parser.add_argument('--n_eigen_threshold','-itw', help='threshold for smallest eigenvalue to use', type=float, default=None)
    # parser.add_argument('--encoder_noise','-itw', help='Noise to be added to the encoder', type=float, default=1e-6)

    # args = parser.parse_args()
    
    params = dict(
        step_scale = [10, 100, 1000],
        target_weight = [1e-12, 1e-8, 1e-6, 1e-3],
        n_eigen = [None, 5, 20, 100],
        n_eigen_threshold = [None, 1e-5, 1e-2, 10, 1e5],
        encoder_noise = [1e-6, 1e-3],
    )
    
    # prod = itertools.product(step_scale, target_weight, n_eigen, n_eigen_threshold, encoder_noise)
    
    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))
    params = list(dict_product(params))

    to_delete = []
    for i, el in enumerate(params):
        if (el['n_eigen_threshold'] is None) == (el['n_eigen'] is None):
            to_delete.append(i)
    params = [p for i, p in enumerate(params) if i not in to_delete]
    

    encoder, datamodule = prepare_everything()

    for p in params:
        print(p)
        metric = define_model_and_run(encoder, datamodule, p["step_scale"], p["target_weight"], p["n_eigen"], p["n_eigen_threshold"], p["encoder_noise"])
        print(metric)