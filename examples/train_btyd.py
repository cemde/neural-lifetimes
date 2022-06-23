import datetime
from pathlib import Path

import numpy as np

import pytorch_lightning as pl

from neural_lifetimes import run_model
from neural_lifetimes.data.datamodules import SequenceDataModule
from neural_lifetimes.data.datasets.btyd import BTYD, GenMode
from neural_lifetimes.models.modules import VariationalEventModel
from neural_lifetimes.utils.data import FeatureDictionaryEncoder, Tokenizer, TargetCreator
from examples import eventsprofiles_datamodel


LOG_DIR = str(Path(__file__).parent / 'logs')
data_dir = str(Path(__file__).parent.absolute())

START_TOKEN_DISCR = "<StartToken>"
COLS = eventsprofiles_datamodel.target_cols + eventsprofiles_datamodel.cont_feat + eventsprofiles_datamodel.discr_feat

if __name__ == "__main__":
    pl.seed_everything(9473)

    btyd_dataset = BTYD.from_modes(
        modes=[
            GenMode(
                a=5,
                b=100,
                r=5,
                alpha=10,
                discrete_dist = {
                    "PRODUCT_TYPE":{"CARD":0.8, "BALANCE":0.2},
                    "SUCCESSFUL_ACTION":{0:0.9, 1:0.1},
                    'SOURCE_CURRENCY':{"GBP":0.7, "EUR":0.2, "USD":0.1}
                },
                cont_dist = {
                    "INVOICE_VALUE_GBP_log": (-2,1)
                }),
            GenMode(
                a=20,
                b=100,
                r=10,
                alpha=600,
                discrete_dist = {
                    "PRODUCT_TYPE":{"CARD": 0.3, "BALANCE": 0.7},
                    "SUCCESSFUL_ACTION":{0: 0.8, 1: 0.2},
                    'SOURCE_CURRENCY':{"GBP": 0.1, "DKK": 0.9}
                },
                cont_dist = {
                    "INVOICE_VALUE_GBP_log": (5,2)
                }),
        ],
        num_customers=10000,
        mode_ratios=[1, 5],  # generate equal number of transactions from each mode
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
        batch_points=4096,
        min_points=1,
    )

    net = VariationalEventModel(
        feature_encoder_config=encoder.config_dict(),
        rnn_dim=256,
        emb_dim=256,
        drop_rate=0.5,
        bottleneck_dim=32,
        lr=0.001,
        target_cols=COLS,
        vae_sample_z=True,
        vae_sampling_scaler=1.0,
        vae_KL_weight=0.01,
    )

    run_model(
        datamodule,
        net,
        log_dir=LOG_DIR,
        num_epochs=200,
        val_check_interval=10,
        limit_val_batches=20,
        gradient_clipping=0.0000001,
        trainer_kwargs={'accelerator': 'gpu', "log_every_n_steps": 10}
    )
