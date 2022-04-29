import datetime
from pathlib import Path

import pytorch_lightning as pl

from neural_lifetimes import run_model
from neural_lifetimes.data.datamodules import SequenceDataModule
from neural_lifetimes.data.datasets.btyd import BTYD, GenMode
from neural_lifetimes.models import TargetCreator
from neural_lifetimes.models.modules import ClassicModel
from neural_lifetimes.data.encoder import CombinedFeatureEncoder
from neural_lifetimes.data.datamodel import DataModel

from examples import eventsprofiles_datamodel

LOG_DIR = str(Path(__file__).parent / "logs")
data_dir = str(Path(__file__).parent.absolute())

START_TOKEN_DISCR = "StartToken"
COLS = eventsprofiles_datamodel.target_cols + eventsprofiles_datamodel.cont_feat + eventsprofiles_datamodel.discr_feat

if __name__ == "__main__":
    pl.seed_everything(9473)

    btyd_dataset = BTYD.from_modes(
        modes=[
            GenMode(a=1.5, b=20, r=1, alpha=14),
            GenMode(a=2, b=50, r=2, alpha=6),
        ],
        num_customers=1000,
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

    # btyd_dataset[:]
    # print(f"Expected Num Transactions per mode: {btyd_dataset.expected_num_transactions_from_priors()}")
    # print(f"Expected p churn per mode: {btyd_dataset.expected_p_churn_from_priors()}")
    # print(f"Expected time interval per mode: {btyd_dataset.expected_time_interval_from_priors()}")
    # print(f"Truncated sequences: {btyd_dataset.truncated_sequences}")

    # btyd_dataset.plot_tracked_statistics().show()

    discrete_values = btyd_dataset.get_discrete_feature_values(
        start_token=START_TOKEN_DISCR,
    )  # TODO START_TOKEN SHOULD NOT BE PROCESSED here

    # TODO clean this up in this file. Move DataModel to seperate file
    datamodel = DataModel(
        continuous_features=eventsprofiles_datamodel.cont_feat,
        discrete_features=btyd_dataset.get_discrete_feature_values(),
        start_tokens={"discrete": START_TOKEN_DISCR, "continuous": 0},
        targets=eventsprofiles_datamodel.target_cols,
    )

    transform = CombinedFeatureEncoder(datamodel.continuous_features, datamodel.discrete_features)

    target_transform = TargetCreator(
        cols=datamodel.columns,
        encoder=transform,
        max_item_len=100,
        start_token_discr=datamodel.start_tokens["discrete"],
        start_token_cont=datamodel.start_tokens["continuous"],
    )

    # state_full datamodule
    datamodule = SequenceDataModule(
        dataset=btyd_dataset,
        transform=transform,
        target_transform=target_transform,
        test_size=0.2,
        batch_points=1024,
        min_points=1,
    )

    net = ClassicModel(
        data_config=datamodel.to_dict(),
        rnn_dim=256,
        drop_rate=0.5,
        bottleneck_dim=32,
        lr=0.001,
        vae_sample_z=True,
        vae_sampling_scaler=1.0,
        vae_KL_weight=0.01,
    )

    run_model(
        datamodule,
        net,
        log_dir=LOG_DIR,
        num_epochs=50,
        val_check_interval=20,
        limit_val_batches=20,
        gradient_clipping=0.0000001,
    )
