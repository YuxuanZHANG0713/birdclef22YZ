import warnings
import torch

#import src.callbacks as clb
import src.configuration as C
import src.models as models
import src.utils as utils

from catalyst.dl import SupervisedRunner

from pathlib import Path

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # args = utils.get_parser().parse_args()

    CONFIG_PATH = './configs/000_ResNet34.yml'
    config = utils.load_config(CONFIG_PATH)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.get_logger(output_dir / "output.log")

    utils.set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])

    df, datadir = C.get_metadata(config)
    splitter = C.get_split(config)

    for i, (trn_idx, val_idx) in enumerate(
            splitter.split(df, y=df["primary_label"])):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)

        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)

        loaders = {
            phase: C.get_loader(df_, datadir, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }
        model = models.get_model(config).to(device)
        criterion = C.get_criterion(config).to(device)
        optimizer = C.get_optimizer(model, config)
        scheduler = C.get_scheduler(optimizer, config)
        # callbacks = clb.get_callbacks(config)

        runner = SupervisedRunner(
            # engine=device,
            input_key=global_params["input_key"],
            target_key=global_params["input_target_key"])
        runner.train(
            model=model,
            criterion=criterion,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=global_params["num_epochs"],
            verbose=True,
            logdir=output_dir / f"fold{i}",
            # callbacks=callbacks,
            # main_metric=global_params["main_metric"],
            # minimize_metric=global_params["minimize_metric"]
            )