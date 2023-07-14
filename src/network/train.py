"""
This file includes the main libraries in the network training module.
"""

import json
import os
import signal
import sys
import time
from functools import partial
from os import path as osp

import numpy as np
import torch
# from dataloader.dataset_fb import FbSequenceDataset
from dataloader.tlio_data import TlioData
from network.losses import get_loss, get_loss_so3, loss_mse_so3
from network.model_factory import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.logging import logging
from utils.utils import to_device
from utils.math_utils import sixD2so3, so32sixD


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(network, data_loader, device, epoch, transforms=[]):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    network.eval()

    for bid, sample in enumerate(data_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        pred, pred_cov = network(feat)

        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:, -1, :]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:, 1:, :].permute(0, 2, 1)

        loss = get_loss(pred, pred_cov, targ, epoch)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        losses_all.append(torch_to_numpy(loss))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
    }
    return attr_dict

def get_inference_so3(network, data_loader, device, epoch, transforms=[]):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    network.eval()

    for bid, sample in enumerate(data_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        pred, pred_cov = network(feat)
        pred = sixD2so3(pred.unsqueeze(2)).squeeze()  # pred: (1024, 3, 3)

        if len(pred.shape) == 3:
            targ = sample["targ_dR_World"][:, -1, :, :]  # trag: (1024, 3, 3)
            #targ = so32sixD(targ)  # trag: (1024, 6)
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dR_World"][:, 1:, :, :].permute(0, 2, 3, 1) # trag: (1024, 3, 3, 199)
            #targ = so32sixD(targ)  # trag: (1024, 6, 199)

        loss = get_loss_so3(pred, pred_cov, targ, epoch)
        # loss = loss_mse_so3(pred, targ) # dR 학습하는 경우 사용

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        losses_all.append(torch_to_numpy(loss))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
    }
    return attr_dict


def do_train(network, train_loader, device, epoch, optimizer, transforms=[]):
    """
    displacement를 계산하는 TLIO 의 기존 network
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()

    # for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        pred, pred_cov = network(feat)

        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:, -1, :] #targ: (1024,3)
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:, 1:, :].permute(0, 2, 1)

        loss = get_loss(pred, pred_cov, targ, epoch)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))

        # print("Loss full: ", loss)

        loss = loss.mean()
        loss.backward()

        # print("Loss mean: ", loss.item())

        # print("Gradients:")
        # for name, param in network.named_parameters():
        #    if param.requires_grad:
        #        print(name, ": ", param.grad)

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict


def do_train_R(network, train_loader, device, epoch, optimizer, transforms=[]):
    """
    rotation matrix를 계산하는 network, 이 네트워크를 사용하기 위해서는 net_train에서 do_train 부분을 do_train_R로 바꿔주면 됨.
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()

    # for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        pred, pred_cov = network(feat) # pred: (1024, 6) or (1024,6,199)  # nan 발생 지정
        pred = sixD2so3(pred.unsqueeze(2)).squeeze()  # pred: (1024, 3, 3)

        if len(pred.shape) == 3:
            targ = sample["R_W_i"][:, -1, :, :]  # trag: (1024, 3, 3) 1:N:20
        else:
        # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["R_W_i"][:, 1:, :, :].permute(0, 2, 3, 1) # trag: (1024, 3, 3, 199)

        loss = get_loss_so3(pred, pred_cov, targ, epoch)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
        # print("bid: ", bid)
        if torch.any(torch.isnan(loss)):
            print("loss is finite: ", torch.any(torch.isfinite(loss)))

        loss = loss.mean()
        loss.backward()
        # NaN debugging
        if torch.any(torch.isnan(loss)):
            print("pred is finite: ", torch.any(torch.isfinite(pred)))
            print("pred: ", pred.mean())
            print("loss: ", loss)
        if torch.any(torch.isnan(pred)):
            print("pred is finite: ", torch.any(torch.isfinite(pred)))
            print("pred: ", pred.mean())
            print("loss: ", loss)
            input('stop: ')
        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict

def do_train_dR(network, train_loader, device, epoch, optimizer, transforms=[]):
    """
    rotation matrix의 차이를 계산하는 network, 이 네트워크를 사용하기 위해서는 net_train에서 do_train 부분을 do_train_dR로 바꿔주고
    get_inference_so3 get_loss 부분을 loss_mse_so3로 바꿔주기.
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()

    # for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        pred, pred_cov = network(feat) # pred: (1024, 6) or (1024,6,199)
        pred = sixD2so3(pred.unsqueeze(2)).squeeze()  # pred: (1024, 3, 3)

        if len(pred.shape) == 3:
            targ = sample["targ_dR_World"][:, -1, :, :]  # trag: (1024, 3, 3)
            #targ = so32sixD(targ)  # trag: (1024, 6)
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dR_World"][:, 1:, :, :].permute(0, 2, 3, 1) # trag: (1024, 3, 3, 199)
            #targ = so32sixD(targ)  # trag: (1024, 6, 199)

        loss = loss_mse_so3(pred, targ)
        # loss = get_loss_so3(pred, pred_cov, targ, epoch)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))

        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict

def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """

    mse_loss = np.mean((attr_dict["targets"] - attr_dict["preds"]) ** 2, axis=0)  #shape (3,3)
    mse_so3_loss = np.mean(loss_mse_so3(attr_dict["preds"], attr_dict["targets"]).cpu().detach().numpy())   #pred와 targ각도차
    ml_loss = np.average(attr_dict["losses"])  #shape (1)
    sigmas = np.exp(attr_dict["preds_cov"])  #shape (3,3)
    # print("mse_loss size: ", mse_loss.shape)
    # print("ml_loss size: ", ml_loss.shape)
    # print("sigmas size: ", sigmas.shape)
    # If it's sequential, take the last one
    # if len(mse_loss.shape) == 2:
    #     assert mse_loss.shape[0] == 3
    #     mse_loss = mse_loss[:, -1]
    #     assert sigmas.shape[1] == 3
    #     sigmas = sigmas[:, :, -1]
    summary_writer.add_scalar(f"{mode}_loss/mse_loss_avg", np.mean(mse_loss), epoch)
    summary_writer.add_scalar(f"{mode}_loss/mse_so3_loss_avg", np.mean(mse_so3_loss), epoch)
    summary_writer.add_scalar(f"{mode}_loss/nll_loss_full", ml_loss, epoch)
    summary_writer.add_scalar(f"{mode}_dist/sigma_x", np.mean(sigmas[:, 0]), epoch)
    summary_writer.add_scalar(f"{mode}_dist/sigma_y", np.mean(sigmas[:, 1]), epoch)
    summary_writer.add_scalar(f"{mode}_dist/sigma_z", np.mean(sigmas[:, 2]), epoch)
    summary_writer.add_scalar(f"{mode}_dist/sigma_avg", np.mean(sigmas), epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(
        f"{mode}: average ml loss: {ml_loss}, average mse loss: {mse_loss}/{np.mean(mse_loss)}"
    )


def save_model(args, epoch, network, optimizer, best, interrupt=False):
    if interrupt:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
    if best:
        model_path = osp.join(args.out_dir, "checkpoint_best.pt")
    else:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dict(
        [
            ("past_data_size", int(args.past_time * args.imu_freq)),
            ("window_size", int(args.window_time * args.imu_freq)),
            ("future_data_size", int(args.future_time * args.imu_freq)),
            ("step_size", int(args.imu_freq / args.sample_freq)),
        ]
    )
    net_config = {
        "in_dim": (
                          data_window_config["past_data_size"]
                          + data_window_config["window_size"]
                          + data_window_config["future_data_size"]
                  )
                  // 32
                  + 1
    }

    return data_window_config, net_config


def net_train(args):
    """
    Main function for network training
    """
    # os.mkdir(args.model_path, exist_ok=True)
    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        # if args.train_list is None:
        #    raise ValueError("train_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not osp.isdir(osp.join(args.out_dir, "checkpoints")):
                os.makedirs(osp.join(args.out_dir, "checkpoints"))
            if not osp.isdir(osp.join(args.out_dir, "logs")):
                os.makedirs(osp.join(args.out_dir, "logs"))
            with open(
                    os.path.join(args.out_dir, "parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        # if args.val_list is None:
        #    logging.warning("val_list is not specified.")
        if args.continue_from is not None:
            if osp.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Perturb on bias: %s" % args.do_bias_shift)
    logging.info("Perturb on gravity: %s" % args.perturb_gravity)
    logging.info("Sample frequency: %s" % args.sample_freq)

    train_loader, val_loader = None, None
    start_t = time.time()

    data = TlioData(
        args.root_dir,
        batch_size=args.batch_size,
        dataset_style=args.dataset_style,
        num_workers=args.workers,
        persistent_workers=args.persistent_workers,
    )
    data.prepare_data()

    train_list = data.get_datalist("train")

    """
    try:
        train_dataset = FbSequenceDataset(
            args.root_dir, train_list, args, data_window_config, mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
    except OSError as e:
        logging.error(e)
        return
    """
    train_loader = data.train_dataloader()
    train_transforms = data.get_train_transforms()

    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(data.train_dataset)}")

    # if args.val_list is not None:
    if data.val_dataset is not None:
        val_list = data.get_datalist("val")
        """
        try:
            val_dataset = FbSequenceDataset(
                args.root_dir, val_list, args, data_window_config, mode="val"
            )
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
        except OSError as e:
            logging.error(e)
            return
        """
        val_loader = data.val_dataloader()
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(data.val_dataset)}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    network = get_model(args.arch, net_config, args.input_dim, args.output_dim)
    network.to(device)
    total_params = network.get_num_params()
    logging.info(f'Network "{args.arch}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from is not None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    summary_writer = SummaryWriter(osp.join(args.out_dir, "logs"))
    summary_writer.add_text("info", f"total_param: {total_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")

    # attr_dict = get_inference(network, train_loader, device, start_epoch, train_transforms)
    # write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train")
    # if val_loader is not None:
    #    attr_dict = get_inference(network, val_loader, device, start_epoch)
    #    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "val")

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, best=False, interrupt=True)
        sys.exit()

    best_val_loss = np.inf
    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        train_attr_dict = do_train_R(network, train_loader, device, epoch, optimizer, train_transforms)
        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if val_loader is not None:
            val_attr_dict = get_inference_so3(network, val_loader, device, epoch)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer, best=True)
        else:
            save_model(args, epoch, network, optimizer, best=False)

    logging.info("Training complete.")

    return
