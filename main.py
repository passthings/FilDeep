import os
import logging

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.data_loader import FilDeepDataLoader
from core.losses import loss_p, loss_r
from Transformer_based_FilDeep.FilDeep import FilDeep
from utils.tensorboard_logging import Tensorboard_Logging
from utils.diff_recover import diff_recover

def train(
        low_train_dataloader,
        low_eval_dataloader,
        high_train_dataloader,
        high_eval_dataloader,
        device,
        exp_name,
        tl_writer,
):
    low_train_datas = DataLoader(
        low_train_dataloader,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    high_train_datas = DataLoader(
        high_train_dataloader,
        batch_size=high_train_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    low_eval_datas = DataLoader(
        low_eval_dataloader,
        batch_size=eval_batch_size,
        num_workers=num_workers
    )

    high_eval_datas = DataLoader(
        high_eval_dataloader,
        batch_size=high_eval_batch_size,
        num_workers=num_workers
    )

    low_coordinate_weights = [
            low_train_dataloader.scale_factor_for_unload[i] -
            low_train_dataloader.scale_factor_for_unload[i + 3]
            for i in range(3)
    ]

    high_coordinate_weights = [
            high_train_dataloader.scale_factor_for_unload[i] -
            high_train_dataloader.scale_factor_for_unload[i + 3]
            for i in range(3)
    ]

    checkpoints_save_path = os.path.join(checkpoints_path, exp_name)
    os.makedirs(checkpoints_save_path, exist_ok=True)

    torch.cuda.manual_seed(seed)

    net = FilDeep(
        is_frozen_CL=is_frozen_CL,
        is_frozen_CS=is_frozen_CS,
        is_frozen_MP=is_frozen_MP,
        is_frozen_WRFE=is_frozen_WRFE,
        is_frozen_ECFE=is_frozen_ECFE,
        is_frozen_low_Attention=is_frozen_low_Attention
    ).to(device)

    params_r = {
        n: p for n, p in net.low.named_parameters() if p.requires_grad and ("cse" in n or "csd" in n)
    }

    params_p_low = {
        n: p for n, p in net.low.named_parameters() if p.requires_grad and n not in params_r.keys()
    }

    params_p_high = {
        n: p for n, p in net.named_parameters() if
        p.requires_grad and n not in params_r.keys() and n not in params_p_low.keys()
    }

    optimizer_p_low = optim.Adam(
        params=params_p_low.values(),
        lr=lr_p,
        weight_decay=weight_decay_p
    )

    optimizer_r = optim.Adam(
        params=params_r.values(),
        lr=lr_r,
        weight_decay=weight_decay_r
    )

    optimizer_p_high = optim.Adam(
        params=params_p_high.values(),
        lr=high_lr_p,
        weight_decay=high_weight_decay_p
    )

    scheduler_p_low = CosineAnnealingLR(
        optimizer_p_low,
        T_max=epochs,
        eta_min=lr_p / lr_p_decay_min
    )

    scheduler_r = CosineAnnealingLR(
        optimizer_r,
        T_max=epochs,
        eta_min=lr_r / lr_r_decay_min
    )

    scheduler_p_high = CosineAnnealingLR(
        optimizer_p_high,
        T_max=epochs,
        eta_min=high_lr_p / high_lr_p_decay_min
    )

    len_low_train_datas = len(low_train_datas)
    len_high_train_datas = len(high_train_datas)
    len_low_eval_datas = len(low_eval_datas)
    len_high_eval_datas = len(high_eval_datas)

    min_low_train_loss_p = 1 << 30
    min_train_loss_r = 1 << 30
    min_high_train_loss_p = 1 << 30

    min_low_eval_loss_p = 1 << 30
    min_eval_loss_r = 1 << 30
    min_high_eval_loss_p = 1 << 30

    for epoch in range(epochs):
        net = net.train()

        mean_low_train_loss_p = 0.
        mean_train_loss_r = 0.
        for i, low_train_data in enumerate(low_train_datas):
            strip, mould, section, params,   unloaded_strip = \
                list(
                    map(
                        lambda x: x.to(device), low_train_data
                    )
                )
            recover_section, low_pred_unloaded_strip = net(
                strip, mould, section, params, fidelity=False)
            low_train_loss_p = loss_p(
                unloaded_strip, low_pred_unloaded_strip, low_coordinate_weights)
            train_loss_r = loss_r(section, recover_section)

            mean_low_train_loss_p += low_train_loss_p.data
            mean_train_loss_r += train_loss_r.data

            tl_writer.write_2d_figure(
                "low_train/train_loss_p_unloading", low_train_loss_p.data, epoch * len_low_train_datas + i
            )
            tl_writer.write_2d_figure(
                "low_train/train_loss_r", train_loss_r.data, epoch * len_low_train_datas + i
            )

            min_low_train_loss_p = min(
                min_low_train_loss_p,
                low_train_loss_p.data
            )
            min_train_loss_r = min(
                min_train_loss_r,
                train_loss_r.data
            )

            tl_writer.write_2d_figure(
                "low_train/min_train_loss_p_unloading",
                min_low_train_loss_p,
                epoch * len_low_train_datas + i
            )
            tl_writer.write_2d_figure(
                "low_train/min_train_loss_r",
                min_train_loss_r,
                epoch * len_low_train_datas + i
            )

            optimizer_p_low.zero_grad()
            optimizer_r.zero_grad()
            low_train_loss_p.backward(retain_graph=True)
            train_loss_r.backward()
            optimizer_p_low.step()
            optimizer_r.step()

        scheduler_p_low.step()
        scheduler_r.step()

        mean_low_train_loss_p /= len_low_train_datas
        mean_train_loss_r /= len_low_train_datas

        tl_writer.write_2d_figure(
            "low_train/mean_train_loss_p_unloading", mean_low_train_loss_p, epoch
        )
        tl_writer.write_2d_figure(
            "low_train/mean_train_loss_r", mean_train_loss_r, epoch
        )

        net = net.eval()
        with torch.no_grad():
            mean_low_eval_loss_p = 0.
            mean_eval_loss_r = 0.
            for i, low_eval_data in enumerate(low_eval_datas):
                strip, mould, section, params,   unloaded_strip = \
                    list(
                        map(
                            lambda x: x.to(device), low_eval_data
                        )
                    )
                recover_section, low_pred_unloaded_strip = net(
                    strip, mould, section, params, fidelity=False)
                low_eval_loss_p = loss_p(
                    unloaded_strip, low_pred_unloaded_strip, low_coordinate_weights)
                eval_loss_r = loss_r(section, recover_section)

                mean_low_eval_loss_p += low_eval_loss_p.data
                mean_eval_loss_r += eval_loss_r.data

                tl_writer.write_2d_figure(
                    "low_eval/eval_loss_p_unloading", low_eval_loss_p.data, epoch * len_low_eval_datas + i
                )
                tl_writer.write_2d_figure(
                    "low_eval/eval_loss_r", eval_loss_r.data, epoch * len_low_eval_datas + i
                )
            mean_low_eval_loss_p /= len_low_eval_datas
            mean_eval_loss_r /= len_low_eval_datas

            tl_writer.write_2d_figure(
                "low_eval/mean_eval_loss_p_unloading", mean_low_eval_loss_p,
                epoch
            )
            tl_writer.write_2d_figure(
                "low_eval/mean_eval_loss_r", mean_eval_loss_r,
                epoch
            )

            min_low_eval_loss_p = min(
                min_low_eval_loss_p, mean_low_eval_loss_p
            )
            min_eval_loss_r = min(
                min_eval_loss_r, mean_eval_loss_r
            )

        net = net.train()
        mean_high_train_loss_p = 0.
        for i, high_train_data in enumerate(high_train_datas):
            strip, mould, section, params,   unloaded_strip = \
                list(
                    map(
                        lambda x: x.to(device), high_train_data
                    )
                )
            high_pred_unloaded_strip = net(
                strip, mould, section, params, fidelity=True)
            high_train_loss_p = loss_p(
                unloaded_strip, high_pred_unloaded_strip, high_coordinate_weights)

            mean_high_train_loss_p += high_train_loss_p.data

            tl_writer.write_2d_figure(
                "high_train/train_loss_p_unloading",
                high_train_loss_p.data,
                epoch * len_high_train_datas + i
            )

            min_high_train_loss_p = min(
                min_high_train_loss_p,
                high_train_loss_p.data
            )

            tl_writer.write_2d_figure(
                "high_train/min_train_loss_p_unloading", min_high_train_loss_p, epoch * len_high_train_datas + i
            )

            optimizer_p_high.zero_grad()

            high_train_loss_p.backward(retain_graph=True)
            optimizer_p_high.step()

        scheduler_p_high.step()
        mean_high_train_loss_p /= len_high_train_datas

        tl_writer.write_2d_figure(
            "high_train/mean_train_loss_p_unloading", mean_high_train_loss_p, epoch
        )

        net = net.eval()
        with torch.no_grad():
            mean_high_eval_loss_p = 0.
            for i, high_eval_data in enumerate(high_eval_datas):
                strip, mould, section, params,   unloaded_strip = \
                    list(
                        map(
                            lambda x: x.to(device), high_eval_data
                        )
                    )
                high_pred_unloaded_strip = net(
                    strip, mould, section, params, fidelity=True)
                high_eval_loss_p = loss_p(
                    unloaded_strip, high_pred_unloaded_strip, high_coordinate_weights)

                mean_high_eval_loss_p += high_eval_loss_p.data

                tl_writer.write_2d_figure(
                    "high_eval/eval_loss_p_unloading",
                    high_eval_loss_p.data,
                    epoch * len_high_eval_datas + i
                )

            mean_high_eval_loss_p /= len_high_eval_datas

            tl_writer.write_2d_figure(
                "high_eval/mean_eval_loss_p_unloading", mean_high_eval_loss_p, epoch
            )

            if mean_high_eval_loss_p < min_high_eval_loss_p:
                min_high_eval_loss_p = mean_high_eval_loss_p

                saved_params = {
                    n: p for n, p in net.state_dict().items()
                }
                torch.save(
                    saved_params,
                    os.path.join(checkpoints_save_path, "best_model.pth")
                )
                logging.info(f"checkpoint saved at epoch {epoch} with loss {min_high_eval_loss_p}")

    if is_require_finetuning:
        optimizer_p_finetuning = optim.Adam(
            params=params_p_high.values(),
            lr=lr_p,
            weight_decay=weight_decay_p
        )

        scheduler_p_finetuning = CosineAnnealingLR(
            optimizer_p_finetuning,
            T_max=epochs,
            eta_min=lr_p / lr_p_decay_min
        )
        net = net.train()
        for epoch in range(finetuning_epoch):
            mean_high_train_loss_p = 0.
            for i, high_train_data in enumerate(high_train_datas):
                strip, mould, section, params,   unloaded_strip = \
                    list(
                        map(
                            lambda x: x.to(device), high_train_data
                        )
                    )
                high_pred_unloaded_strip = net(
                    strip, mould, section, params, fidelity=True)
                high_train_loss_p = loss_p(
                    unloaded_strip, high_pred_unloaded_strip, high_coordinate_weights)

                mean_high_train_loss_p += high_train_loss_p.data

                tl_writer.write_2d_figure(
                    "high_train/train_loss_p_unloading",
                    high_train_loss_p.data,
                    (epochs+epoch) * len_high_train_datas + i
                )

                min_high_train_loss_p = min(
                    min_high_train_loss_p,
                    high_train_loss_p.data
                )

                tl_writer.write_2d_figure(
                    "high_train/min_train_loss_p_unloading", min_high_train_loss_p, (epochs+epoch) * len_high_train_datas + i
                )

                optimizer_p_finetuning.zero_grad()
                high_train_loss_p.backward(retain_graph=True)
                optimizer_p_finetuning.step()
            scheduler_p_finetuning.step()
            mean_high_train_loss_p /= len_high_train_datas

            tl_writer.write_2d_figure(
                "high_train/mean_train_loss_p_unloading", mean_high_train_loss_p, epoch + epochs
            )

            net = net.eval()
            with torch.no_grad():
                mean_high_eval_loss_p = 0.
                for i, high_eval_data in enumerate(high_eval_datas):
                    strip, mould, section, params,   unloaded_strip = \
                        list(
                            map(
                                lambda x: x.to(device), high_eval_data
                            )
                        )
                    high_pred_unloaded_strip = net(
                        strip, mould, section, params, fidelity=True)
                    high_eval_loss_p = loss_p(
                        unloaded_strip, high_pred_unloaded_strip, high_coordinate_weights)

                    mean_high_eval_loss_p += high_eval_loss_p.data

                    tl_writer.write_2d_figure(
                        "high_eval/eval_loss_p_unloading",
                        high_eval_loss_p.data,
                        (epoch + epochs) * len_high_eval_datas + i
                    )

                mean_high_eval_loss_p /= len_high_eval_datas

                tl_writer.write_2d_figure(
                    "high_eval/mean_eval_loss_p_unloading",
                    mean_high_eval_loss_p,
                    epoch + epochs
                )

                if mean_high_eval_loss_p < min_high_eval_loss_p:
                    min_high_eval_loss_p = mean_high_eval_loss_p

                    saved_params = {
                        n: p for n, p in net.state_dict().items()
                    }
                    torch.save(
                        saved_params,
                        os.path.join(checkpoints_save_path, "best_model.pth")
                    )
                    logging.info(f"checkpoint saved at epoch {epoch} with loss {min_high_eval_loss_p}")


    logging.info("Training Finished")
    logging.info(f"min high eval loss p unloading: {min_high_eval_loss_p}")
    logging.info(f"min low eval loss p unloading: {min_low_eval_loss_p}")
    logging.info(f"min train loss r: {min_train_loss_r}")

def test(data_path, test_dataloader, device, exp_name, logs_save_path):
    coordinate_weights = [
        test_dataloader.scale_factor_for_unload[i] -
        test_dataloader.scale_factor_for_unload[i + 3]
        for i in range(3)
    ]

    test_datas = DataLoader(
        test_dataloader,
        batch_size=test_batch_size,
        num_workers=num_workers
    )

    data_save_path = os.path.join(data_path, f"pred_results/{exp_name}")

    pred_loaded_strip_path = os.path.join(
        data_save_path,
        "pred_loaded_strip_line"
    )
    pred_unloaded_strip_path = os.path.join(
        data_save_path,
        "pred_unloaded_strip_line"
    )
    recover_section_path = os.path.join(
        data_save_path,
        "recover_section"
    )

    os.makedirs(pred_loaded_strip_path, exist_ok=True)
    os.makedirs(pred_unloaded_strip_path, exist_ok=True)
    os.makedirs(recover_section_path, exist_ok=True)

    net = FilDeep().to(device)
    checkpoint_load_path = exp_name.replace("test", "train")
    checkpoint_load_path = os.path.join(
        checkpoints_path, checkpoint_load_path, "best_model.pth")

    parameters = torch.load(checkpoint_load_path, map_location=device)

    net_parameters = net.state_dict()

    for key, _ in net.named_parameters():
        if key not in parameters.keys():
            print(f"Error checkpoint file, missing parameter: {key}")

    net.load_state_dict(parameters)

    len_test_datas = len(test_datas)

    pred_loaded_strips, pred_unloaded_strips, recover_sections = [], [], []
    net = net.eval()
    with torch.no_grad():
        mean_test_loss_p = 0.

        for i, test_data in enumerate(test_datas):
            strip, mould, section, params, unloaded_strip = \
                list(
                    map(
                        lambda x: x.to(device), test_data
                    )
                )
            pred_unloaded_strip = net(
                strip, mould, section, params, fidelity=True)

            test_loss_p = loss_p(
                unloaded_strip, pred_unloaded_strip, coordinate_weights)


            mean_test_loss_p += test_loss_p.data

            pred_unloaded_strips.append(
                pred_unloaded_strip.cpu().detach().numpy())

        mean_test_loss_p /= len_test_datas
        logging.info("Test Loss")
        logging.info(
            f"mean loss for pred unloaded strip: {mean_test_loss_p}")
        logging.info("")

    pred_unloaded_strips = np.concatenate(pred_unloaded_strips, axis=0)

    for i in range(3):
        pred_unloaded_strips[:, i, :] = pred_unloaded_strips[:, i, :] * coordinate_weights[i] \
                                        + test_dataloader.scale_factor_for_unload[i + 3]

    def transform_and_save_strip(pred_strip, save_path, pred_type="unloading"):
        df = pd.DataFrame(columns=['type', 'id', 'last point', 'each sample'])

        logging.info(f"Test Error for Stage: {pred_type}")
        mean_test_dist_last_point, max_test_dist_last_point, min_test_dist_last_point = 0, 0, 1 << 30
        mean_sample_mean_test_dist, max_sample_mean_test_dist, min_sample_mean_test_dist = 0, 0, 1 << 30

        gt_path = test_dataloader.unload_line_paths

        for j in range(pred_strip.shape[0]):
            pred_points = diff_recover(pred_strip[j, :, :].T)
            with open(gt_path[j]) as f:
                gt_points = np.array(
                    list(
                        map(
                            lambda x: list(
                                map(
                                    lambda y: float(y),
                                    x.split()
                                )
                            ),
                            f.readlines()
                        )
                    )
                )
            distance = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))
            test_dist_last_point = np.sqrt(
                np.sum((pred_points[-1] - gt_points[-1]) ** 2))
            mean_test_dist_last_point += test_dist_last_point
            max_test_dist_last_point = max(
                max_test_dist_last_point, test_dist_last_point)
            min_test_dist_last_point = min(
                min_test_dist_last_point, test_dist_last_point)

            sample_mean_test_dist = np.mean(distance)
            max_sample_mean_test_dist = max(
                max_sample_mean_test_dist, sample_mean_test_dist)
            min_sample_mean_test_dist = min(
                min_sample_mean_test_dist, sample_mean_test_dist)

            mean_sample_mean_test_dist += sample_mean_test_dist

            type_idx = test_dataloader.mould_line_paths[j].find("type")
            type_id = test_dataloader.mould_line_paths[j][type_idx: type_idx + 6]
            type_path = os.path.join(save_path, type_id)

            logging.info(
                f"{test_dataloader.mould_line_paths[j][-8:]} {type_id} Distance of last point: {test_dist_last_point}")
            logging.info(
                f"{test_dataloader.mould_line_paths[j][-8:]} {type_id} Distance for each sample: {sample_mean_test_dist}")

            new_record = pd.DataFrame(
                [{'type': type_id, 'id': test_dataloader.mould_line_paths[j][-8:], 'last point': test_dist_last_point,
                  'each sample': sample_mean_test_dist}])
            df = pd.concat([df, new_record], ignore_index=True)

            if not os.path.exists(type_path):
                os.mkdir(type_path)
            file_path = os.path.join(
                str(type_path),
                str(test_dataloader.mould_line_paths[j][-8:])
            )
            with open(file_path, 'w', encoding="utf8") as w:
                for k in range(pred_points.shape[0]):
                    w.write(
                        f"{pred_points[k][0]} {pred_points[k][1]} {pred_points[k][2]}\n")
        mean_sample_mean_test_dist /= pred_strip.shape[0]
        mean_test_dist_last_point /= pred_strip.shape[0]

        logging.info(f"Test Error for Stage: {pred_type}")
        logging.info(
            f"mean distance of last point: {mean_test_dist_last_point}")
        logging.info(f"max distance of last point: {max_test_dist_last_point}")
        logging.info(f"min distance of last point: {min_test_dist_last_point}")
        logging.info(
            f"mean of distance for each sample: {mean_sample_mean_test_dist}")
        logging.info(
            f"max of distance for each sample: {max_sample_mean_test_dist}")
        logging.info(
            f"min of distance for each sample: {min_sample_mean_test_dist}")
        logging.info("")

        return df

    df_unloading = transform_and_save_strip(pred_unloaded_strips,
                                            pred_unloaded_strip_path, "unloading")
    unloading_save_path = os.path.join(logs_save_path, f"unloading.csv")

    df_unloading.to_csv(unloading_save_path, index=False)

def main():
    exp_name = f"{mode}_{mode_id}"
    device = torch.device(
        f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    logs_save_path = os.path.join(logs_path, exp_name)
    os.makedirs(logs_save_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=os.path.join(logs_save_path, "log.log"),
        filemode='w'
    )

    low_strip_path = os.path.join(low_data_path, "original_strip_line")
    low_mould_path = os.path.join(low_data_path, "mould_line")
    low_section_path = os.path.join(low_data_path, "strip_section_tiff")
    low_params_path = os.path.join(low_data_path, "stretch_bending_params")
    low_unload_path = os.path.join(low_data_path, "unload_strip_line")

    high_strip = os.path.join(high_data_path, "original_strip_line")
    high_mould_path = os.path.join(high_data_path, "mould_line")
    high_section_path = os.path.join(high_data_path, "strip_section_tiff")
    high_params_path = os.path.join(high_data_path, "stretch_bending_params")
    high_unload_path = os.path.join(high_data_path, "unload_strip_line")

    if mode == "train":
        low_train_dataloader = FilDeepDataLoader(
            strip_path=low_strip_path,
            mould_path=low_mould_path,
            section_path=low_section_path,
            params_path=low_params_path,
            unload_path=low_unload_path,
            mode="train",
            start_data_ratio=start_dataset_ratio,
            train_data_ratio=train_dataset_ratio,
            eval_data_ratio=eval_dataset_ratio
        )
        high_train_dataloader = FilDeepDataLoader(
            strip_path=high_strip,
            mould_path=high_mould_path,
            section_path=high_section_path,
            params_path=high_params_path,
            unload_path=high_unload_path,
            mode="train",
            start_data_ratio=high_start_dataset_ratio,
            train_data_ratio=high_train_dataset_ratio,
            eval_data_ratio=high_eval_dataset_ratio
        )
        low_eval_dataloader = FilDeepDataLoader(
            strip_path=low_strip_path,
            mould_path=low_mould_path,
            section_path=low_section_path,
            params_path=low_params_path,
            unload_path=low_unload_path,
            mode="eval",
            start_data_ratio=start_dataset_ratio,
            train_data_ratio=train_dataset_ratio,
            eval_data_ratio=eval_dataset_ratio
        )
        high_eval_dataloader = FilDeepDataLoader(
            strip_path=high_strip,
            mould_path=high_mould_path,
            section_path=high_section_path,
            params_path=high_params_path,
            unload_path=high_unload_path,
            mode="eval",
            start_data_ratio=high_start_dataset_ratio,
            train_data_ratio=high_train_dataset_ratio,
            eval_data_ratio=high_eval_dataset_ratio
        )
        tl_writer = Tensorboard_Logging(logs_save_path)
        train(
            low_train_dataloader,
            low_eval_dataloader,
            high_train_dataloader,
            high_eval_dataloader,
            device,
            exp_name,
            tl_writer,
        )
        tl_writer.writer_close()
    elif mode == "test":
        test_dataloader = FilDeepDataLoader(
            strip_path=high_strip,
            mould_path=high_mould_path,
            section_path=high_section_path,
            params_path=high_params_path,
            unload_path=high_unload_path,
            mode="test",
            start_data_ratio=start_dataset_ratio,
            train_data_ratio=train_dataset_ratio,
            eval_data_ratio=eval_dataset_ratio
        )
        test(
            high_data_path,
            test_dataloader,
            device,
            exp_name,
            logs_save_path
        )


if __name__ == "__main__":
    from configparser import ConfigParser

    config = ConfigParser()
    config.read("./config.ini", encoding="utf-8")
    mode = config.get("settings", "mode")
    mode_id = config.getint("settings", "mode_id")
    device_id = config.getint("settings", "device_id")
    low_data_path = config.get("settings", "low_data_path")
    high_data_path = config.get("settings", "high_data_path")
    checkpoints_path = config.get("settings", "checkpoints_path")
    logs_path = config.get("settings", "logs_path")
    num_workers = config.getint("settings", "num_workers")

    is_require_finetuning = config.getboolean("settings", "is_require_finetuning")
    finetuning_epoch = config.getint("settings", "finetuning_epoch")

    start_dataset_ratio = config.getfloat("settings", "start_dataset_ratio")
    train_dataset_ratio = config.getfloat("settings", "train_dataset_ratio")
    eval_dataset_ratio = config.getfloat("settings", "eval_dataset_ratio")
    high_start_dataset_ratio = config.getfloat("settings", "high_start_dataset_ratio")
    high_train_dataset_ratio = config.getfloat("settings", "high_train_dataset_ratio")
    high_eval_dataset_ratio = config.getfloat("settings", "high_eval_dataset_ratio")

    is_frozen_CL = config.getboolean("settings", "is_frozen_CL")
    is_frozen_CS = config.getboolean("settings", "is_frozen_CS")
    is_frozen_MP = config.getboolean("settings", "is_frozen_MP")
    is_frozen_WRFE = config.getboolean("settings", "is_frozen_WRFE")
    is_frozen_ECFE = config.getboolean("settings", "is_frozen_ECFE")
    is_frozen_low_Attention = config.getboolean("settings", "is_frozen_low_Attention")

    epochs = config.getint("hyper_parameters", "epochs")
    seed = config.getint("hyper_parameters", "seed")

    train_batch_size = config.getint("hyper_parameters", "train_batch_size")
    eval_batch_size = config.getint("hyper_parameters", "eval_batch_size")
    test_batch_size = config.getint("hyper_parameters", "test_batch_size")
    high_train_batch_size = config.getint("hyper_parameters", "high_train_batch_size")
    high_eval_batch_size = config.getint("hyper_parameters", "high_eval_batch_size")

    lr_p = config.getfloat("hyper_parameters", "lr_p")
    lr_r = config.getfloat("hyper_parameters", "lr_r")
    high_lr_p = config.getfloat("hyper_parameters", "high_lr_p")

    weight_decay_p = config.getfloat(
        "hyper_parameters", "weight_decay_p")
    weight_decay_r = config.getfloat("hyper_parameters", "weight_decay_r")
    high_weight_decay_p = config.getfloat(
        "hyper_parameters", "high_weight_decay_p")

    lr_p_decay_min = config.getfloat(
        "hyper_parameters", "lr_p_decay_min")
    lr_r_decay_min = config.getfloat("hyper_parameters", "lr_r_decay_min")
    high_lr_p_decay_min = config.getfloat(
        "hyper_parameters", "high_lr_p_decay_min")

    main()
