import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from utils.resample_3d_curve import resample_3d_curve

@dataclass
class FilDeepDataLoader(Dataset):
    strip_path: str
    mould_path: str
    section_path: str
    params_path: str
    unload_path: str
    high_strip_path: str = ""
    high_mould_path: str = ""
    high_section_path: str = ""
    high_params_path: str = ""
    high_unload_path: str = ""
    mode: str = "train"
    start_data_ratio: float = 0
    train_data_ratio: float = 8
    eval_data_ratio: float = 9
    n_type: int = 3
    n_points: float = 301
    numeric_type = np.float32

    def __post_init__(self):
        self.strip_line_paths = self._read_paths(self.strip_path)
        self.mould_line_paths = self._read_paths(self.mould_path)
        self.section_sdf_paths = self._read_paths(self.section_path)
        self.params_paths = self._read_paths(self.params_path)
        self.unload_line_paths = self._read_paths(self.unload_path)

        self.scale_factor_for_mould = []
        self.scale_factor_for_unload = []

        mould_line_paths = self._read_path_for_scale_factor(self.mould_path)
        unload_line_paths = self._read_path_for_scale_factor(self.unload_path)
        self._obtain_scale_factor(mould_line_paths, "mould")
        self._obtain_scale_factor(unload_line_paths, "unload")

    def _obtain_scale_factor(self, path_list, flag):
        diff_x_max_all = -10
        diff_y_max_all = -10
        diff_z_max_all = -10
        diff_x_min_all = 10
        diff_y_min_all = 10
        diff_z_min_all = 10
        for path in path_list:
            point = self.read_2d_array_from_txt(path)
            resampled_curve = resample_3d_curve(point, self.n_points)
            line_diff = np.diff(resampled_curve, axis = 0)
            diff_x_max = np.max(line_diff[:, 0])
            diff_y_max = np.max(line_diff[:, 1])
            diff_z_max = np.max(line_diff[:, 2])
            diff_x_min = np.min(line_diff[:, 0])
            diff_y_min = np.min(line_diff[:, 1])
            diff_z_min = np.min(line_diff[:, 2])
            diff_x_max_all = max(diff_x_max, diff_x_max_all)
            diff_y_max_all = max(diff_y_max, diff_y_max_all)
            diff_z_max_all = max(diff_z_max, diff_z_max_all)
            diff_x_min_all = min(diff_x_min, diff_x_min_all)
            diff_y_min_all = min(diff_y_min, diff_y_min_all)
            diff_z_min_all = min(diff_z_min, diff_z_min_all)
        diff_x_max_all = round(diff_x_max_all, 4)
        diff_y_max_all = round(diff_y_max_all, 4)
        diff_z_max_all = round(diff_z_max_all, 4)
        diff_x_min_all = round(diff_x_min_all, 4)
        diff_y_min_all = round(diff_y_min_all, 4)
        diff_z_min_all = round(diff_z_min_all, 4)
        if flag == "mould":
            self.scale_factor_for_mould = [diff_x_max_all, diff_y_max_all, diff_z_max_all, diff_x_min_all, diff_y_min_all, diff_z_min_all]
        elif flag == "unload":
            self.scale_factor_for_unload = [diff_x_max_all, diff_y_max_all, diff_z_max_all, diff_x_min_all, diff_y_min_all, diff_z_min_all]
        else:
            print(f"flag error: {flag} is not in ['mould', 'unload']")
        print(flag, [diff_x_max_all, diff_y_max_all, diff_z_max_all, diff_x_min_all, diff_y_min_all, diff_z_min_all])


    def _data_split(self, n):
        if self.mode == "train" or self.mode == "train_finetuning":
            start, end = int(n // 10 * self.start_data_ratio), int(n // 10 * self.train_data_ratio)
        elif self.mode == "eval" or self.mode == "eval_finetuning":
            start, end = int(n // 10 * self.train_data_ratio), int(n // 10 * self.eval_data_ratio)
        elif self.mode == "test":
            start, end = int(n // 10 * self.eval_data_ratio), n
        elif self.mode == "additional":
            start, end = int(n // 10 * self.train_data_ratio), int(n // 10 * self.eval_data_ratio)
        else:
            raise "Mode type error!"
        return start, end

    @staticmethod
    def read_2d_array_from_txt(line_path):
        with open(line_path, 'r', encoding = "utf8") as f:
            points = np.array(
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
        return points

    def _read_paths(self, path):
        paths = []
        for i in range(1, self.n_type+1):
            type_path = os.path.join(path, f"type_{i}")
            files = os.listdir(type_path)
            start, end = self._data_split(len(files))
            files.sort()
            files = list(map(lambda x: os.path.join(type_path, x), files))
            paths += files[start: end]

        print(len(paths))
        return paths

    def _read_path_for_scale_factor(self, path):
        paths = []
        for i in range(1, self.n_type+1):
            type_path = os.path.join(path, f"type_{i}")
            files = os.listdir(type_path)
            files.sort()
            files = list(map(lambda x: os.path.join(type_path, x), files))
            paths += files

        return paths

    def _read_line_diff(self, path, scale_factor_type = "unloading"):
        line = self.read_2d_array_from_txt(path)
        if len(line) != self.n_points:
            line = resample_3d_curve(line, self.n_points)
        line_diff = np.diff(line, axis = 0)
        if scale_factor_type == "unloading":
            scale_factor = self.scale_factor_for_unload
        else:
            scale_factor = self.scale_factor_for_mould

        for i in range(3):
            line_diff[:, i] = (line_diff[:, i] - scale_factor[i + 3]) / (scale_factor[i] - scale_factor[i + 3])

        return line_diff

    def _read_section_sdf(self, path):
        return np.expand_dims(cv2.imread(path, cv2.IMREAD_UNCHANGED), axis = 0)

    def _read_section_sdf_deprecated(self, path):
        sdf = self.read_2d_array_from_txt(path)
        min_val, max_val = np.min(sdf), np.max(sdf)
        normalized_sdf = (sdf - min_val) / (max_val - min_val)
        normalized_sdf = np.expand_dims(normalized_sdf, axis = 0)
        return normalized_sdf

    def _read_params(self, path):
        params = np.loadtxt(path, delimiter = ",")
        params = np.expand_dims(np.sum(params, axis = 0), axis = 0)
        min_val, max_val = np.min(params[0, :3]), np.max(params[0, :3])
        params[0, :3] = (params[0, :3] - min_val) / (max_val - min_val)
        min_val, max_val = np.min(params[0, 3:]), np.max(params[0, 3:])
        params[0, 3:] = (params[0, 3:] - min_val) / (max_val - min_val)
        return params

    def __getitem__(self, index):
        strip_line_diff = self._read_line_diff(
            self.strip_line_paths[index]
        ).T
        mould_line_diff = self._read_line_diff(
            self.mould_line_paths[index],
            scale_factor_type = "mould"
        ).T
        unload_line_diff = self._read_line_diff(
            self.unload_line_paths[index]
        ).T
        section_sdf = self._read_section_sdf(
            self.section_sdf_paths[index]
        )
        params = self._read_params(
            self.params_paths[index]
        )

        return \
            list(
                map(
                    lambda x: x.astype(self.numeric_type),
                    (
                        strip_line_diff,
                        mould_line_diff,
                        section_sdf,
                        params,
                        unload_line_diff
                    )
                )
            )

    def __len__(self):
        return len(self.strip_line_paths)
