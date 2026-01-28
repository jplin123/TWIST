import csv
from typing import Dict, List

import torch


class UpperBodyCSVPlayer:
    """Utility that streams upper-body joint targets from a CSV file."""

    def __init__(
        self,
        file_path: str,
        joint_mapping: Dict[str, int],
        playback_rate_hz: float,
        default_pose: Dict[str, float],
        loop: bool = True,
        interpolate: bool = True,
        device: str = "cpu",
    ) -> None:
        self.file_path = file_path
        self.device = torch.device(device)
        self.playback_rate = playback_rate_hz
        self.dt = 1.0 / playback_rate_hz
        self.loop = loop
        self.interpolate = interpolate
        self.joint_names = list(joint_mapping.keys())
        self.num_joints = len(self.joint_names)
        self.default_pose = torch.tensor(
            [default_pose.get(name, 0.0) for name in self.joint_names],
            dtype=torch.float,
            device=self.device,
        )

        self._frame_targets = self._load_frames(file_path, joint_mapping)
        self.num_frames = self._frame_targets.shape[0]
        self.total_duration = self.num_frames * self.dt

    def _load_frames(
        self, file_path: str, joint_mapping: Dict[str, int]
    ) -> torch.Tensor:
        with open(file_path, "r", newline="") as fh:
            reader = csv.reader(fh)
            rows: List[List[float]] = []
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                rows.append([float(val) for val in row])
        if not rows:
            raise ValueError(f"Upper-body CSV {file_path} is empty.")
        data = torch.tensor(rows, dtype=torch.float, device=self.device)
        if header:
            header_map = {name.strip(): idx for idx, name in enumerate(header)}
        else:
            header_map = {}
        column_indices: List[int] = []
        for joint_name, column in joint_mapping.items():
            if isinstance(column, str):
                if column not in header_map:
                    raise KeyError(
                        f"CSV {file_path} does not contain a column named '{column}'."
                    )
                column_indices.append(header_map[column])
            else:
                column_indices.append(column)
        cols = torch.tensor(column_indices, dtype=torch.long, device=self.device)
        return data.index_select(1, cols)

    def sample(
        self,
        time_s: torch.Tensor,
        amplitude: torch.Tensor,
        speed_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Returns interpolated joint targets for each env."""
        if time_s.ndim == 0:
            time_s = time_s.unsqueeze(0)
        if amplitude.ndim == 0:
            amplitude = amplitude.unsqueeze(0)
        if speed_scale.ndim == 0:
            speed_scale = speed_scale.unsqueeze(0)

        scaled_time = time_s * speed_scale
        frame_float = scaled_time * self.playback_rate
        frame_idx = torch.floor(frame_float).long()
        frac = frame_float - frame_idx.float()

        if self.loop:
            frame_idx = torch.remainder(frame_idx, self.num_frames)
            next_idx = torch.remainder(frame_idx + 1, self.num_frames)
        else:
            clamped = torch.clamp(frame_idx, 0, self.num_frames - 1)
            next_idx = torch.clamp(clamped + 1, 0, self.num_frames - 1)
            frame_idx = clamped

        current = self._frame_targets[frame_idx]
        if self.interpolate:
            next_frame = self._frame_targets[next_idx]
            interp = current + (next_frame - current) * frac.unsqueeze(1)
        else:
            interp = current

        targets = self.default_pose + amplitude.unsqueeze(1) * (interp - self.default_pose)
        return targets
