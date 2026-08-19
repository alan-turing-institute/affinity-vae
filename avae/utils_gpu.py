import logging

import lightning as lt
import torch


def set_device(gpu: bool) -> torch.device:
    """Set the torch device to use for training and inference.

    Parameters
    ----------
    gpu: bool
        If True, the model will be trained on GPU.

    Returns
    -------
    device: torch.device

    """
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        logging.warning(
            "\n\nWARNING: no GPU available, running on CPU instead.\n"
        )
    return device


def parse_requested_cuda_devices(
    gpu_devices: str | None, n_visible: int
) -> list[int]:
    """Parse and validate requested CUDA device indices."""
    if gpu_devices is None or gpu_devices.strip() == "":
        return list(range(n_visible))

    requested: list[int] = []
    for token in gpu_devices.split(','):
        item = token.strip()
        if item == '':
            continue
        try:
            idx = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid gpu_devices value '{item}'. Use comma-separated integer indices."
            ) from exc

        if idx < 0 or idx >= n_visible:
            raise ValueError(
                f"Requested CUDA device index {idx} is out of range for {n_visible} visible devices."
            )
        requested.append(idx)

    unique_requested = sorted(set(requested))
    if not unique_requested:
        raise ValueError(
            "gpu_devices was provided but no valid CUDA indices were parsed."
        )

    return unique_requested


def probe_usable_cuda_devices(candidate_devices: list[int]) -> list[int]:
    """Return CUDA device indices that can be selected and used."""
    usable_devices: list[int] = []

    for idx in candidate_devices:
        try:
            torch.cuda.set_device(idx)
            # Force a tiny allocation so we catch runtime availability issues.
            _ = torch.empty(1, device=f"cuda:{idx}")
            usable_devices.append(idx)
        except Exception as exc:
            logging.warning(
                "Skipping CUDA device index %s due to availability error: %s",
                idx,
                exc,
            )

    return usable_devices


def setup_gpus(
    use_gpu: bool,
    gpu_devices: str | None,
    strategy: str,
) -> lt.Fabric:
    """Build a Lightning Fabric object with CPU/GPU settings.

    This preserves the existing behavior for manual/auto GPU selection,
    probing usable CUDA devices, and strategy fallback on single device.
    """
    n_devices = torch.cuda.device_count()
    logging.info('GPus available: {}'.format(n_devices))

    if n_devices > 0 and use_gpu is True:
        accelerator = 'gpu'
        manual_device_selection = (
            gpu_devices is not None and gpu_devices.strip() != ""
        )
        requested_devices = parse_requested_cuda_devices(
            gpu_devices=gpu_devices,
            n_visible=n_devices,
        )

        if manual_device_selection:
            usable_devices = requested_devices
        else:
            usable_devices = probe_usable_cuda_devices(requested_devices)

        if not usable_devices:
            raise RuntimeError(
                "No usable CUDA devices were found. Check CUDA_VISIBLE_DEVICES, scheduler GPU allocation, and competing processes."
            )

        n_devices = len(usable_devices)
        if (not manual_device_selection) and n_devices < len(
            requested_devices
        ):
            logging.warning(
                "Only %s/%s requested CUDA devices are usable. Launching on usable devices: %s",
                n_devices,
                len(requested_devices),
                usable_devices,
            )

        if manual_device_selection:
            logging.info(
                "GPU manual mode enabled. Launching with requested CUDA devices: %s",
                usable_devices,
            )

        # Tensor Core GPUs benefit from higher FP32 matmul throughput settings.
        torch.set_float32_matmul_precision('high')

        selected_strategy = strategy
        if n_devices == 1 and strategy in {"ddp", "deepspeed", "fsdp"}:
            logging.warning(
                "Requested strategy '%s' requires multi-device execution; falling back to 'auto' for single usable GPU.",
                strategy,
            )
            selected_strategy = 'auto'

        if n_devices <= 4:
            n_nodes = 1
        else:
            # Baskerville has 4 GPUs per node.
            n_nodes = (n_devices + 3) // 4

        logging.info(
            f'Setting up fabric with strategy {selected_strategy}, accelerator {accelerator}, devices {usable_devices}, num_nodes {n_nodes}'
        )
        return lt.Fabric(
            strategy=selected_strategy,
            accelerator=accelerator,
            devices=usable_devices,
            num_nodes=n_nodes,
        )

    if use_gpu and n_devices == 0:
        logging.warning(
            "GPU requested but no CUDA devices are visible. Falling back to CPU.",
        )
    return lt.Fabric(strategy='auto', accelerator='cpu')
