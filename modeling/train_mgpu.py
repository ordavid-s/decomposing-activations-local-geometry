from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
from tqdm import tqdm
from pathlib import Path

def _cpu_state_dict(model):
    # Safe copy of weights to CPU (so we don’t pin VRAM)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def _eval_nll_distributed(model, loader, accelerator):
    model.eval()
    tot_nll_local, tot_n_local = 0.0, 0
    device = accelerator.device

    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.view(x.size(0), -1).to(device, non_blocking=True)

        nll = model(x)
        B = x.size(0)
        sum_nllB = nll.detach() * B
        g_sum_nllB = accelerator.gather_for_metrics(sum_nllB)
        g_sum_B    = accelerator.gather_for_metrics(torch.tensor(B, device=device))

        tot_nll_local += g_sum_nllB.sum().item()
        tot_n_local   += g_sum_B.sum().item()
    model.train()
    return (tot_nll_local / max(1, tot_n_local)) if tot_n_local > 0 else float("nan")

def train_nll(
    model,
    loader,
    *,
    val_loader=None,
    epochs=5,
    lr=1e-3,
    grad_clip=None,
    save_path=None,
    save_func=None,
    log_interval=100,
    val_interval=100_000,
    steps_per_epoch=None,
    wandb_project: str | None = None, 
    run_name: str | None = None,
    start_epoch = 1
):
    """
    Accelerate-powered NLL training (multi-GPU / multi-node).
    - Works with iterable loaders; respects `steps_per_epoch`.
    - Proper distributed reduction for validation metrics.
    - Saves from the main process only (using your save_func if provided).
    """
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
        use_orig_params=True,
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, log_with="wandb" if wandb_project else None)

    if wandb_project:
        accelerator.init_trackers(
            project_name=wandb_project or "train_nll",
            config={
                "lr": lr, "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
            },
            init_kwargs={
                "wandb": {
                    "name": run_name or "train_nll",           # uses WANDB_PROJECT if you prefer
                }
            },
        )
    
    device = accelerator.device

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare everything for distributed training
    if val_loader is not None:
        model, opt, loader, val_loader = accelerator.prepare(model, opt, loader, val_loader)
    else:
        model, opt, loader = accelerator.prepare(model, opt, loader)


    best_metric = float("inf")
    # keep a CPU copy of the best weights (from unwrapped model)
    best_state  = _cpu_state_dict(accelerator.unwrap_model(model))
    best_epoch  = 0
    global_step = 0

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_nll_running, total_n_running = 0.0, 0

        iterable = enumerate(loader, 1)
        pbar = tqdm(iterable,
                    total=steps_per_epoch,
                    disable=not accelerator.is_local_main_process)

        for batch_idx, batch in pbar:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            nll = model(x)              # mean over this process's batch
            accelerator.backward(nll)
            
            if grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            # running (local) average for nice progress bars
            B = x.size(0)
            total_nll_running += float(nll.detach().item()) * B
            total_n_running   += B
            global_step += 1
            if (batch_idx % log_interval) == 0:
                avg_so_far = total_nll_running / max(1, total_n_running)
                if accelerator.is_local_main_process:
                    pbar.set_description(f"Epoch {ep:02d} | Step {batch_idx:06d} Train NLL={avg_so_far:.6f}")
                if wandb_project and accelerator.is_main_process:  # ← main process for logging
                    accelerator.log({"train_nll": avg_so_far, "epoch": ep, "step": global_step})

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            del x, nll  # free ASAP
            if global_step % val_interval == 0:
                model.eval()
                # Train NLL (local running avg just for log)
                avg_train_nll = (total_nll_running / max(1, total_n_running)) if total_n_running > 0 else float("nan")

                # Validation with proper global reduction
                if val_loader is not None:
                    val_nll = _eval_nll_distributed(model, val_loader, accelerator)
                    select_metric = val_nll
                else:
                    val_nll = float("nan")
                    select_metric = avg_train_nll
                model.train()

                improved = (select_metric < best_metric) if not (torch.isnan(torch.tensor(select_metric))) else False

                if accelerator.is_main_process:
                    if wandb_project:
                        accelerator.log({"epoch": ep, "train_nll_epoch": avg_train_nll, "val_nll": val_nll}, step=global_step)
                    if improved:
                        best_metric = select_metric
                        best_state  = _cpu_state_dict(accelerator.unwrap_model(model))
                        best_epoch  = ep
                        if save_path and save_func:
                            p = Path(save_path)
                            final_save_path = p.parent / f"{ep}_{p.name}"
                            save_func(model, str(final_save_path), accelerator=accelerator)
                    print(
                        f"[epoch {ep:02d}] "
                        f"train NLL={avg_train_nll:.6f}  "
                        f"val NLL={val_nll:.6f} "
                        f"{'** best **' if improved else ''}"
                    )
                # Make sure every process waits here (keeps epochs in lockstep)
                accelerator.wait_for_everyone()

    # Restore best state on all processes
    accelerator.unwrap_model(model).load_state_dict(best_state)
    if accelerator.is_main_process:
        print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")
        
    if getattr(accelerator, "trackers", None):
        for t in accelerator.trackers:
            t.finish()
    return dict(best_epoch=best_epoch, best_metric=best_metric)
