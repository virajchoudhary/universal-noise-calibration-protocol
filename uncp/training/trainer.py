"""Full UNCP pipeline — orchestrates NSA → CNI → phased retraining → SRD."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from baselines.erm import ERMTrainer
from uncp.cni.calibrator import CNICalibrator, CalibrationConfig
from uncp.cni.noise_schedules import ThreePhaseSchedule, create_schedule_from_config
from uncp.models import build_model
from uncp.nsa import NoiseSensitivityProfile, SensitivityProbe, get_noise_generators
from uncp.nsa.noise_generators import NoiseGenerator
from uncp.utils import get_device, save_json, set_seed, timestamped_dir


class UNCPPipeline:
    """Stage 1 (NSA) → Stage 2 (CNI) → Stage 3 (phased retrain) → Stage 4 (validation)."""

    def __init__(
        self,
        config: DictConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        shifted_test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        domain: str = "vision",
        run_dir: Optional[Path] = None,
        run_name: str = "uncp",
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.shifted_test_loader = shifted_test_loader
        self.device = device or get_device(str(config.get("device", "auto")))
        self.domain = domain
        self.run_name = run_name
        self.run_dir = Path(run_dir) if run_dir else timestamped_dir("./results", prefix=run_name)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._model_factory = model_factory or self._default_model_factory

    # ------------------------------------------------------------------
    def _default_model_factory(self) -> nn.Module:
        return build_model(self.config.model.name,
                           num_classes=int(self.config.model.num_classes))

    # ------------------------------------------------------------------
    # STAGE 1 — NSA
    # ------------------------------------------------------------------
    def stage1_nsa(self, erm_model: nn.Module) -> NoiseSensitivityProfile:
        print("[stage 1] NSA — probing ERM model")
        generators = get_noise_generators(self.domain,
                                          list(self.config.nsa.noise_types))
        probe = SensitivityProbe(
            model=erm_model, noise_generators=generators,
            num_samples=int(self.config.nsa.num_samples),
            device=self.device,
            batch_size=int(self.config.training.batch_size),
            model_name=f"{self.run_name}_erm", domain=self.domain,
        )
        nsp = probe.probe(self.test_loader)
        nsp.save(self.run_dir / "nsp_before.pkl")
        print(f"[stage 1] vulnerability ranking: "
              f"{[(nt, round(r, 3)) for nt, r in nsp.get_vulnerability_ranking()]}")
        return nsp

    # ------------------------------------------------------------------
    # STAGE 2 — CNI
    # ------------------------------------------------------------------
    def stage2_cni(self, nsp: NoiseSensitivityProfile) -> tuple[CalibrationConfig, ThreePhaseSchedule]:
        print("[stage 2] CNI — calibrating noise")
        calibrator = CNICalibrator(
            nsp,
            calibration_method=str(self.config.cni.calibration_method),
            target_flip_rate=float(self.config.cni.target_flip_rate),
        )
        cal_cfg = calibrator.calibrate()
        cal_cfg.save(self.run_dir / "calibration.pkl")
        print(cal_cfg.describe())

        total = int(self.config.training.epochs)
        phase_ratios = (
            float(self.config.training.phase_a_epochs) / total,
            float(self.config.training.phase_b_epochs) / total,
            float(self.config.training.phase_c_epochs) / total,
        )
        schedule = create_schedule_from_config(
            cal_cfg, total_epochs=total, phase_ratios=phase_ratios,
            annealing=str(self.config.training.get("annealing", "cosine")),
        )
        try:
            schedule.visualize(save_path=self.run_dir / "schedule.pdf")
        except Exception as exc:
            print(f"[warn] schedule plot failed: {exc}")
        return cal_cfg, schedule

    # ------------------------------------------------------------------
    # STAGE 3 — Retraining
    # ------------------------------------------------------------------
    def stage3_retrain(
        self,
        calibration: CalibrationConfig,
        schedule: ThreePhaseSchedule,
        initial_state: Optional[dict] = None,
    ) -> tuple[nn.Module, List[Dict[str, Any]]]:
        print(f"[stage 3] retraining with noise='{calibration.recommended_noise_type}' "
              f"σ∈[{schedule.sigma_low:.2f},{schedule.sigma_high:.2f}] "
              f"phases={schedule.phase_a_epochs}/{schedule.phase_b_epochs}/"
              f"{schedule.phase_c_epochs}")

        generators = get_noise_generators(self.domain,
                                          [calibration.recommended_noise_type])
        noise_gen: NoiseGenerator = generators[calibration.recommended_noise_type]

        model = self._model_factory().to(self.device)
        if initial_state is not None:
            model.load_state_dict(initial_state)

        optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.training.lr),
            weight_decay=float(self.config.training.weight_decay),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(schedule.total_epochs, 1),
        )

        history: List[Dict[str, Any]] = []
        best_wga = -1.0
        best_state: Optional[dict] = None
        for epoch in range(schedule.total_epochs):
            phase = schedule.get_phase(epoch)
            sigma = schedule.get_magnitude(epoch)
            model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for batch in self.train_loader:
                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device)
                if sigma > 1e-6:
                    x = noise_gen.apply(x, float(sigma))
                optim.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optim.step()
                loss_sum += float(loss.item()) * x.size(0)
                correct += int((logits.argmax(1) == y).sum().item())
                total += x.size(0)
            sched.step()

            val = self._eval(model, self.val_loader)
            log = {
                "epoch": epoch, "phase": phase, "sigma": sigma,
                "train_loss": loss_sum / max(total, 1),
                "train_acc": correct / max(total, 1),
                "val_acc": val["acc"],
                "val_wga": val["worst_group_acc"],
            }
            history.append(log)
            wga = val["worst_group_acc"] or 0.0
            if wga > best_wga:
                best_wga = wga
                best_state = copy.deepcopy(model.state_dict())
            print(f"  ep {epoch:02d} [{phase:<15s}] σ={sigma:.3f} "
                  f"tr_acc={log['train_acc']*100:.2f} val_acc={val['acc']*100:.2f} "
                  f"val_wga={wga*100:.2f}")

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save({"model": best_state, "best_wga": best_wga},
                       self.checkpoint_dir / "uncp_best.pt")
        return model, history

    # ------------------------------------------------------------------
    # STAGE 4 — Validation
    # ------------------------------------------------------------------
    def stage4_validate(
        self, model: nn.Module, nsp_before: NoiseSensitivityProfile,
    ) -> Dict[str, Any]:
        print("[stage 4] re-probing with NSA on UNCP-trained model")
        generators = get_noise_generators(self.domain,
                                          list(self.config.nsa.noise_types))
        probe = SensitivityProbe(
            model=model, noise_generators=generators,
            num_samples=int(self.config.nsa.num_samples),
            device=self.device,
            batch_size=int(self.config.training.batch_size),
            model_name=f"{self.run_name}_uncp", domain=self.domain,
        )
        nsp_after = probe.probe(self.test_loader)
        nsp_after.save(self.run_dir / "nsp_after.pkl")

        test = self._eval(model, self.test_loader)
        shifted = self._eval(model, self.shifted_test_loader) \
            if self.shifted_test_loader else None

        rank_before = dict(nsp_before.get_vulnerability_ranking())
        rank_after = dict(nsp_after.get_vulnerability_ranking())
        delta_nsp = {nt: rank_after.get(nt, 0.0) - rank_before.get(nt, 0.0)
                     for nt in rank_before}
        return {
            "test_id": test,
            "test_ood": shifted,
            "nsp_after_ranking": nsp_after.get_vulnerability_ranking(),
            "delta_nsp": delta_nsp,
            "nsp_after_path": str(self.run_dir / "nsp_after.pkl"),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels, all_groups = [], [], []
        for batch in loader:
            x = batch["image"].to(self.device); y = batch["label"].to(self.device)
            g = batch.get("group_label")
            preds = model(x).argmax(1)
            correct += int((preds == y).sum().item())
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            if g is not None:
                all_groups.append(g)
        preds = torch.cat(all_preds); labels = torch.cat(all_labels)
        group_accs: Dict[int, float] = {}
        if all_groups:
            groups = torch.cat(all_groups)
            for gr in torch.unique(groups).tolist():
                m = (groups == gr)
                if m.any():
                    group_accs[int(gr)] = float((preds[m] == labels[m]).float().mean().item())
        wga = min(group_accs.values()) if group_accs else None
        return {"acc": correct / max(total, 1),
                "per_group_acc": group_accs, "worst_group_acc": wga}

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run_full_pipeline(self, warm_start_model: Optional[nn.Module] = None,
                          warm_start_epochs: int = 5) -> Dict[str, Any]:
        set_seed(int(self.config.get("seed", 42)))
        start = time.time()

        # --- stage 0: ERM warm-start if no model given
        if warm_start_model is None:
            print(f"[stage 0] training ERM warm-start for {warm_start_epochs} epochs")
            warm_model = self._model_factory()
            warm_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
            warm_cfg.training.epochs = warm_start_epochs
            erm = ERMTrainer(
                model=warm_model,
                train_loader=self.train_loader, val_loader=self.val_loader,
                test_loader=self.test_loader, shifted_test_loader=self.shifted_test_loader,
                config=warm_cfg, device=self.device,
                run_name=f"{self.run_name}_warmstart",
                checkpoint_dir=str(self.checkpoint_dir),
                use_wandb=False,
            )
            erm_results = erm.run()
            save_json(erm_results, self.run_dir / "erm_warmstart.json")
            erm_model = erm.model
            erm_test_id = erm_results["test"]
            erm_test_ood = erm_results["shifted_test"]
        else:
            erm_model = warm_start_model
            erm_test_id = self._eval(erm_model, self.test_loader)
            erm_test_ood = self._eval(erm_model, self.shifted_test_loader) \
                if self.shifted_test_loader else None

        # --- stages 1–4
        nsp_before = self.stage1_nsa(erm_model)
        cal_cfg, schedule = self.stage2_cni(nsp_before)
        uncp_model, history = self.stage3_retrain(cal_cfg, schedule,
                                                  initial_state=erm_model.state_dict())
        validation = self.stage4_validate(uncp_model, nsp_before)

        elapsed = time.time() - start
        results = {
            "run_name": self.run_name,
            "elapsed_sec": elapsed,
            "erm_test_id": erm_test_id,
            "erm_test_ood": erm_test_ood,
            "calibration": {
                "recommended_noise_type": cal_cfg.recommended_noise_type,
                "sigma_spurious": cal_cfg.sigma_spurious,
                "sigma_low": cal_cfg.sigma_low,
                "sigma_high": cal_cfg.sigma_high,
                "confidence": cal_cfg.confidence,
                "metadata": cal_cfg.metadata,
            },
            "schedule": schedule.get_config_snapshot(),
            "history": history,
            "validation": validation,
        }
        save_json(results, self.run_dir / "results.json")
        return results

    def resume_from_checkpoint(self, path: str | Path) -> nn.Module:
        state = torch.load(path, map_location=self.device, weights_only=False)
        model = self._model_factory().to(self.device)
        model.load_state_dict(state["model"])
        return model


class UNCPPipelineVision(UNCPPipeline):
    """Vision-specific defaults + ResNet/ViT helpers."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("domain", "vision")
        super().__init__(*args, **kwargs)

    @staticmethod
    def make_resnet18_small(num_classes: int = 10) -> nn.Module:
        return build_model("resnet18_small", num_classes=num_classes)

    @staticmethod
    def make_resnet18_imagenet(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
        return build_model("resnet18_imagenet", num_classes=num_classes,
                           pretrained=pretrained)
