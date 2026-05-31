from dataclasses import dataclass, field


@dataclass
class TrainingSummaryTracker:
    best_train_loss: float = float("inf")
    best_test_loss: float = float("inf")
    last_train_loss: float | None = None
    last_test_loss: float | None = None
    best_train_loss_epoch: int | None = None
    best_train_loss_iter: int | None = None
    best_test_loss_epoch: int | None = None
    best_test_loss_iter: int | None = None

    best_fid: float = float("inf")
    best_fvd: float = float("inf")
    last_fid: float | None = None
    last_fvd: float | None = None
    best_fid_epoch: int | None = None
    best_fid_iter: int | None = None
    best_fvd_epoch: int | None = None
    best_fvd_iter: int | None = None

    # Curves for post-hoc analysis across long runs.
    train_loss_curve: list[dict] = field(default_factory=list)
    test_loss_curve: list[dict] = field(default_factory=list)
    fid_curve: list[dict] = field(default_factory=list)
    fvd_curve: list[dict] = field(default_factory=list)

    def update_losses(self, train_loss: float, test_loss: float, epoch: int, iteration: int) -> None:
        self.last_train_loss = train_loss
        self.last_test_loss = test_loss

        self.train_loss_curve.append({"epoch": epoch, "iter": iteration, "value": float(train_loss)})
        self.test_loss_curve.append({"epoch": epoch, "iter": iteration, "value": float(test_loss)})

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            self.best_train_loss_epoch = epoch
            self.best_train_loss_iter = iteration

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_test_loss_epoch = epoch
            self.best_test_loss_iter = iteration

    def update_generation_metrics(self, metrics: dict, epoch: int, iteration: int) -> None:
        if "FID" in metrics:
            fid_value = float(metrics["FID"])
            self.last_fid = fid_value
            self.fid_curve.append({"epoch": epoch, "iter": iteration, "value": fid_value})
            if fid_value < self.best_fid:
                self.best_fid = fid_value
                self.best_fid_epoch = epoch
                self.best_fid_iter = iteration

        if "FVD" in metrics:
            fvd_value = float(metrics["FVD"])
            self.last_fvd = fvd_value
            self.fvd_curve.append({"epoch": epoch, "iter": iteration, "value": fvd_value})
            if fvd_value < self.best_fvd:
                self.best_fvd = fvd_value
                self.best_fvd_epoch = epoch
                self.best_fvd_iter = iteration

    @classmethod
    def from_dict(cls, data: dict):
        tracker = cls()

        tracker.best_train_loss = data.get("best_train_loss", float("inf")) if data.get("best_train_loss") is not None else float("inf")
        tracker.best_test_loss = data.get("best_test_loss", float("inf")) if data.get("best_test_loss") is not None else float("inf")
        tracker.last_train_loss = data.get("last_train_loss")
        tracker.last_test_loss = data.get("last_test_loss")
        tracker.best_train_loss_epoch = data.get("best_train_loss_epoch")
        tracker.best_train_loss_iter = data.get("best_train_loss_iter")
        tracker.best_test_loss_epoch = data.get("best_test_loss_epoch")
        tracker.best_test_loss_iter = data.get("best_test_loss_iter")

        tracker.best_fid = data.get("best_fid", float("inf")) if data.get("best_fid") is not None else float("inf")
        tracker.best_fvd = data.get("best_fvd", float("inf")) if data.get("best_fvd") is not None else float("inf")
        tracker.last_fid = data.get("last_fid")
        tracker.last_fvd = data.get("last_fvd")
        tracker.best_fid_epoch = data.get("best_fid_epoch")
        tracker.best_fid_iter = data.get("best_fid_iter")
        tracker.best_fvd_epoch = data.get("best_fvd_epoch")
        tracker.best_fvd_iter = data.get("best_fvd_iter")

        curves = data.get("curves", {})
        tracker.train_loss_curve = list(curves.get("train_loss", []))
        tracker.test_loss_curve = list(curves.get("test_loss", []))
        tracker.fid_curve = list(curves.get("fid", []))
        tracker.fvd_curve = list(curves.get("fvd", []))

        return tracker

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return None if value == float("inf") else value

    def to_dict(
        self,
        run_id: str,
        run_start,
        run_end,
        run_duration_sec: int,
        run_config: dict,
        final_iter: int,
        final_epoch: int,
    ) -> dict:
        return {
            "run_id": run_id,
            "run_start": run_start.isoformat(timespec="seconds"),
            "run_end": run_end.isoformat(timespec="seconds"),
            "run_duration_sec": run_duration_sec,
            "run_config": run_config,
            "last_train_loss": self.last_train_loss,
            "best_train_loss": self._finite_or_none(self.best_train_loss),
            "best_train_loss_epoch": self.best_train_loss_epoch,
            "best_train_loss_iter": self.best_train_loss_iter,
            "last_test_loss": self.last_test_loss,
            "best_test_loss": self._finite_or_none(self.best_test_loss),
            "best_test_loss_epoch": self.best_test_loss_epoch,
            "best_test_loss_iter": self.best_test_loss_iter,
            "last_fid": self.last_fid,
            "best_fid": self._finite_or_none(self.best_fid),
            "best_fid_epoch": self.best_fid_epoch,
            "best_fid_iter": self.best_fid_iter,
            "last_fvd": self.last_fvd,
            "best_fvd": self._finite_or_none(self.best_fvd),
            "best_fvd_epoch": self.best_fvd_epoch,
            "best_fvd_iter": self.best_fvd_iter,
            "final_iter": final_iter,
            "final_epoch": final_epoch,
            "curves": {
                "train_loss": self.train_loss_curve,
                "test_loss": self.test_loss_curve,
                "fid": self.fid_curve,
                "fvd": self.fvd_curve,
            },
        }
