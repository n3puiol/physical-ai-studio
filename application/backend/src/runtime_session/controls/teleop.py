from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from robots.features import feature_names
from runtime_session.controls.base import Control
from runtime_session.store import ACTION

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.store import Store
    from schemas.runtime_session import TeleopControlConfig


class LeaderLostError(ConnectionError):
    """The leader failed every read for longer than the control tolerates."""


class TeleopControl(Control):
    """Command the follower whatever the leader measures, due by the leader's next reading.

    The leader can be anywhere when teleop starts, so for the first ``grace_s``
    each command is stamped due at the end of the grace period: the robot worker
    closes the gap over it rather than jumping. It tracks the leader the whole
    time, so the follower is on it when the grace period ends.

    A leader read that fails skips the tick: nothing new is commanded, so the
    follower holds the last command. A leader behind a connection (a simulator
    over HTTP, say) drops the odd read, especially while it resets. Only when
    every read has failed for ``outage_s`` does the control give up and end,
    which the session reports.
    """

    outage_s: float = 2.0
    """How long every leader read may fail before the control ends."""

    def __init__(self, store: Store, leader: Robot, config: TeleopControlConfig, bus: Bus | None = None) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names)))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The leader's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        super().__init__(store, config, name="TeleopControl", hz=config.hz, bus=bus)
        self.leader = leader
        self.grace_s = config.grace_s
        self.started_at = 0.0
        self.failing_since: float | None = None
        """When the current run of failed leader reads began; ``None`` while the leader answers."""
        self.failed_reads = 0
        """How many leader reads in a row have failed."""

    def on_start(self) -> None:
        self.started_at = time.monotonic()

    def on_tick(self) -> None:
        try:
            observation = self.leader.get_observation()
        except Exception as error:
            self._missed(error)
            return
        if self.failing_since is not None:
            logger.info(
                "Leader answers again after {} failed reads over {:.2f}s",
                self.failed_reads,
                time.monotonic() - self.failing_since,
            )
            self.failing_since, self.failed_reads = None, 0
        due = max(observation.timestamp + 1 / self.hz, self.started_at + self.grace_s)
        self.store.write(ACTION, observation.joint_positions, timestamp=due)

    def _missed(self, error: Exception) -> None:
        """Skip this tick, the follower holding the last command, unless the leader has been failing too long.

        Raises:
            LeaderLostError: Every read has failed for ``outage_s``.
        """
        now = time.monotonic()
        self.failed_reads += 1
        if self.failing_since is None:
            self.failing_since = now
            logger.warning("Leader read failed, holding the follower until it answers: {}", error)
        if now - self.failing_since >= self.outage_s:
            raise LeaderLostError(
                f"The leader has not answered for {now - self.failing_since:.1f}s "
                f"({self.failed_reads} failed reads): {error}"
            ) from error
