"""Pygame visual paradigm for lower-limb MI."""

from __future__ import annotations

import time
from typing import Literal

import pygame

Phase = Literal["prep", "cue", "rest"]


class ParadigmGUI:
    """Display cross, arrow, and circle phases."""

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        prep_sec: float = 2.0,
        cue_sec: float = 4.0,
        rest_sec: float = 4.0,
        fullscreen: bool = False,
    ) -> None:
        self.prep_sec = prep_sec
        self.cue_sec = cue_sec
        self.rest_sec = rest_sec
        pygame.init()
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("Lower Limb MI — Sun et al. 2026")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 48)

    def _draw_cross(self) -> None:
        self.screen.fill((30, 30, 30))
        w, h = self.screen.get_size()
        cx, cy = w // 2, h // 2
        pygame.draw.line(self.screen, (255, 255, 255), (cx - 40, cy), (cx + 40, cy), 4)
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy - 40), (cx, cy + 40), 4)
        pygame.display.flip()

    def _draw_arrow(self, direction: str) -> None:
        self.screen.fill((30, 30, 30))
        w, h = self.screen.get_size()
        cx, cy = w // 2, h // 2
        color = (0, 200, 255)
        if direction == "left":
            points = [(cx + 60, cy - 50), (cx - 20, cy), (cx + 60, cy + 50)]
        else:
            points = [(cx - 60, cy - 50), (cx + 20, cy), (cx - 60, cy + 50)]
        pygame.draw.polygon(self.screen, color, points)
        label = "Imagine LEFT thigh lift" if direction == "left" else "Imagine RIGHT thigh lift"
        text = self.font.render(label, True, (220, 220, 220))
        self.screen.blit(text, (w // 2 - text.get_width() // 2, h - 80))
        pygame.display.flip()

    def _draw_circle(self) -> None:
        self.screen.fill((30, 30, 30))
        w, h = self.screen.get_size()
        pygame.draw.circle(self.screen, (180, 180, 180), (w // 2, h // 2), 50, 3)
        pygame.display.flip()

    def _wait_phase(self, duration: float, draw_fn) -> float:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit("Paradigm closed by user")
            draw_fn()
            self.clock.tick(60)
        return t0

    def run_trial(self, cue: str) -> dict[str, float]:
        """Run one 10 s trial; cue is 'left' or 'right'."""
        times = {}
        times["trial_start"] = time.perf_counter()
        times["prep_start"] = self._wait_phase(self.prep_sec, self._draw_cross)
        times["cue_start"] = self._wait_phase(self.cue_sec, lambda: self._draw_arrow(cue))
        times["rest_start"] = self._wait_phase(self.rest_sec, self._draw_circle)
        times["trial_end"] = time.perf_counter()
        return times

    def close(self) -> None:
        pygame.quit()
