#!/usr/bin/env python3
"""
Render one Risk game, dump PNG frames, optionally build an animated GIF.

Usage examples
--------------
python render_trace.py --frames frames/
python render_trace.py --frames frames/ --gif gifs/game42.gif
"""

from __future__ import annotations
import imageio.v2 as iio
import glob
import argparse
import itertools
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from pyrisk.game import Game
from AI.ppoagent import PPOAgent

LOG = logging.getLogger("trace")
COUNTER = itertools.count(0)

def board_str_safe(game: Game) -> str:
    """Return an ASCII map even if Display lacks .board_str()."""
    if hasattr(game.display, "board_str"):
        return game.display.board_str()  # type: ignore[attr-defined]
    lines = []
    for terr in game.world.territories.values():
        owner = terr.owner.name[0] if terr.owner else "•"
        lines.append(f"{terr.name:15s} {owner} {terr.forces}")
    return "\n".join(lines)

def save_board_png(game: Game, n: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("off")
    ax.text(
        0,
        1,
        board_str_safe(game),
        va="top",
        family="monospace",
        fontsize=9,
    )
    fname = out_dir / f"g42_{n:03d}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    LOG.debug("Saved frame %s", fname)

def main(frame_dir: str, gif_path: Optional[str], max_turns: int, deal: bool) -> None:
    g = Game(curses=False, color=False, delay=0, wait=False, deal=deal)
    if not hasattr(Game, "_gid_counter"):
        Game._gid_counter = itertools.count(1)  # type: ignore[attr-defined]
    g.this_game_id = next(Game._gid_counter)

    orig_add_player = g.add_player

    def patched_add_player(name, ai_cls, **kw):
        orig_add_player(name, ai_cls, **kw)
        g.players[name].game_id = g.this_game_id

    g.add_player = patched_add_player  # type: ignore[method-assign]

    g.add_player("PPO", lambda p, *a: PPOAgent(p, g, g.world))
    g.add_player("RandomBot", lambda p, *a: g.players["PPO"].ai.strategies["random"].__class__(p, g, g.world))

    frame_path = Path(frame_dir)
    frame_path.mkdir(parents=True, exist_ok=True)

    Path("traces").mkdir(parents=True, exist_ok=True)

    orig_event = g.event
    def patched_event(msg, territory=None, player=None):
        save_board_png(g, next(COUNTER), frame_path)
        orig_event(msg, territory=territory, player=player)

    g.event = patched_event  # type: ignore[method-assign]

    winner = g.play()
    LOG.info("Winner: %s", winner)

    with open(Path("traces") / f"game_{g.this_game_id}.json", "w") as f:
        json.dump({"winner": winner}, f, indent=2)

    if gif_path:
        gif_dir = Path(gif_path).parent
        gif_dir.mkdir(parents=True, exist_ok=True)

        pattern = str(frame_path / "g42_*.png")
        frames = sorted(glob.glob(pattern))
        imgs = [iio.imread(f) for f in frames]
        iio.mimsave(gif_path, imgs, fps=20)
        LOG.info("GIF saved via imageio → %s", gif_path)

        # Optional fallback to ImageMagick (for CLI users)
        png_glob = str(frame_path / "g42_*.png")
        magick_ok = subprocess.run("magick -version", shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        cmd = (
            ["magick", "convert", "-delay", "50", "-loop", "0", png_glob, gif_path]
            if magick_ok else
            ["convert", "-delay", "50", "-loop", "0", png_glob, gif_path]
        )

        LOG.info("Running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, shell=False, check=True)
            LOG.info("GIF saved → %s", gif_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            LOG.error("ImageMagick not found or failed; GIF was not created.\n"
                      "• Install ImageMagick and ensure ‘magick’ or ‘convert’ is on PATH.\n"
                      "• Or run without --gif and use imageio-generated version.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Folder to write PNG frames")
    parser.add_argument("--gif", help="Output GIF path (optional)")
    parser.add_argument("--turns", type=int, default=250, help="Max turns (cap)")
    parser.add_argument("--deal", action="store_true", help="Deal-territories mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main(args.frames, args.gif, args.turns, args.deal)
