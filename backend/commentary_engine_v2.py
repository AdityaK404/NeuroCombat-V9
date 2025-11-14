"""
Commentary Generation Engine for NeuroCombat - V2 FIXED
=======================================================

NO DUPLICATES - Generates clean, distinct commentary lines

Features:
- One commentary line per distinct move
- Temporal awareness
- Natural language variety
- Player-specific commentary
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class CommentaryLine:
    """Single commentary line"""
    text: str
    timestamp: float
    player: Optional[int]
    confidence: float
    move_type: Optional[str] = None


class CommentaryEngine:
    # Enhanced commentary templates with more variety
    COMMENTARY_TEMPLATES = {
        "jab": {
            1: [
                "Player 1 snaps out a sharp jab!",
                "Quick jab from Player 1!",
                "Player 1 with a stiff jab to the face!",
                "Crisp jab lands for Player 1!",
                "Player 1 fires off a measuring jab!",
            ],
            2: [
                "Player 2 throws a probing jab!",
                "Quick jab from Player 2!",
                "Player 2 with a measured jab!",
                "Sharp jab from Player 2!",
                "Player 2 snaps the jab out!",
            ]
        },
        "cross": {
            1: [
                "Player 1 throws a powerful cross!",
                "Big cross from Player 1!",
                "Player 1 with the straight right!",
                "Powerful cross lands for Player 1!",
                "Player 1 unloads the cross!",
            ],
            2: [
                "Player 2 fires the cross!",
                "Heavy cross from Player 2!",
                "Player 2 with the power punch!",
                "Straight cross from Player 2!",
                "Player 2 lands a solid cross!",
            ]
        },
        "hook": {
            1: [
                "Player 1 whips in a hook!",
                "Devastating hook from Player 1!",
                "Player 1 connects with the hook!",
                "Wide hook from Player 1!",
                "Player 1 throws a looping hook!",
            ],
            2: [
                "Player 2 throws a vicious hook!",
                "Big hook from Player 2!",
                "Player 2 lands the hook!",
                "Sweeping hook from Player 2!",
                "Player 2 with a powerful hook!",
            ]
        },
        "uppercut": {
            1: [
                "Player 1 digs in an uppercut!",
                "Brutal uppercut from Player 1!",
                "Player 1 goes upstairs!",
                "Devastating uppercut by Player 1!",
                "Player 1 with a rising uppercut!",
            ],
            2: [
                "Player 2 with the uppercut!",
                "Big uppercut from Player 2!",
                "Player 2 lands clean uppercut!",
                "Rising uppercut from Player 2!",
                "Player 2 throws a nasty uppercut!",
            ]
        },
        "front_kick": {
            1: [
                "Player 1 fires the front kick!",
                "Push kick from Player 1!",
                "Player 1 with the teep!",
                "Front kick lands for Player 1!",
                "Player 1 measures the distance with a front kick!",
            ],
            2: [
                "Player 2 throws a front kick!",
                "Teep from Player 2!",
                "Player 2 with the push kick!",
                "Front kick from Player 2!",
                "Player 2 fires off a teep!",
            ]
        },
        "roundhouse_kick": {
            1: [
                "Player 1 unleashes a roundhouse!",
                "Big roundhouse from Player 1!",
                "Player 1 with the power kick!",
                "Massive roundhouse by Player 1!",
                "Player 1 throws a thunderous roundhouse!",
            ],
            2: [
                "Player 2 throws a roundhouse!",
                "Brutal kick from Player 2!",
                "Player 2 lands the roundhouse!",
                "Power kick from Player 2!",
                "Player 2 with a devastating roundhouse!",
            ]
        },
        "side_kick": {
            1: [
                "Player 1 hits a side kick!",
                "Side kick from Player 1!",
                "Player 1 with lateral attack!",
                "Sharp side kick by Player 1!",
            ],
            2: [
                "Player 2 throws side kick!",
                "Side kick lands for Player 2!",
                "Player 2 with the sidekick!",
                "Precision side kick from Player 2!",
            ]
        },
        "takedown": {
            1: [
                "Player 1 shoots for the takedown!",
                "Takedown attempt from Player 1!",
                "Player 1 goes for the legs!",
                "Player 1 changes levels!",
                "Player 1 driving for the takedown!",
            ],
            2: [
                "Player 2 shoots in!",
                "Takedown attempt by Player 2!",
                "Player 2 goes for the double leg!",
                "Player 2 changes levels!",
                "Player 2 looking for the takedown!",
            ]
        },
        "guard": {
            1: [
                "Player 1 in defensive stance",
                "Player 1 stays composed",
                "Player 1 holds guard",
                "Player 1 keeping tight defense",
            ],
            2: [
                "Player 2 in defensive mode",
                "Player 2 holds position",
                "Player 2 staying cautious",
                "Player 2 maintains guard",
            ]
        },
        "clinch": {
            1: [
                "Player 1 initiates the clinch!",
                "Player 1 ties up!",
                "Clinch work from Player 1!",
                "Player 1 controls the clinch!",
            ],
            2: [
                "Player 2 in the clinch!",
                "Player 2 ties up!",
                "Clinch battle from Player 2!",
                "Player 2 working the clinch!",
            ]
        }
    }

    # Transitional phrases for natural flow
    TRANSITIONS = [
        "And now,",
        "Here we go,",
        "Watch this,",
        "Look at this,",
        "Beautiful,",
        "Nice,",
        "There it is,",
        "Oh!",
    ]

    def __init__(self, fps: float = 30, enable_tts: bool = False):
        self.fps = fps
        self.enable_tts = enable_tts
        self.single_fighter_mode = False

    def generate_commentary(
        self,
        moves_json_path: str,
        output_path: str = None
    ) -> List[CommentaryLine]:
        """
        Generate commentary from moves JSON - ONE LINE PER MOVE
        """
        moves_json_path = Path(moves_json_path)
        
        if not moves_json_path.exists():
            raise FileNotFoundError(f"Moves JSON not found: {moves_json_path}")

        with open(moves_json_path, "r") as f:
            moves_data = json.load(f)

        metadata = moves_data.get("metadata", {})
        moves = moves_data.get("moves", [])

        print(f"Generating commentary for {len(moves)} moves...")

        # Check if single fighter mode
        player_ids = set(m.get("player_id", 1) for m in moves)
        self.single_fighter_mode = len(player_ids) == 1

        commentary_lines = []
        
        # Opening line
        commentary_lines.append(CommentaryLine(
            text=self._get_opening_line(),
            timestamp=0.0,
            player=None,
            confidence=1.0
        ))
        
        # Generate ONE line per move
        for move in moves:
            move_type = move.get("move_type", "neutral")
            
            # Skip neutral moves
            if move_type == "neutral":
                continue
            
            player_id = move.get("player_id", 1)
            confidence = move.get("confidence", 0.5)
            frame_start = move.get("frame_start", 0)
            frame_end = move.get("frame_end", frame_start + 1)
            duration_frames = frame_end - frame_start
            
            # Calculate timestamp (use start of move)
            timestamp = frame_start / self.fps
            
            # Generate commentary text
            text = self._generate_commentary_text(move_type, player_id, duration_frames)
            
            line = CommentaryLine(
                text=text,
                timestamp=timestamp,
                player=player_id,
                confidence=confidence,
                move_type=move_type
            )
            
            commentary_lines.append(line)
        
        # Add closing commentary
        if commentary_lines:
            last_timestamp = commentary_lines[-1].timestamp + 2.0
            commentary_lines.append(CommentaryLine(
                text=self._get_closing_line(),
                timestamp=last_timestamp,
                player=None,
                confidence=1.0
            ))

        print(f"✓ Generated {len(commentary_lines)} commentary lines")

        if output_path:
            self._save_outputs(commentary_lines, output_path, metadata)

        return commentary_lines

    def _generate_commentary_text(
        self, 
        move_type: str, 
        player_id: int,
        duration_frames: int
    ) -> str:
        """
        Generate natural commentary text for a move
        """
        # Get base template
        if move_type in self.COMMENTARY_TEMPLATES:
            templates = self.COMMENTARY_TEMPLATES[move_type].get(player_id, [])
            if templates:
                text = random.choice(templates)
            else:
                text = f"Player {player_id} executes a {move_type.replace('_', ' ')}!"
        else:
            text = f"Player {player_id} with a {move_type.replace('_', ' ')}!"
        
        # Add transition occasionally for variety
        if random.random() < 0.3:
            transition = random.choice(self.TRANSITIONS)
            text = f"{transition} {text.lower()}"
        
        # Add intensity modifier for longer moves
        if duration_frames > 10:
            intensifiers = ["That's a long", "Extended", "Committed"]
            if random.random() < 0.4:
                text = text.replace("a ", f"a {random.choice(intensifiers).lower()} ")
        
        return text

    def _get_opening_line(self) -> str:
        """Get opening commentary line"""
        openings = [
            "And here we go!",
            "The fight is underway!",
            "Let's get this fight started!",
            "Fighters touch gloves, and we're off!",
            "The action begins!",
            "And the bell rings!",
        ]
        return random.choice(openings)

    def _get_closing_line(self) -> str:
        """Get closing commentary line"""
        closings = [
            "What a sequence!",
            "Incredible action!",
            "Both fighters gave it their all!",
            "That's going to be hard to top!",
            "What a display of skill!",
            "The crowd is on their feet!",
        ]
        return random.choice(closings)

    def _save_outputs(
        self,
        lines: List[CommentaryLine],
        output_path: str,
        metadata: Dict
    ):
        """Save commentary to JSON and TXT files"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON output
        json_path = output_path.with_suffix(".json")
        output_data = {
            "metadata": metadata,
            "commentary": [asdict(line) for line in lines],
            "stats": {
                "total_lines": len(lines),
                "player_1_lines": sum(1 for l in lines if l.player == 1),
                "player_2_lines": sum(1 for l in lines if l.player == 2),
                "general_lines": sum(1 for l in lines if l.player is None),
            }
        }
        
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved commentary JSON to {json_path}")
        
        # TXT output
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("NEUROCOMBAT COMMENTARY TRANSCRIPT\n")
            f.write("=" * 60 + "\n\n")
            
            for line in lines:
                timestamp_str = f"{int(line.timestamp//60):02d}:{int(line.timestamp%60):02d}"
                player_str = f"[P{line.player}]" if line.player else "[GEN]"
                f.write(f"{timestamp_str} {player_str} {line.text}\n")
        
        print(f"✓ Saved commentary transcript to {txt_path}")
        
        if self.enable_tts:
            self._generate_tts(lines, output_path)

    def _generate_tts(self, lines: List[CommentaryLine], output_path: Path):
        """Generate TTS audio (placeholder)"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            audio_dir = output_path.parent / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            for i, line in enumerate(lines):
                audio_file = audio_dir / f"line_{i:04d}.mp3"
                engine.save_to_file(line.text, str(audio_file))
            
            engine.runAndWait()
            print(f"✓ Generated TTS audio in {audio_dir}")
        except ImportError:
            print("⚠️ pyttsx3 not installed, skipping TTS generation")
        except Exception as e:
            print(f"⚠️ TTS generation failed: {e}")