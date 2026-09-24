"""Workflow help and shortcuts from the same registry as keyboard handlers."""

from . import shortcuts


def help_text():
    workflow = """Quick Start

1. Create/Open Project. Choose a project folder, classes and behaviors.
2. Add Files or Add Folder in Project Videos. Select a video to open it.
3. Find animals: W draws a box; C lets you click an animal. Choose its class in the Inspector.
4. Select a box and Continue this animal. Choose Fast tracking, Accurate propagation, or Copy to next frame.
5. Assign behaviors in the Inspector and edit intervals on the current video's Timeline.
6. Pre-label videos with the model generates suggestions. Pick frames to label chooses frames and behavior clips across your chosen scope. Both are the same window with one checkbox the other way round.
7. Review Next moves across videos. Accept, correct or reject suggestions. Mark finished videos Done in Project Videos.
8. Save writes all videos. Save and Export Frames also writes the images used for training.

Your workspace

Tool Rail: Select, Draw, Draw rotated box, Click animal, Pan and Delete.
Canvas: shows the current video; F fits the image, +/- zoom, middle-drag pans.
Inspector: class, track, behavior and the next action for your selection.
Review Inbox: check uncertain predictions and track problems. Project Review Inbox covers every video.
Timeline: annotations, tracks, behaviors, review markers and samples for this video.
Project Videos: filter, search, select videos for batch work, or locate a missing file.

AI assistance

Automatic detection requires a checkpoint and a Python environment with PyTorch.
Choose the checkpoint in Project > Model...; the top bar says whether there is one. Missing capabilities are disabled with a reason.
Machine-generated suggestions remain outside annotations.json until accepted.
Tracking identities belong to each video; tracks are never connected across videos.
Cache hits reuse completed predictions; Cancel keeps completed videos and a failed video does not stop the batch.

Advanced

Settings > Assistance > Advanced assistance selects a click backend; Automatic chooses an available helper, predictions, or local fitting.
Settings > Tracking chooses the continuation method. Copy always works without a model.
Settings > Project controls what happens at the last frame.
Project files restore review decisions, behavior intervals, samples, and each video's position. Open annotations.json to import a legacy dataset.
Undo/Redo belongs to the current video and may cover multiple frames.
Project progress uses explicit status. Visited frames indicate navigation, not completed annotation.

Keyboard shortcuts
"""
    lines = []
    for binding in shortcuts.for_scope("annotator"):
        label = shortcuts.label_for(binding.action)
        description = shortcuts.help_for(binding.action)
        line = f"{label}: {description}"
        if line not in lines:
            lines.append(line)
    return workflow + "\n".join(lines)
