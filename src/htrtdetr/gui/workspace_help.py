"""Workflow help and shortcuts from the same registry as keyboard handlers."""

from . import shortcuts


def help_text():
    workflow = """Quick Start

This loop needs no trained model. A checkpoint makes it faster later; it is not
how you start.

1. Create/Open Project. Choose a project folder and the class names.
2. Add Files or Add Folder in Project Videos. Select a video to open it.
3. Pick frames to label (Ctrl+R) reads the videos in scope and chooses the
   frames worth your time. Ctrl+Right steps to the next chosen frame, and the
   circles on the Timeline are clickable.
4. Find animals on a chosen frame: C clicks one animal and a box is fitted
   to it; W draws one by hand. Choose the class in the Inspector. The
   Inspector's Find all at once (optional) boxes every one of them in a
   single press, when a helper for it is installed.
5. Select a box and press Continue to carry that animal forward, or
   Review > Continue every animal across this clip for a whole clip. Assign
   behaviors in the Inspector and edit intervals on the Timeline.
6. Review shows what came back. Accept, correct or reject. Nothing reaches
   annotations.json until you accept it.
7. Go back to 3. The next round avoids frames like the ones you just labelled.
   Mark finished videos Done in Project Videos.
8. Save writes all videos. Save and Export Frames also writes the images used
   for training.
9. Train, in the top bar, builds a detector from the labels you have so far -
   here, in this window. It saves, collects the frames, trains, and makes the
   result this project's model. There is nothing to fill in: how long you are
   willing to wait is the only question, and it is optional.
10. Analyze then runs that model over the videos and brings back the boxes it
   was least sure about, as candidates in the Review Inbox. Go back to 3.
   The workflow window is for the production run over finished labels, once.

Your workspace

Top bar, row 1: which video you are in, whether it is saved, and whether a
model is set. Row 2 is the loop, left to right: Pick frames to label, Review,
Train, Analyze - all four run in this window - then Undo, Redo, Save and
Settings. Train and Analyze dim, with the reason, until they can run.
Tool Rail: Tools (Select, Box, Rotate, Click animal, Pan, Delete), Assist
(Continue, Keep tool) and View (Fit, Panels, zoom readout).
Canvas: shows the current video; F fits the image, +/- zoom, middle-drag pans.
Inspector: class, track, behavior and the next action for your selection.
Review Inbox: check uncertain predictions and track problems. Review > Review Inbox (all videos) covers every video.
Timeline: annotations, tracks, behaviors, review markers and samples for this video.
Project Videos: filter, search, select videos for batch work, or locate a missing file.

AI assistance

Detector pre-labelling requires a checkpoint and a Python environment with PyTorch. It is the accelerator that arrives after you have trained once; the loop above does not wait for it.
Choose the checkpoint in Project > Settings > Model...; the top bar says whether there is one. Missing capabilities are disabled with a reason.
The segmentation helper is a different thing: it is what answers C and the Inspector's Find all, and it needs no checkpoint of yours. It starts itself on your first C click.
Machine-generated candidates remain outside annotations.json until accepted.
Tracking identities belong to each video; tracks are never connected across videos.
Cache hits reuse completed predictions; Cancel keeps completed videos and a failed video does not stop the batch.

Advanced

Settings > Assistance > Advanced assistance selects a click backend; Automatic chooses an available helper, predictions, or local fitting.
Settings > Tracking chooses the continuation method. Copy always works without a model.
Settings > Project controls what happens at the last frame, and the expected animals per frame.
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
