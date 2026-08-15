"""Shared building blocks for the MV-OPSD (multi-view privileged OPSD) data pipeline.

The modules here exist to keep one property true end to end: the images and the
prompt text a student sees during RL are produced by the same rules the
SpatialStack SFT dataloader used, so the only intended difference between
teacher and student is the number of views.
"""
