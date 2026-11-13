"""awe_synth: physics-based synthetic data generator for alkaline water electrolysis (AWE) test benches.

The generator writes (i) measurement files with the same schema a real bench would deliver and (ii) the hidden
ground truth in a SEPARATE folder. The platform reads only the measurement files; the truth is read only by
validation code.
"""
GENERATOR_VERSION = "1.0.0"
