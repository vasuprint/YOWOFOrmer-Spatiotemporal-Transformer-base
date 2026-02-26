# AVA Official Evaluation Code

This directory contains the official AVA evaluation code from:

1. **Google Research AVA Dataset**
   - Source: https://research.google.com/ava/
   - License: Apache License 2.0
   - Original Authors: Google Research Team

2. **YOWOv3 Implementation**
   - Adapted from: https://github.com/yjh0410/YOWOv3
   - Integration with PyTorch detection pipelines

## Files

- `object_detection_evaluation.py` - Core evaluation logic from TensorFlow Object Detection API
- `per_image_evaluation.py` - Per-image evaluation utilities
- `standard_fields.py` - Field definitions for AVA dataset
- `label_map_util.py` - Label map parsing utilities
- `metrics.py` - Metric computation functions

## License

These files are licensed under the Apache License, Version 2.0.
See the original repositories for full license details.

## Citation

If you use this evaluation code, please cite:

```bibtex
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={CVPR},
  year={2018}
}
```