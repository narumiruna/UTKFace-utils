# UTKFace-utils

UTKFace
├── faces
│   ├── part1
│   ├── part2
│   └── part3
└── faces_aligned_cropped
    ├── crop_part1
    └── UTKFace

```python
from utkface import UTKFace
from torchvision import transforms

transform = transforms.Compose([
        ...,
        transforms.ToTensor()
    ])

dataset = UTKFace('path/to/UTKFace/faces', transform=transform)

or 

dataset = UTKFace('path/to/UTKFace/faces_aligned_cropped', transform=transform)
```