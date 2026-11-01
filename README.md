# BREM-WEAS

Core implementation of **BREM** (54-dimensional S+G features) and **WEA-S**
(weak evidence augmentation by shrinkage) for compressed speech steganalysis.

Paper: *Weak positive augmentation and local relative statistics for compressed
speech steganalysis under embedding rate mismatch*.

## Install

Python 3.12. The dependency versions below match the implementation environment.

```bash
git clone https://github.com/GAOFeipeng/BREM-WEAS.git
cd BREM-WEAS
python -m pip install -r requirements.txt
```

## Input and use

Supply decoded G.723.1 spectral quantization indices as integer NumPy arrays
of shape `(N, T, 3)`: recordings, frames, and the three indices in codec order.
Use the first `T` frames of each recording; the paper's default is `T = 333`.
All inputs to a fitted detector share the same frame count and QIM method.

Use disjoint recording identities for `P` (reference tables),
`F` (scaler/classifier), `C` (cover calibration), and evaluation.
Cover and source-rate stego in `F` have equal sample counts.
The following arrays are supplied by the caller:

```python
from brem import BREM
from weas import WEASDetector

features = BREM().fit(P_cover, P_stego)
detector = WEASDetector().fit(
    features.transform(F_cover),
    features.transform(F_stego),
    features.transform(C_cover),
)

X = features.transform(evaluation_segments)
scores = detector.decision_function(X)  # higher means stego
labels = detector.predict(X)            # 0: cover; 1: stego
```

Reuse the fitted objects across target embedding rates. Fit separately for each
source QIM method, language, source rate, and segment length.

## Defaults

BREM uses 18 relations (3 unary, 6 directed within-frame, 9 directed
adjacent-frame). Its coordinates are 36 class-frequency sums followed by
18 mean natural-log support differences; `epsilon = 1e-6`.

WEA-S uses `lambdas = (0.1, 0.25, 0.5)`, `beta = 0.1`, and a linear-kernel
C-SVC with `C = 1`. Standardization uses real `F` samples only.
Calibration uses `eta = 0.05` and the upper-tail rank
`p(s) = (1 + count(C_scores >= s)) / (len(C_scores) + 1)`.

## License

MIT. See [LICENSE](LICENSE).
