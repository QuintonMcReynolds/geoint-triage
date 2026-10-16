"""Export the robust model to ONNX (logits + embedding outputs) and check parity with PyTorch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoint import data as D  # noqa: E402
from geoint.infer import load_net  # noqa: E402
from geoint.model import ExportWrapper  # noqa: E402


def main(variant: str = "robust"):
    net = load_net(variant, dev=torch.device("cpu"))
    out = D.ART / "chipnet.onnx"
    dummy = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(ExportWrapper(net).eval(), dummy, out, input_names=["chips"], output_names=["logits", "embedding"],
                      dynamic_axes={"chips": {0: "n"}, "logits": {0: "n"}, "embedding": {0: "n"}}, opset_version=17,
                      dynamo=False)
    x = torch.from_numpy(D.normalize(D.load()["x_test"][:256]))
    with torch.no_grad():
        ref_l, ref_z = ExportWrapper(net)(x)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    l, z = sess.run(None, {"chips": x.numpy()})
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); max |logit diff| {np.abs(l - ref_l.numpy()).max():.2e}, "
          f"argmax agreement {(l.argmax(1) == ref_l.numpy().argmax(1)).mean():.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
