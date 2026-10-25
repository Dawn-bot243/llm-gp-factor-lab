"""Smoke test for Qlib Alpha158 on the downloaded cn_data_v3 dataset."""
import os
import sys

from qlib.config import C

# Windows-safe: avoid joblib spawn/multiprocessing issues with the project path.
C.joblib_backend = "threading"
C.kernels = 4

import qlib
from qlib.contrib.data.handler import Alpha158


def main():
    print("exe", sys.executable)
    print("prefix", sys.prefix)
    print("cwd", os.getcwd())
    print("file", __file__)

    qlib.init(provider_uri="D:/qlib_data/cn_data_v3", region="cn")
    handler = Alpha158(
        instruments="csi300",
        start_time="2020-01-01",
        end_time="2020-03-31",
        fit_start_time="2020-01-01",
        fit_end_time="2020-03-31",
    )
    df = handler.fetch(selector=slice("2020-01-02", "2020-01-10"), col_set="feature")
    print("shape", df.shape)
    print("columns", list(df.columns)[:20])
    print("nan_ratio", float(df.isna().mean().mean()))
    print(df.iloc[:3, :8])


if __name__ == "__main__":
    main()
