import pydiverse.transform as pdt


def num_rows(tbl: pdt.Table) -> int:
    return tbl >> pdt.summarize(num_rows=pdt.count()) >> pdt.export(pdt.Scalar)
