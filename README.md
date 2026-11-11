# splitenum

splitenum enumerates every compatible collection of Harnack splits of the
degree-d simplex and records the real scheme of the T-curve each collection
produces. By Haas' theorem a T-curve is maximal exactly when its signs arise
from Harnack's by such a collection, so every node of the search is an M-curve
and the output is the exact distribution of nodes over the M-schemes realized
this way. All isotopy computations are done by
[libisotopy](https://github.com/polymake/libisotopy).

## Paper

Zoe Geiselmann, Michael Joswig, Lars Kastner and Marcel Wack.
*Complete enumeration of maximal T-curves in degrees eight and nine.*
FPSAC 2027. arXiv:XXXX.XXXXX <!-- TODO: preprint link -->

Related work:

- *121 Patchworked Curves of Degree Seven*,
  [arXiv:2602.06888](https://arxiv.org/abs/2602.06888), data at
  [dmg-lab/CombinatorialPatchworking](https://github.com/dmg-lab/CombinatorialPatchworking).
- *Fast Isotopy Computation for T-Curves*, ICMS 2026,
  [arXiv:2604.09221](https://arxiv.org/abs/2604.09221), the algorithm behind
  libisotopy.

## Results

| Degree | Splits | Nodes | M-schemes | Data |
|---|---:|---:|---:|---|
| 5 | 78 | 7,915 | 1 | |
| 6 | 222 | 1,035,083 | 3 | |
| 7 | 444 | 296,740,751 | 14 | |
| 8 | 873 | 105,949,232,275 | 38 | [statistics](results/deg8), [witnesses](https://github.com/dmg-lab/CombinatorialPatchworking/tree/main/deg8) |
| 9 | 1,437 | 57,065,860,838,381 | 225 | [statistics](results/deg9), [witnesses](https://github.com/dmg-lab/CombinatorialPatchworking/tree/main/deg9) |

## Building

Requires a C++20 compiler with OpenMP. MPI and CUDA are needed only for their
targets.

```sh
git clone --recurse-submodules <this repository>
cd splitenum
make            # libisotopy, then ./splitenum
make test       # test suite
make tools      # bin/witnesses, bin/pcomcheck and the other tools
make mpi        # ./splitenum-mpi, needs mpicxx
make gpu        # ./splitenum-gpu, needs nvcc; CUDA_ARCHS="75 86" selects targets
```

`ARCH` defaults to `-march=native`; build with `ARCH=` for binaries that run on
other machines.

`MAX_DELTA`, default 8, sizes fixed arrays inside libisotopy and is part of its
ABI. The Makefile passes it to both builds; always change it through `make
MAX_DELTA=9`, which degree 9 needs, and never for one side only.

## Usage

```sh
./splitenum --delta 6 --out out/d6
./splitenum --delta 7 --count-only
./splitenum --delta 8 --scheduler jobs --classify gpu --out out/d8
./splitenum -h
```

A run can be stopped at any time; rerunning the same command resumes it.
`--count-only` counts the tree without any isotopy computation. With
`--scheduler jobs` a master hands out budgeted jobs to the worker threads,
`--classify gpu` sends the classification through the CUDA kernel of
libisotopy, and `--listen PORT` together with `--connect DIR` spreads one run
over several machines.

## Output

Everything is written to the `--out` directory:

- `splits.csv`: the split table; witnesses refer to its split ids.
- `types.csv`: one row per real scheme and slack profile, with the node count
  and the shallowest, then lexicographically smallest, witness collection.
- `run.json`: parameters and statistics of the run.

`bin/witnesses RUN_DIR OUT_PREFIX` re-verifies every witness of a finished run
and writes it as a `.pcom` patchwork file.

## License

GPL-3.0, see [LICENSE](LICENSE).
