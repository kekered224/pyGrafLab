# Third-party notices

The MIT licence in `LICENSE` covers the Python source, the front-end source and
the documentation in this repository. The items below are **not** covered by it.

## Application icons

`icons/app-icon.*` and `pygraflab/web/static/favicon.*` are original artwork
produced with an AI image-generation tool for this project.

## Front-end

The web UI is plain JavaScript with no bundled third-party libraries, web
fonts, CDNs or tile servers; it runs fully offline.

## Reference implementation

GrafLab 2.1.4 is a separate work by **Blažej Bucha and Juraj Janák**:

> Bucha, B., Janák, J. (2013). *A MATLAB-based graphical user interface
> program for computing functionals of the geopotential up to ultra-high
> degrees and orders.* Computers & Geosciences, 56, 186–196.
> https://doi.org/10.1016/j.cageo.2013.03.012

**Copyright and all other rights in the original GrafLab software and its
documentation remain with those authors and their rights holders. This project
claims no ownership of that work.** The code and readme archive published by
the authors (PANGAEA, https://doi.org/10.1594/PANGAEA.808577) is distributed
under CC-BY-3.0; this repository nonetheless contains no MATLAB source from
it. pyGrafLab is an independent Python port. Cite Bucha & Janák (2013) when
publishing results obtained with this implementation.

## Bundled model subset

`GGM/EIGEN-6C4_n180.gfc` is a degree-180 truncation of **EIGEN-6C4**, published
by GFZ Potsdam and GRGS Toulouse and distributed through ICGEM. The coefficients
are the work of the model authors, not of this project; they are included, with
the original header, only so that a fresh checkout can run the example and the
tests. Metadata and the citation are in the file header and in README.md:

> Förste, C., Bruinsma, S. L., Abrikosov, O., Lemoine, J.-M., Marty, J. C.,
> Flechtner, F., Balmino, G., Barthelmes, F., Biancale, R. (2014).
> *EIGEN-6C4 – The latest combined global gravity field model including GOCE
> data up to degree and order 2190 of GFZ Potsdam and GRGS Toulouse.*
> GFZ Data Services. https://doi.org/10.5880/icgem.2015.1

## Python dependencies

`numpy`, `scipy`, and optionally `flask` and `h5py` are installed separately
and remain under their own licences; consult their distributions.
