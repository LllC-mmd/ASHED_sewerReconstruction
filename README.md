# Sewer Reconstructor using Incomplete Information

Sewer reconstructor is a Python-based tool to help generate urban drainage models (in [SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm)-based format) from existing layouts and digital elevation model (DEM).

## Main functions

The main functions for sewer reconstruction include:

- `shp2swmm` in `main.py`: generates the SWMM input file from the sewer shapefiles and DEM.

- `split_drainage_line_with_inlet` in `sewer_reconstructor.py`: splits the existing sewer segments with a given spacing.

- `adjust_swmm_LinkDirection` in `sewer_reconstructor.py`: estimates the gravitational flow direction of the sewer network considering necessray topological constraints.

- `adjust_swmm_NodeDepth` in `sewer_reconstructor.py`: estimates the nodal depth of the sewer network constrained by the minimum and maximum allowable slopes.

More details can be found in [this preprint paper](https://doi.org/10.5194/egusphere-2024-3780).

## Example usage

To minimize the environment difference, we offer a [docker](https://docs.docker.com/get-started/introduction/)-based implementation which can be downloaded via this [link](https://drive.google.com/file/d/1UHm_-1LRS61cvZ02fzk6wGUWQA91gXpf/view?usp=sharing).

We can load the prepared docker image by

```shell {cmd}
docker load -i thu-sewer-reconstructor.tar
```

Then we can create a local container for sewer reconstruction application

```shell {cmd}
docker run -v /home/swmm_case/example:/data --name my-sewer-reconstructor -itd thu-sewer-reconstructor:v250526 bash
```

where:

- `/home/swmm_case/example` is a host folder [mounted](https://docs.docker.com/engine/storage/bind-mounts/) into the container folder `/data`. We can specify this host folder according to our needs so we can exchange necessary input data between the host and container.
- `my-sewer-reconstructor` is the name of the created container.

After creating the container, we can perform sewer reconstruction as follows:

```shell {cmd}
docker exec my-sewer-reconstructor \
/root/anaconda3/envs/lyy_sewer/bin/python3 /home/sewer-reconstructor/main.py \
                --slope_config_path /data/case1/slope_config.json \
                --swmm_config_path /data/case1/swmm_config.json \
                --solver_config_path /data/case1/LP_config.json
```

where:

- `slope_config_path` is a JSON file which defines the minimum and maximum slope for different pipe sizes (in the unit of `mm`). 
An example can be given as follows:

```json {cmd}
{
    "slope_max": {
        "300": 0.41,
        "400": 0.28,
        "500": 0.21,
        "600": 0.16,
        "800": 0.11,
        "1000": 0.08,
        "1200": 0.06,
        "1400": 0.05,
        "1500": 0.05
    },
    "slope_min": {
        "300": 0.0020,
        "400": 0.0015,
        "500": 0.0012,
        "600": 0.0010,
        "800": 0.0008,
        "1000": 0.0006,
        "1200": 0.0006,
        "1400": 0.0005,
        "1500": 0.0005
    }
}
```

- `swmm_config_path` is a JSON file which defines the geometry and simulation settings of SWMM model.
An example can be given as follows:

```json {cmd}
{
    "parameter": {
        "inlet_spacing": 50,
        "DEM_path": "/data/case2/_sw_dem.tif"
    },
    "geometry_settings": {
        "sewer": "/data/case2/_sw_line.gpkg",
        "outfall": "/data/case2/_sw_outfall.gpkg",
        "fixed_direction": "/data/case2/_sw_fixedDir.gpkg",
        "masked_area": "/data/case2/_sw_inletMask.gpkg"
    },
    "simulation_settings": {
        "input_file": "/data/case2/_test.inp",
        "start_date": "07/11/2022",
        "start_time": "00:00:00",
        "end_date": "07/11/2022",
        "end_time": "16:00:01",
        "report_step": "0:05:00",
        "EPSG_code": 32648
    }
}
```

where `fixed_direction` and `masked_area` are optional entries for fixed flow directions (by the order of drawing lines) and regions without rainwater inlets.

- `solver_config_path` is a JSON file which defines the solution method of mathematical programming (MP) used for sewer reconstruction.

```json {cmd}
{
    "solver": "cbc",
    "license": "",
    "name": "test",
    "outfall_relaxed": "True",
    "save_prefix": "/data/case2"
}
```

where:
  - `solver` is the name of the MP solver (we use [cbc](https://github.com/coin-or/Cbc) by default).
  - `license` is the path to the license file (if commercial solvers like `gurobi` is select as `solver`).
  - `name` is the name of the MP case (used for exporting `.lp` file for feasibilty analysis).
  - `outfall_relaxed` controls whether outfall elevations will be fixed as given in their `INVERTELEV` columns (which can be manually specified by local surveying).
  - `save_prefix` is the path of the folder where the `.lp` file will be exported.

## Citation

Please cite the paper if you use aforementioned functions/algorithms in your research.

```bibtex
@article{li2025enhancing,
  title={Enhancing Urban Pluvial Flood Modelling through Graph Reconstruction of Incomplete Sewer Networks},
  author={Li, Ruidong and Liu, Jiapei and Sun, Ting and Jian, Shao and Tian, Fuqiang and Ni, Guangheng},
  journal={EGUsphere},
  volume={2025},
  pages={1--32},
  year={2025},
  publisher={Copernicus Publications G{\"o}ttingen, Germany},
  DOI = {https://doi.org/10.5194/egusphere-2024-3780}
}
```