import os
import json
import numpy as np
import h5py
import pandas as pd
import geopandas as gpd
import networkx as nx
import fiona

from osgeo import gdal
from shapely.geometry import Point, LineString

from sewer_reconstructor import setLinkDirection, reset_swmm_NodeDepth
    
# ********* miscellaneous functions (START) *********
def getRasterVal_bilinear(raster_dta, x, y, x0, y0, dx, dy, vectorized=False):
    height, width = raster_dta.shape

    if vectorized:
        col_nearest = np.rint((x - x0) / dx).astype(int)
        row_nearest = np.rint((y - y0) / dy).astype(int)

        out_bounds_flag = (col_nearest < 0) | (col_nearest >= width) | (row_nearest < 0) | (row_nearest >= height)

        x_nearest = x0 + (col_nearest + 0.5) * dx
        y_nearest = y0 + (row_nearest + 0.5) * dy

        col_left = np.clip(col_nearest - (x < x_nearest), 0, width-1)
        col_right = np.clip(col_nearest + (x >= x_nearest), 0, width-1)
        row_upper = np.clip(row_nearest - (y >= y_nearest), 0, height-1)
        row_lower = np.clip(row_nearest + (y < y_nearest), 0, height-1)

        q11 = raster_dta[row_upper, col_left]
        q12 = raster_dta[row_lower, col_left]
        q21 = raster_dta[row_upper, col_right]
        q22 = raster_dta[row_lower, col_right]

        x_frac = (x - x0 - (col_left + 0.5) * dx) / dx
        y_frac = (y - y0 - (row_upper + 0.5) * dy) / dy

        value = (1 - x_frac) * (1 - y_frac) * q11 + \
                (1 - x_frac) * y_frac * q12 + \
                x_frac * (1 - y_frac) * q21 + \
                x_frac * y_frac * q22
        
        # ------if indices are at the boundary grids, use the center value of the grid they fall in
        value[out_bounds_flag] = raster_dta[np.clip(row_nearest[out_bounds_flag], 0, height-1), 
                                            np.clip(col_nearest[out_bounds_flag], 0, width-1)]
    else:
        # Calculate the column and row indices of the nearest grid point
        col_nearest = int(round((x - x0) / dx))
        row_nearest = int(round((y - y0) / dy))

        # Check if the indices are within the raster bounds or at the boundary grids
        # ------if indices are at the boundary grids, use the center value of the grid they fall in
        if col_nearest < 0 or col_nearest >= width or row_nearest < 0 or row_nearest >= height:
            col_c = max(min(int(np.floor((x - x0) / dx)), width - 1), 0)
            row_c = max(min(int(np.floor((y - y0) / dy)), height - 1), 0)
            value = raster_dta[row_c, col_c]
        # ------else use the bilinear interpolation
        else:
            # Get the coordinates of the nearest grid point
            x_nearest = x0 + (col_nearest + 0.5) * dx
            y_nearest = y0 + (row_nearest + 0.5) * dy

            # Get the indices of the four surrounding grid points
            # ---if the center point of the nearest grid's x, i.e., `x_nearest`, is larger than `x`,
            # ------we select the left side of the nearest grid as the left side of the interpolation area
            col_left = max(col_nearest - 1, 0) if x < x_nearest else col_nearest
            col_right = min(col_nearest + 1, width - 1) if x >= x_nearest else col_nearest
            row_upper = max(row_nearest - 1, 0) if y >= y_nearest else row_nearest
            row_lower = min(row_nearest + 1, height - 1) if y < y_nearest else row_nearest

            # Get the values of the four surrounding grid points
            q11 = raster_dta[row_upper, col_left]
            q12 = raster_dta[row_lower, col_left]
            q21 = raster_dta[row_upper, col_right]
            q22 = raster_dta[row_lower, col_right]

            # Perform bilinear interpolation
            x_frac = (x - x0 - (col_left + 0.5) * dx) / dx
            y_frac = (y - y0 - (row_upper + 0.5) * dy) / dy

            value = (1 - x_frac) * (1 - y_frac) * q11 + \
                    (1 - x_frac) * y_frac * q12 + \
                    x_frac * (1 - y_frac) * q21 + \
                    x_frac * y_frac * q22

    return value


def LineToCoords(row: gpd.GeoSeries):
    if isinstance(row["geometry"], LineString):
        coords = [c for c in list(row["geometry"].coords)]
    else:
        coords = [c for p in row["geometry"] for c in list(p.coords)]
    return coords


def getLineCoords(input_gdf: gpd.GeoDataFrame):
    line_coords = input_gdf.apply(lambda row: LineToCoords(row), axis=1)
    line_coords = np.array(sum([i for i in line_coords.values], []))
    return np.unique(line_coords, axis=0)


def LineToSeg(row: gpd.GeoSeries, col_namelist: list):
    seg = {}
    if isinstance(row["geometry"], LineString):
        seg["geometry"] = [LineString([c1, c2]) for c1, c2 in zip(list(row["geometry"].coords), list(row["geometry"].coords[1:]))]
    else:
        seg["geometry"] = [LineString([c1, c2]) for p in row["geometry"].geoms for c1, c2 in zip(list(p.coords), list(p.coords[1:]))]
    for col in col_namelist:
        seg[col] = row[col]
    return seg
        

def explodeLineSeg(ini_gdf: gpd.GeoDataFrame, simplified=True, tolerance=1.0, save_path=None) -> gpd.GeoDataFrame:
    if simplified:
        ini_gdf["geometry"] = ini_gdf["geometry"].simplify(tolerance=tolerance, preserve_topology=True)
    property_cols = [col for col in ini_gdf.columns if col != "geometry"]
    # ------split GeoDataFrame with LineString and MultiLineString into segments
    line_segs = ini_gdf.apply(lambda row: LineToSeg(row, property_cols), axis=1, result_type="expand")
    line_segs = line_segs.explode(column="geometry", ignore_index=True)
    line_segs = gpd.GeoDataFrame(line_segs, crs=ini_gdf.crs)
    # ------drop the segments with empty geometry
    line_segs = line_segs[np.array(line_segs.apply(lambda x: x["geometry"] is not None, axis=1))]

    if save_path is not None:
        line_segs.to_file(save_path)
    
    return line_segs


def getIntersect2D(arr_A, arr_B):
    nrows, ncols = arr_A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [arr_A.dtype]}

    arr_C = np.intersect1d(arr_A.view(dtype), arr_B.view(dtype))
    arr_C = arr_C.view(arr_A.dtype).reshape(-1, ncols)
    return arr_C


def snapPointToLine(pt: Point, line: LineString, min_length_ratio=0.2, min_shift_angle=np.pi/18.0, min_distance=0.1):
    pt_snapped = line.interpolate(line.project(pt))

    # ------identify if the point should be attached with the from or to point of the segment
    pt_seg_from, pt_seg_to = line.boundary.geoms
    dis_snapped = pt.distance(pt_snapped)
    dis_proj_from = pt_snapped.distance(pt_seg_from)
    dis_proj_to = pt_snapped.distance(pt_seg_to)
    dis_seg = dis_proj_from + dis_proj_to

    if dis_snapped < min_distance:
        # ------if segment is too small
        if dis_seg <= min_distance:
            if dis_proj_from < dis_proj_to:
                return {"geometry": pt_seg_from, "snapped_distance": pt.distance(pt_seg_from), "projected_distance": 0.0}
            else:
                return {"geometry": pt_seg_to, "snapped_distance": pt.distance(pt_seg_to), "projected_distance": 0.0}
        else:
            # ------compare the absolute distance first in case of dis_snapped=0
            if dis_proj_from <= min_distance:
                return {"geometry": pt_seg_from, "snapped_distance": pt.distance(pt_seg_from), "projected_distance": 0.0}
            elif dis_proj_to <= min_distance:
                return {"geometry": pt_seg_to, "snapped_distance": pt.distance(pt_seg_to), "projected_distance": 0.0}
            # ------compare the relative distance second
            elif dis_proj_from / dis_seg <= min_length_ratio:
                return {"geometry": pt_seg_from, "snapped_distance": pt.distance(pt_seg_from), "projected_distance": 0.0}
            elif dis_proj_to / dis_seg <= min_length_ratio:
                return {"geometry": pt_seg_to, "snapped_distance": pt.distance(pt_seg_to), "projected_distance": 0.0}
            else:
                return {"geometry": pt_snapped, "snapped_distance": dis_snapped, "projected_distance": min(dis_proj_from, dis_proj_to)}
    else:
        if dis_seg <= min_distance:
            if dis_proj_from < dis_proj_to:
                return {"geometry": pt_seg_from, "snapped_distance": pt.distance(pt_seg_from), "projected_distance": 0.0}
            else:
                return {"geometry": pt_seg_to, "snapped_distance": pt.distance(pt_seg_to), "projected_distance": 0.0}
        else:
            if dis_proj_from / dis_seg <= min_length_ratio or np.arctan(dis_proj_from/dis_snapped) <= min_shift_angle or dis_proj_from <= min_distance:
                return {"geometry": pt_seg_from, "snapped_distance": pt.distance(pt_seg_from), "projected_distance": 0.0}
            elif dis_proj_to / dis_seg <= min_length_ratio or np.arctan(dis_proj_to/dis_snapped) <= min_shift_angle or dis_proj_to <= min_distance:
                return {"geometry": pt_seg_to, "snapped_distance": pt.distance(pt_seg_to), "projected_distance": 0.0}
            else:
                return {"geometry": pt_snapped, "snapped_distance": dis_snapped, "projected_distance": min(dis_proj_from, dis_proj_to)}


def getGraphFromGeoDataFrame(line_gdf: gpd.GeoDataFrame, simplified=False, tolerance=0.5, col_namelist=None, sym="@", prefix="RV"):
    line_graph = nx.Graph()

    node_geom = {}
    node_coords_ref = {}
    rv_nid = 0

    seg_info = {"geometry": [], "Pt1": [], "Pt2": [], "tauX": [], "tauY": []}
    index_list = []

    if col_namelist is not None:
        col_namelist_recorded = list(set(col_namelist) - {"tauX", "tauY"} - {"Pt1", "Pt2"})
        directed_flag = ("tauX" in col_namelist) and ("tauY" in col_namelist)
        named_flag = ("Pt1" in col_namelist) and ("Pt2" in col_namelist)
        edge_attributes = {}
        for col in col_namelist:
            if col not in seg_info:
                seg_info[col] = []
    else:
        directed_flag = False
        named_flag = False

    if simplified:
        line_gdf["geometry"] = line_gdf["geometry"].simplify(tolerance=tolerance, preserve_topology=True)

    for ids, row in line_gdf.iterrows():
        line = row["geometry"]
        for pt_c1, pt_c2 in zip(list(line.coords), list(line.coords[1:])):
            reversed_flag = False
            x_s, y_s = pt_c1
            x_e, y_e = pt_c2
            tau = np.array([x_e-x_s, y_e-y_s])
            tau = tau / np.sqrt(np.sum(tau**2))

            # ---determine the direction of the segement
            if directed_flag:
                if row["tauX"] * tau[0] + row["tauY"] * tau[1] < 0:
                    reversed_flag = True

            # ---determine the name of ending points of the segement
            if named_flag:
                if (x_s, y_s) not in node_coords_ref:
                    if reversed_flag:
                        sname = row["Pt2"]
                    else:
                        sname = row["Pt1"]
                    node_coords_ref[(x_s, y_s)] = sname
                    node_geom[sname] = Point(x_s, y_s)
                else:
                    sname = node_coords_ref[(x_s, y_s)]
                
                if (x_e, y_e) not in node_coords_ref:
                    if reversed_flag:
                        ename = row["Pt1"]
                    else:
                        ename = row["Pt2"]
                    node_coords_ref[(x_e, y_e)] = ename
                    node_geom[ename] = Point(x_e, y_e)
                else:
                    ename = node_coords_ref[(x_e, y_e)]
            else:
                if (x_s, y_s) not in node_coords_ref:
                    sname = prefix + str(rv_nid)
                    node_coords_ref[(x_s, y_s)] = sname
                    node_geom[sname] = Point(x_s, y_s)
                    rv_nid += 1
                else:
                    sname = node_coords_ref[(x_s, y_s)]

                if (x_e, y_e) not in node_coords_ref:
                    ename = prefix + str(rv_nid)
                    node_coords_ref[(x_e, y_e)] = ename
                    node_geom[ename] = Point(x_e, y_e)
                    rv_nid += 1
                else:
                    ename = node_coords_ref[(x_e, y_e)]

            # ---set the attributes
            if reversed_flag:
                line_graph.add_edge(ename, sname)
                seg_info["geometry"].append(LineString([[x_e, y_e], [x_s, y_s]]))
                seg_info["Pt1"].append(ename)
                seg_info["Pt2"].append(sname)
                seg_info["tauX"].append(-tau[0])
                seg_info["tauY"].append(-tau[1])
                index_list.append(ename + sym + sname)
            else:
                line_graph.add_edge(sname, ename)
                seg_info["geometry"].append(LineString([[x_s, y_s], [x_e, y_e]]))
                seg_info["Pt1"].append(sname)
                seg_info["Pt2"].append(ename)
                seg_info["tauX"].append(tau[0])
                seg_info["tauY"].append(tau[1])
                index_list.append(sname + sym + ename)
                
            if col_namelist is not None:
                for col in col_namelist_recorded:
                    seg_info[col].append(row[col])
                edge_attributes[(sname, ename)] = {col: row[col] for col in col_namelist}
    
    if col_namelist is not None:
        nx.set_edge_attributes(line_graph, values=edge_attributes)

    line_gdf_exploded = gpd.GeoDataFrame(seg_info, index=index_list, crs=line_gdf.crs)
    
    return line_graph, node_geom, line_gdf_exploded
# ********* miscellaneous functions (END) *********


# ********* input formatter for SWMM (START) *********
def shp2swmm(shp_info, save_path, DEM_path, simulation_settings, transection_H5=None, inlet_depth_col="DEPTH", link_type_col="XSECTION", link_geom1_col="GEOM1", link_geom2_col="GEOM2", ManningN_col="ROUGHNESS", sewer_depth=0.6, ponding_area=0.5, min_length_ratio=0.2, min_shift_angle=np.pi/18.0, min_distance=0.5, id_saved=False, shp_saved=False, save_prefix=None, save_epsg=32650, solver_config=None):
    """convert GIS-based files into the input file of SWMM (.inp)

    Args:
        shp_info:
            Input dict which contains the path information of GIS-based geometry files (in vector data format, e.g., ESRI ShapeFile, GeoPackage), for example:
                {
                    "sewer": "sewer.gpkg",
                    "river": "river.gpkg",
                    "river_boundary": "_bc.gpkg",
                    "road_inlet": None,
                    "outfall": "outfalls.gpkg"
                    "culvert": "culvert.gpkg"
                }
        save_path:
            Output basename for SWMM's .inp file.
        DEM_path:
            Input path of DEM file.
        simulation_settings:
            Input dict which describe the simulation settings of SWMM, for example:
                {
                    "start_date": "07/07/2018",
                    "start_time": "09:00:00",
                    "end_date": "07/08/2018",
                    "end_time": "09:00:00",
                    "report_step": "0:01:00",
                }
        transection_H5:
            Input path of river cross section data in HDF5 format.
        inlet_depth_col:
            Column name of inlet depth in `road_inlet`.
        link_type_col:
            Column name of link type in `sewer`, `river`, `culvert` and `road_inlet`.
        link_geom1_col:
            Column name of link height in `sewer`, `river`, `culvert` and `road_inlet`.
        link_geom2_col:
            Column name of link width in `sewer`, `river`, `culvert` and `road_inlet`.
        ManningN_col:
            Column name of Manning's n in `sewer`, `river`, `culvert` and `road_inlet`.
        sewer_depth:
            Default depth of sewer inlets.
        ponding_area:
            Default ponding area of the junction.
        min_length_ratio:
            Minimum length ratio of the segment to be considered as a new segment.
        min_shift_angle:
            Minimum angle between the segment and the line to be considered as a new segment.
        min_distance:
            Minimum distance between the point and the line to be considered as a new segment.
        id_saved:
            Whether to export the ID of the nodes and links (for checking).
        shp_saved:
            Whether to export the geometry files of generated models (for checking).
        save_prefix:
            Output folder for SWMM's .inp file.
        save_epsg:
            EPSG code of the output shapefile.
        solver_config:
            Input dict which describe the solver settings of link direction derivation, for example:
                {
                    "solver": "cbc",
                    "license": "",
                    "path": ""
                }

    Returns:
        None
    """
    inlet_prefix = "IL"
    sewer_prefix = "SW"
    river_prefix = "RV"
    outfall_prefix = "OF"
    outfallAux_prefix = "OAux"
    culvert_prefix = "CL"
    sewer_river_link_prefix = "DRA"
    junction_prefix = "J"
    conduit_prefix = "C"

    tauX_key = "tauX"
    tauY_key = "tauY"
    fromEle_key = "INVERTELE1"
    toEle_key = "INVERTELE2"

    sw_inlet_linkage_prefix = "_".join([inlet_prefix, sewer_prefix])
    rv_inlet_linkage_prefix = "_".join([inlet_prefix, river_prefix])

    d_alpha = 0.01

    link_graph = nx.DiGraph()

    # ------peform the necessary checks for the input files
    if "sewer" in shp_info and os.path.exists(shp_info["sewer"]):
        sewer_path = shp_info["sewer"]
        print(f"Input sewer shapefile found: {sewer_path}")
        sewer_gdf = gpd.read_file(sewer_path, engine="pyogrio", use_arrow=True)
        if link_type_col not in sewer_gdf.columns:
            raise ValueError(f"Column '{link_type_col}' not found in sewer shapefile and please add '{link_type_col}' for link type.")
        if link_geom1_col not in sewer_gdf.columns:
            raise ValueError(f"Column '{link_geom1_col}' not found in sewer shapefile and please add '{link_geom1_col}' for link height.")
        if link_geom2_col not in sewer_gdf.columns:
            raise ValueError(f"Column '{link_geom2_col}' not found in sewer shapefile and please add '{link_geom2_col}' for link width.")
        if ManningN_col not in sewer_gdf.columns:
            raise ValueError(f"Column '{ManningN_col}' not found in sewer shapefile and please add '{ManningN_col}' for link roughness.")
    else:
        raise ValueError(f"Input sewer shapefile not found: {shp_info['sewer']}")

    if "outfall" in shp_info and os.path.exists(shp_info["outfall"]):
        outfall_path = shp_info["outfall"]
        print(f"Input outfall shapefile found: {outfall_path}")
        outfall_gdf = gpd.read_file(outfall_path, engine="pyogrio", use_arrow=True)
        if "INVERTELEV" not in outfall_gdf.columns:
            raise ValueError(f"Column 'INVERTELEV' not found in outfall shapefile and please add 'INVERTELEV' for outfall elevation (by default, you can set it as -100000).")
    else:
        raise ValueError(f"Input outfall shapefile not found: {shp_info['outfall']}")
    
    # ------read .tif file for elevation retrieval
    DEM_ds = gdal.Open(DEM_path)
    DEM_gt = DEM_ds.GetGeoTransform()
    DEM_dta = DEM_ds.ReadAsArray()
    # DEM_dta = fill_nan_nearest(DEM_dta)
    h, w = DEM_dta.shape
    node_ele = {}

    # ------read .shp file and drop unnecessary columns
    # ---------sewers and outfalls are necessary input files
    outfall_gdf = outfall_gdf[np.array(outfall_gdf.apply(lambda x: x["geometry"] is not None, axis=1))]
    num_outfall = outfall_gdf.shape[0]
    outfall_coords_info = {}
    outfall_depth_info = {}
    outfall_type_info = {}
    outfall_coords_info_ref = {}
    outfall_numLinkEdge_info = {}
    outfall_linkNode_Info = {}
    for i in range(0, num_outfall):
        outfall_row_tmp = outfall_gdf.iloc[i]
        outfall_name_tmp = outfall_prefix + str(i)
        coords_tmp = list(outfall_row_tmp["geometry"].coords)[0]
        outfall_coords_info[outfall_name_tmp] = coords_tmp
        outfall_depth_info[outfall_name_tmp] = 0.0
        # outfall_type_info[outfall_name_tmp] = outfall_row_tmp["TYPE"]
        outfall_type_info[outfall_name_tmp] = "FREE"
        outfall_coords_info_ref[coords_tmp] = outfall_name_tmp
        if outfall_row_tmp["INVERTELEV"] > -10000:
            node_ele[outfall_name_tmp] = outfall_row_tmp["INVERTELEV"]
        else:
            #y_tmp, x_tmp = getRasterValLoc(coords_tmp[0], coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
            node_ele[outfall_name_tmp] = getRasterVal_bilinear(DEM_dta, coords_tmp[0], coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5])
        # ---------initialize the number of link edges of outfalls as 0
        outfall_numLinkEdge_info[outfall_name_tmp] = 0
        outfall_linkNode_Info[outfall_name_tmp] = []

    prop_cols = [link_type_col, link_geom1_col, link_geom2_col, ManningN_col]
    drop_col = set(sewer_gdf.columns) - set(prop_cols + ["geometry"])
    sewer_gdf = sewer_gdf.drop(columns=list(drop_col))
    sewer_gdf = sewer_gdf[np.array(sewer_gdf.apply(lambda x: x["geometry"] is not None, axis=1))]
    sewer_gdf = explodeLineSeg(sewer_gdf, tolerance=1.0)

    # ---------rivers, road inlets, culverts and cross sections are optional input files
    if "river" in shp_info:
        if shp_info["river"] is not None and os.path.exists(shp_info["river"]):
            river_path = shp_info["river"]
            river_gdf = gpd.read_file(river_path, engine="pyogrio", use_arrow=True)

            if link_type_col not in river_gdf.columns:
                raise ValueError(f"Column '{link_type_col}' not found in river shapefile and please add '{link_type_col}' for link type.")
            if link_geom1_col not in river_gdf.columns:
                raise ValueError(f"Column '{link_geom1_col}' not found in river shapefile and please add '{link_geom1_col}' for link height.")
            if link_geom2_col not in river_gdf.columns:
                raise ValueError(f"Column '{link_geom2_col}' not found in river shapefile and please add '{link_geom2_col}' for link width.")
            if ManningN_col not in river_gdf.columns:
                raise ValueError(f"Column '{ManningN_col}' not found in river shapefile and please add '{ManningN_col}' for link roughness.")
        
            drop_col = set(river_gdf.columns) - set(prop_cols + ["geometry"])
            river_gdf = river_gdf.drop(columns=list(drop_col))
            river_gdf = explodeLineSeg(river_gdf, tolerance=1.0)
    else:
        river_gdf = None
    
    if "river_boundary" in shp_info:
        if shp_info["river_boundary"] is not None and os.path.exists(shp_info["river_boundary"]):
            river_bc_path = shp_info["river_boundary"]
            river_bc_gdf = gpd.read_file(river_bc_path)
            river_bc_gdf = river_bc_gdf.drop(columns=[col for col in river_bc_gdf.columns if col != "geometry"])
    else:
        river_bc_gdf = None

    if "road_inlet" in shp_info:
        if shp_info["road_inlet"] is not None and os.path.exists(shp_info["road_inlet"]):
            road_inlet_path = shp_info["road_inlet"]
            road_inlet_gdf = gpd.read_file(road_inlet_path, engine="pyogrio", use_arrow=True)

            if link_type_col not in road_inlet_gdf.columns:
                raise ValueError(f"Column '{link_type_col}' not found in road inlet shapefile and please add '{link_type_col}' for link type.")
            if link_geom1_col not in road_inlet_gdf.columns:
                raise ValueError(f"Column '{link_geom1_col}' not found in road inlet shapefile and please add '{link_geom1_col}' for link height.")
            if link_geom2_col not in road_inlet_gdf.columns:
                raise ValueError(f"Column '{link_geom2_col}' not found in road inlet shapefile and please add '{link_geom2_col}' for link width.")
            if inlet_depth_col not in road_inlet_gdf.columns:
                raise ValueError(f"Column '{inlet_depth_col}' not found in road inlet shapefile and please add '{inlet_depth_col}' for link depth.")
            if ManningN_col not in road_inlet_gdf.columns:
                raise ValueError(f"Column '{ManningN_col}' not found in road inlet shapefile and please add '{ManningN_col}' for link roughness.")

            # -----drop the points which do not have geometry or invalid geometry
            road_inlet_gdf = road_inlet_gdf[np.array(road_inlet_gdf.apply(lambda x: x["geometry"] is not None and len(np.where(np.isnan(x["geometry"].coords))[0]) == 0, axis=1))]
    else:
        road_inlet_gdf = None

    if "culvert" in shp_info:
        if shp_info["culvert"] is not None and os.path.exists(shp_info["culvert"]):
            culvert_path = shp_info["culvert"]
            culvert_gdf = gpd.read_file(culvert_path, engine="pyogrio", use_arrow=True)

            if link_type_col not in culvert_gdf.columns:
                raise ValueError(f"Column '{link_type_col}' not found in culvert shapefile and please add '{link_type_col}' for link type.")
            if link_geom1_col not in culvert_gdf.columns:
                raise ValueError(f"Column '{link_geom1_col}' not found in culvert shapefile and please add '{link_geom1_col}' for link height.")
            if link_geom2_col not in culvert_gdf.columns:
                raise ValueError(f"Column '{link_geom2_col}' not found in culvert shapefile and please add '{link_geom2_col}' for link width.")
            if ManningN_col not in culvert_gdf.columns:
                raise ValueError(f"Column '{ManningN_col}' not found in culvert shapefile and please add '{ManningN_col}' for link roughness.")
    else:
        culvert_gdf = None
    
    if transection_H5 is not None and os.path.exists(transection_H5):
        trans_db = h5py.File(transection_H5, "r")
        trans_info = {}
        for trans_name in trans_db.keys():
            trans_info[trans_name] = {}
            trans_info[trans_name]["X"] = trans_db[trans_name]["X"][()]
            trans_info[trans_name]["ELEVATION"] = trans_db[trans_name]["ELEVATION"][()]
            trans_info[trans_name]["ManningN_left"] = trans_db[trans_name]["ManningN_left"][()]
            trans_info[trans_name]["ManningN_right"] = trans_db[trans_name]["ManningN_right"][()]
            trans_info[trans_name]["ManningN_main"] = trans_db[trans_name]["ManningN_main"][()]
    else:
        trans_info = None

    # ------[1] sewer-river topology processing
    # ------we assume that each point shared by the sewer and the river is the linked point of two flow systems
    num_sw_rv = 0
    sw_rv_coords_info = {}
    sw_rv_coords_info_ref = {}
    sw_rv_depth_info = {}
    if river_gdf is not None:
        sw_pt = getLineCoords(sewer_gdf)
        rv_pt = getLineCoords(river_gdf)
        sw_rv_intersect_pt = getIntersect2D(sw_pt, rv_pt)
        for ipt_coords in sw_rv_intersect_pt:
            ipt_coords = (ipt_coords[0], ipt_coords[1])
            ipt_name_tmp = "_".join([sewer_river_link_prefix, junction_prefix+str(num_sw_rv)])
            sw_rv_coords_info[ipt_name_tmp] = ipt_coords
            sw_rv_depth_info[ipt_name_tmp] = sewer_depth
            sw_rv_coords_info_ref[ipt_coords] = ipt_name_tmp
            #y_tmp, x_tmp = getRasterValLoc(ipt_coords[0], ipt_coords[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
            node_ele[ipt_name_tmp] = getRasterVal_bilinear(DEM_dta, ipt_coords[0], ipt_coords[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5]) - sewer_depth
            num_sw_rv += 1

    # ------[2] inlet topology processing
    num_sw_inlet = 0
    num_rv_inlet = 0
    # ---------sewer inlets
    sw_inlet_coords_info = {}
    sw_inlet_coords_info_ref = {}
    sw_inlet_info = {}
    sewer_added_pl = []
    # sewer_added_pt = []
    # ---------river inlets
    rv_inlet_coords_info = {}
    rv_inlet_coords_info_ref = {}
    rv_inlet_info = {}
    river_added_pl = []
    # river_added_pt = []
    river_added_outfall = []

    added_flag = False    # flag which indicates whether new points are inserted into the sewer/river
    #merged_flag = False   # flag which indicates whether inlets are merged into the sewer/river
    if road_inlet_gdf is not None:
        road_inlet_gdf = road_inlet_gdf[np.array(road_inlet_gdf.apply(lambda x: x["geometry"] is not None and not x["geometry"].is_empty, axis=1))]
        for ids, inlet_row_tmp in road_inlet_gdf.iterrows():
            inlet_tmp = inlet_row_tmp["geometry"]
            inlet_depth_tmp = inlet_row_tmp[inlet_depth_col]
            inlet_type_tmp = inlet_row_tmp[link_type_col]
            if "RECT" in inlet_type_tmp:
                inlet_type_tmp = "RECT_CLOSED"
            inlet_geom1_tmp = inlet_row_tmp[link_geom1_col]
            inlet_geom2_tmp = inlet_row_tmp[link_geom2_col] if inlet_row_tmp[link_geom2_col] is not None else 0.0
            # inlet_N_tmp = inlet_row_tmp[ManningN_col]
            inlet_N_tmp = 0.013

            inlet_sw_distance = sewer_gdf["geometry"].distance(inlet_tmp).values
            loc_sw = np.argmin(inlet_sw_distance)
            inlet_sw_distance_min = inlet_sw_distance[loc_sw]

            if river_gdf is not None:
                inlet_rv_distance = river_gdf["geometry"].distance(inlet_tmp).values
                loc_rv = np.argmin(inlet_rv_distance)
                inlet_rv_distance_min = inlet_rv_distance[loc_rv]
            elif river_bc_gdf is not None:
                inlet_rv_distance = river_bc_gdf["geometry"].distance(inlet_tmp).values
                loc_rv = np.argmin(inlet_rv_distance)
                inlet_rv_distance_min = inlet_rv_distance[loc_rv]
            else:
                inlet_rv_distance = None
                loc_rv = None
                inlet_rv_distance_min = 1e9
            
            # ------attach the point to the river or sewer based on distance
            #print(inlet_sw_distance_min, inlet_rv_distance_min, inlet_tmp, inlet_sw_distance)
            if inlet_sw_distance_min < inlet_rv_distance_min:
                sewer_row_tmp = sewer_gdf.iloc[loc_sw]
                sewer_seg = sewer_row_tmp["geometry"]
                sewer_snapped_res = snapPointToLine(inlet_tmp, sewer_seg, min_length_ratio, min_shift_angle, min_distance)
                pt_sewer_snapped = sewer_snapped_res["geometry"]
                # ---------if the projected distance between the point and boundaries of the segment is larger than a threshold,
                # ------------a new point must be inserted into the segment
                if sewer_snapped_res["projected_distance"] > min_distance:
                    if sewer_snapped_res["snapped_distance"] < min_distance:
                        inlet_coords_tmp = list(pt_sewer_snapped.coords)[0]
                        inlet_depth_tmp = sewer_depth
                        inlet_type_tmp = sewer_row_tmp[link_type_col]
                        inlet_geom1_tmp = sewer_row_tmp[link_geom1_col]
                        inlet_geom2_tmp = sewer_row_tmp[link_geom2_col]
                        inlet_N_tmp = sewer_row_tmp[ManningN_col]
                        added_flag = False
                        #merged_flag = True
                    else:
                        inlet_coords_tmp = list(inlet_tmp.coords)[0]
                        line_tmp = LineString([inlet_tmp, pt_sewer_snapped])
                        sewer_added_pl.append(line_tmp)
                        added_flag = True
                        #merged_flag = False
                    # sewer_added_pt.append(pt_sewer_snapped)
                    # ------------add two rows into the GeoDataFrame due to the insertion of snapped point
                    pt_seg_from, pt_seg_to = sewer_seg.boundary
                    prop_info = {c: [sewer_row_tmp[c], sewer_row_tmp[c]] for c in prop_cols}
                    seg_new = {"geometry": [LineString([pt_seg_from, pt_sewer_snapped]), LineString([pt_sewer_snapped, pt_seg_to])], **prop_info}
                    sewer_gdf = sewer_gdf.drop(sewer_gdf.index[loc_sw])
                    sewer_gdf = gpd.GeoDataFrame(pd.concat([sewer_gdf, gpd.GeoDataFrame(seg_new)], ignore_index=True), crs=sewer_gdf.crs)
                # ---------if the projected distance is small (defalut set by 0) but snapped distance is larger than a threshold,
                # ------------we assert that the snapped point must be one of boundaries of the segment
                # ------------thus no new point is needed to be inserted into the segment
                elif sewer_snapped_res["snapped_distance"] > min_distance:
                    inlet_coords_tmp = list(inlet_tmp.coords)[0]
                    line_tmp = LineString([inlet_tmp, pt_sewer_snapped])
                    sewer_added_pl.append(line_tmp)
                    added_flag = True
                    #merged_flag = False
                # ---------if both the projected distance and the snapped distance is small,
                # ------------we move the inlet to the snapped point on the segment (and thus no new point is added)
                # ------------and we get node information from the corresponding link
                # ------------Note: inlets simplified from the road network fall into this category !!!
                else:
                    inlet_coords_tmp = list(pt_sewer_snapped.coords)[0]
                    inlet_depth_tmp = sewer_depth
                    inlet_type_tmp = sewer_row_tmp[link_type_col]
                    inlet_geom1_tmp = sewer_row_tmp[link_geom1_col]
                    inlet_geom2_tmp = sewer_row_tmp[link_geom2_col]
                    inlet_N_tmp = sewer_row_tmp[ManningN_col]
                    added_flag = False
                    #merged_flag = True
                # ---------save the information of the inlet (if two inlets are too close, we simply keep one of them)
                if inlet_coords_tmp not in sw_inlet_coords_info_ref:
                    inlet_name_tmp = "_".join([inlet_prefix, sewer_prefix, junction_prefix+str(num_sw_inlet)])
                    sw_inlet_coords_info[inlet_name_tmp] = inlet_coords_tmp
                    sw_inlet_coords_info_ref[inlet_coords_tmp] = inlet_name_tmp
                    sw_inlet_info[inlet_name_tmp] = {}
                    sw_inlet_info[inlet_name_tmp]["DEPTH"] = inlet_depth_tmp
                    sw_inlet_info[inlet_name_tmp]["TYPE"] = inlet_type_tmp
                    sw_inlet_info[inlet_name_tmp]["GEOM1"] = inlet_geom1_tmp
                    sw_inlet_info[inlet_name_tmp]["GEOM2"] = inlet_geom2_tmp
                    sw_inlet_info[inlet_name_tmp]["N"] = inlet_N_tmp
                    sw_inlet_info[inlet_name_tmp]["ADDED"] = added_flag
                    #y_tmp, x_tmp = getRasterValLoc(inlet_coords_tmp[0], inlet_coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
                    node_ele[inlet_name_tmp] = getRasterVal_bilinear(DEM_dta, inlet_coords_tmp[0], inlet_coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5]) - inlet_depth_tmp
                    num_sw_inlet += 1
            else:
                if river_gdf is not None:
                    river_row_tmp = river_gdf.iloc[loc_rv]
                    river_seg = river_row_tmp["geometry"]
                    river_snapped_res = snapPointToLine(inlet_tmp, river_seg, min_length_ratio, min_shift_angle, min_distance)
                    pt_river_snapped = river_snapped_res["geometry"]
                    if river_snapped_res["projected_distance"] > min_distance:
                        if river_snapped_res["snapped_distance"] < min_distance:
                            inlet_coords_tmp = list(pt_river_snapped.coords)[0]
                            inlet_type_tmp = river_row_tmp[link_type_col]
                            # ---------there are situations where the river goes through a closed culvert so the depth of node should be considered
                            if inlet_type_tmp not in ["CIRCULAR", "RECT_CLOSED"]:
                                inlet_depth_tmp = 0.0
                            inlet_geom1_tmp = river_row_tmp[link_geom1_col]
                            inlet_geom2_tmp = river_row_tmp[link_geom2_col]
                            inlet_N_tmp = river_row_tmp[ManningN_col]
                            added_flag = False
                            #merged_flag = True
                        else:
                            inlet_coords_tmp = list(inlet_tmp.coords)[0]
                            line_tmp = LineString([inlet_tmp, pt_river_snapped])
                            river_added_pl.append(line_tmp)
                            added_flag = True
                            #merged_flag = False
                        # river_added_pt.append(pt_river_snapped)
                        pt_seg_from, pt_seg_to = river_seg.boundary
                        prop_info = {c: [river_row_tmp[c], river_row_tmp[c]] for c in prop_cols}
                        seg_new = {"geometry": [LineString([pt_seg_from, pt_river_snapped]), LineString([pt_river_snapped, pt_seg_to])], **prop_info}
                        river_gdf = river_gdf.drop(river_gdf.index[loc_rv])
                        river_gdf = gpd.GeoDataFrame(pd.concat([river_gdf, gpd.GeoDataFrame(seg_new)], ignore_index=True), crs=river_gdf.crs)
                    elif river_snapped_res["snapped_distance"] > min_distance:
                        inlet_coords_tmp = list(inlet_tmp.coords)[0]
                        line_tmp = LineString([inlet_tmp, pt_river_snapped])
                        river_added_pl.append(line_tmp)
                        added_flag = True
                        #merged_flag = False
                    else:
                        inlet_coords_tmp = list(pt_river_snapped.coords)[0]
                        inlet_type_tmp = river_row_tmp[link_type_col]
                        # ---------there are situations where the river goes through a closed culvert so the depth of node should be considered
                        if inlet_type_tmp not in ["CIRCULAR", "RECT_CLOSED"]:
                            inlet_depth_tmp = 0.0
                        inlet_geom1_tmp = river_row_tmp[link_geom1_col]
                        inlet_geom2_tmp = river_row_tmp[link_geom2_col]
                        inlet_N_tmp = river_row_tmp[ManningN_col]
                        added_flag = False
                else:
                    inlet_coords_tmp = list(inlet_tmp.coords)[0]
                    river_bc_row_tmp = river_bc_gdf.iloc[loc_rv]
                    river_bc_seg = river_bc_row_tmp["geometry"]
                    river_snapped_res = snapPointToLine(inlet_tmp, river_bc_seg, min_length_ratio, min_shift_angle, min_distance)
                    pt_river_snapped = river_snapped_res["geometry"]
                    if pt_river_snapped not in river_added_outfall:
                        river_added_outfall.append(pt_river_snapped)
                    # ------******An outfall node is only permitted to have one link attached to it.******------
                    else:
                        x1, y1 = river_bc_seg.coords[0]
                        x2, y2 = river_bc_seg.coords[1]
                        alpha = river_bc_seg.project(pt_river_snapped) / river_bc_seg.length
                        if alpha > 0.5:
                            d_dir = -1
                        else:
                            d_dir = 1
                        alpha += d_alpha * d_dir
                        pt_river_snapped = Point(x1 + alpha * (x2 - x1), y1 + alpha * (y2 - y1))
                        while (pt_river_snapped in river_added_outfall):
                            alpha += d_alpha * d_dir
                            pt_river_snapped = Point(x1 + alpha * (x2 - x1), y1 + alpha * (y2 - y1))
                        river_added_outfall.append(pt_river_snapped)
                    line_tmp = LineString([inlet_tmp, pt_river_snapped])
                    river_added_pl.append(line_tmp)
                    added_flag = True
                
                if inlet_coords_tmp not in rv_inlet_coords_info_ref:
                    inlet_name_tmp = "_".join([inlet_prefix, river_prefix, junction_prefix+str(num_rv_inlet)])
                    rv_inlet_coords_info[inlet_name_tmp] = inlet_coords_tmp
                    rv_inlet_coords_info_ref[inlet_coords_tmp] = inlet_name_tmp
                    rv_inlet_info[inlet_name_tmp] = {}
                    rv_inlet_info[inlet_name_tmp]["DEPTH"] = inlet_depth_tmp
                    rv_inlet_info[inlet_name_tmp]["TYPE"] = inlet_type_tmp
                    rv_inlet_info[inlet_name_tmp]["GEOM1"] = inlet_geom1_tmp
                    rv_inlet_info[inlet_name_tmp]["GEOM2"] = inlet_geom2_tmp
                    rv_inlet_info[inlet_name_tmp]["N"] = inlet_N_tmp
                    rv_inlet_info[inlet_name_tmp]["ADDED"] = added_flag
                    # y_tmp, x_tmp = getRasterValLoc(inlet_coords_tmp[0], inlet_coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
                    node_ele[inlet_name_tmp] = getRasterVal_bilinear(DEM_dta, inlet_coords_tmp[0], inlet_coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5]) - inlet_depth_tmp
                    num_rv_inlet += 1

    # ------[3] add node and line information of sewers
    sw_id = 0
    sw_inlet_linkage_id = 0
    sw_line_id = 0
    sw_node_coords_info = {}
    sw_node_depth_info = {}
    sw_node_coords_info_ref = {}
    sw_line_info = {}
    sw_inlet_linkage_info = {}

    for ids, row in sewer_gdf.iterrows():
        sw_seg = row["geometry"]
        if len(sw_seg.boundary.geoms) == 0:
            continue
        sw_type, sw_geom1, sw_geom2, sw_N = row[link_type_col], row[link_geom1_col], row[link_geom2_col], row[ManningN_col]
        if "RECT" in sw_type:
            sw_type = "RECT_CLOSED"
        sw_start, sw_end = sw_seg.coords

        linkage_flag_start = False
        linkage_flag_end = False
        
        outfall_flag = False
        if sw_start in sw_node_coords_info_ref.keys():
            from_name = sw_node_coords_info_ref[sw_start]
        else:
            # ---------test whether `sw_start` is a inlet-(sewer) linkage node
            if sw_start in sw_inlet_coords_info_ref:
                linkage_flag_start = True
                # ------by default, we set flow from the inlet node to the sewer node
                from_name = sw_inlet_coords_info_ref[sw_start]
                # ---------if the inlet node is added into the sewer,
                # ------------we retrieve the information of this link from the inlet node
                if sw_inlet_info[from_name]["ADDED"]:
                    sw_type = "RECT_CLOSED" if "RECT" in sw_inlet_info[from_name]["TYPE"] else "CIRCULAR"
                    sw_geom1 = sw_inlet_info[from_name]["GEOM1"]
                    sw_geom2 = sw_inlet_info[from_name]["GEOM2"]
                    sw_N = sw_inlet_info[from_name]["N"]
            else:
                # ---------test whether `sw_start` is a sewer-(river) linkage node
                if sw_start in sw_rv_coords_info_ref:
                    from_name = sw_rv_coords_info_ref[sw_start]
                # ---------test whether `sw_start` is a outfall node
                elif sw_start in outfall_coords_info_ref.keys():
                    from_name = outfall_coords_info_ref[sw_start]
                    outfall_flag = True
                else:
                    # ------since this node can not be found in all types of the current node set, 
                    # ---------we need to add it to the sewer node set
                    sw_name_tmp = "_".join([sewer_prefix, junction_prefix+str(sw_id)])
                    sw_node_coords_info[sw_name_tmp] = sw_start
                    sw_node_depth_info[sw_name_tmp] = sewer_depth
                    sw_node_coords_info_ref[sw_start] = sw_name_tmp
                    # y_tmp, x_tmp = getRasterValLoc(sw_start[0], sw_start[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
                    node_ele[sw_name_tmp] = getRasterVal_bilinear(DEM_dta, sw_start[0], sw_start[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5]) - sewer_depth
                    sw_id += 1
                    from_name = sw_name_tmp

        if sw_end in sw_node_coords_info_ref:
            to_name = sw_node_coords_info_ref[sw_end]
        else:
            if sw_end in sw_inlet_coords_info_ref:
                linkage_flag_end = True
                to_name = sw_inlet_coords_info_ref[sw_end]
            else:
                if sw_end in sw_rv_coords_info_ref:
                    to_name = sw_rv_coords_info_ref[sw_end]
                elif sw_end in outfall_coords_info_ref:
                    to_name = outfall_coords_info_ref[sw_end]
                    outfall_flag = True
                else:
                    sw_name_tmp = "_".join([sewer_prefix, junction_prefix+str(sw_id)])
                    sw_node_coords_info[sw_name_tmp] = sw_end
                    sw_node_depth_info[sw_name_tmp] = sewer_depth
                    sw_node_coords_info_ref[sw_end] = sw_name_tmp
                    # y_tmp, x_tmp = getRasterValLoc(sw_end[0], sw_end[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
                    node_ele[sw_name_tmp] = getRasterVal_bilinear(DEM_dta, sw_end[0], sw_end[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5]) - sewer_depth
                    sw_id += 1
                    to_name = sw_name_tmp
        
        # ------save sewer linkage information
        if outfall_prefix in from_name:
            to_name_f = from_name
            from_name_f = to_name
        elif outfall_prefix in to_name:
            to_name_f = to_name
            from_name_f = from_name
        # ---------we may further consider the added case 
        # ------------where we must set the flow direction from the inlet to the sewer
        else:
            # ---------if both points are in the set of inlet-(sewer) nodes
            # ------------which means that both inlets are added into the sewer 
            # ------------we still treat it as a normal sewer link
            # ---------OR if there is no point in the set of inlet-(sewer) nodes, 
            # ------------we determine the link direction based on the elevation of the two points
            if (linkage_flag_start and linkage_flag_end) or (not linkage_flag_start and not linkage_flag_end):
                linkage_flag = False
                if node_ele[from_name] > node_ele[to_name]:
                    to_name_f = to_name
                    from_name_f = from_name
                else:
                    to_name_f = from_name
                    from_name_f = to_name
            # ---------if there is only one point in the set of inlet-(sewer) nodes, 
            # ------------we must the link direction as inlet node -> sewer node
            elif linkage_flag_start:
                from_name_f = from_name
                to_name_f = to_name
                linkage_flag = True
            #elif linkage_flag_end:
            else:
                from_name_f = to_name
                to_name_f = from_name
                linkage_flag = True

        # ------check whether linkage information of outfalls should be updated
        if outfall_flag:
            if linkage_flag:
                outfall_numLinkEdge_info[to_name_f] += 1
                outfall_linkNode_Info[to_name_f].append({"NAME": from_name_f, 
                                                        "LINK_NAME": "_".join([inlet_prefix, sewer_prefix, conduit_prefix+str(sw_inlet_linkage_id)])})
            else:
                outfall_numLinkEdge_info[to_name_f] += 1
                outfall_linkNode_Info[to_name_f].append({"NAME": from_name_f, 
                                                        "LINK_NAME": "_".join([sewer_prefix, conduit_prefix+str(sw_line_id)])})
            
        if linkage_flag:
            sw_line_name_tmp = "_".join([inlet_prefix, sewer_prefix, conduit_prefix+str(sw_inlet_linkage_id)])
            sw_inlet_linkage_info[sw_line_name_tmp] = {}
            sw_inlet_linkage_info[sw_line_name_tmp]["FROM"] = from_name_f
            sw_inlet_linkage_info[sw_line_name_tmp]["TO"] = to_name_f
            sw_inlet_linkage_info[sw_line_name_tmp]["TYPE"] = sw_type
            sw_inlet_linkage_info[sw_line_name_tmp]["GEOM1"] = sw_geom1
            sw_inlet_linkage_info[sw_line_name_tmp]["GEOM2"] = sw_geom2 if sw_geom2 is not None else 0.0
            sw_inlet_linkage_info[sw_line_name_tmp]["N"] = sw_N
            sw_inlet_linkage_info[sw_line_name_tmp]["LENGTH"] = np.sqrt((sw_start[0]-sw_end[0])**2+(sw_start[1]-sw_end[1])**2)
            sw_inlet_linkage_id += 1
        else:
            sw_line_name_tmp = "_".join([sewer_prefix, conduit_prefix+str(sw_line_id)])
            sw_line_info[sw_line_name_tmp] = {}
            sw_line_info[sw_line_name_tmp]["FROM"] = from_name_f
            sw_line_info[sw_line_name_tmp]["TO"] = to_name_f
            sw_line_info[sw_line_name_tmp]["TYPE"] = sw_type
            sw_line_info[sw_line_name_tmp]["GEOM1"] = sw_geom1
            sw_line_info[sw_line_name_tmp]["GEOM2"] = sw_geom2 if sw_geom2 is not None else 0.0
            sw_line_info[sw_line_name_tmp]["N"] = sw_N
            sw_line_info[sw_line_name_tmp]["LENGTH"] = np.sqrt((sw_start[0]-sw_end[0])**2+(sw_start[1]-sw_end[1])**2)
            sw_line_id += 1

        link_graph.add_edge(from_name_f, to_name_f, 
                            NAME=sw_line_name_tmp, 
                            GEOM1=sw_geom1, 
                            GEOM2=sw_geom2 if sw_geom2 is not None else 0.0,
                            TYPE=sw_type)
        # print("Sewer link: {} -> {}".format(from_name_f, to_name_f))
        
    if len(sewer_added_pl)> 0:
        for sw_line in sewer_added_pl:
            sw_start, sw_end = sw_line.boundary
            sw_start = list(sw_start.coords)[0]
            sw_end = list(sw_end.coords)[0]
            from_name = sw_inlet_coords_info_ref[sw_start]
            sw_line_name_tmp = "_".join([inlet_prefix, sewer_prefix, conduit_prefix+str(sw_inlet_linkage_id)])
            sw_inlet_linkage_info[sw_line_name_tmp] = {}
            sw_inlet_linkage_info[sw_line_name_tmp]["FROM"] = from_name
            if sw_end in sw_node_coords_info_ref:
                to_name = sw_node_coords_info_ref[sw_end]
            elif sw_end in outfall_coords_info_ref:
                to_name = outfall_coords_info_ref[sw_end]
                outfall_numLinkEdge_info[to_name] += 1
                # `DIR = -1` indicates that the flow direction is from the sewer to the outfall
                outfall_linkNode_Info[to_name].append({"NAME": from_name, 
                                                       "LINK_NAME": sw_line_name_tmp})
            else:
                to_name = sw_inlet_coords_info_ref[sw_end]
            sw_inlet_linkage_info[sw_line_name_tmp]["TO"] = to_name
            sw_inlet_linkage_info[sw_line_name_tmp]["TYPE"] = sw_inlet_info[from_name]["TYPE"]
            sw_inlet_linkage_info[sw_line_name_tmp]["GEOM1"] = sw_inlet_info[from_name]["GEOM1"]
            sw_inlet_linkage_info[sw_line_name_tmp]["GEOM2"] = sw_inlet_info[from_name]["GEOM2"]
            sw_inlet_linkage_info[sw_line_name_tmp]["N"] = sw_inlet_info[from_name]["N"]
            sw_inlet_linkage_info[sw_line_name_tmp]["LENGTH"] = np.sqrt((sw_start[0]-sw_end[0])**2+(sw_start[1]-sw_end[1])**2)
            link_graph.add_edge(from_name, to_name, 
                                NAME=sw_line_name_tmp, 
                                GEOM1=sw_inlet_linkage_info[sw_line_name_tmp]["GEOM1"], 
                                GEOM2=sw_inlet_linkage_info[sw_line_name_tmp]["GEOM1"],
                                TYPE=sw_inlet_linkage_info[sw_line_name_tmp]["TYPE"],)
            # print("Sewer link: {} -> {}".format(from_name, to_name))
            sw_inlet_linkage_id  += 1
    
    # ------[4] add node and line information of rivers
    rv_id = 0
    rv_inlet_linkage_id = 0
    rv_line_id = 0
    rv_node_coords_info = {}
    rv_node_depth_info = {}
    rv_node_coords_info_ref = {}
    rv_line_info = {}
    rv_inlet_linkage_info = {}
    if river_gdf is not None:
        for ids, row in river_gdf.iterrows():
            rv_seg = row["geometry"]
            if len(rv_seg.boundary) == 0:
                continue
            rv_type, rv_geom1, rv_geom2, rv_N = row[link_type_col], row[link_geom1_col], row[link_geom2_col], row[ManningN_col]
            rv_start, rv_end = rv_seg.coords

            linkage_flag_start = False
            linkage_flag_end = False

            outfall_flag = False
            if rv_start in rv_node_coords_info_ref.keys():
                from_name = rv_node_coords_info_ref[rv_start]
                # ---------modify the depth of the node if any conduits which the node belongs to are "CIRCULAR", "RECT_CLOSED"
                if rv_type in ["CIRCULAR", "RECT_CLOSED"]:
                    rv_node_depth_info[from_name] = sewer_depth * 3
            else:
                if rv_start in rv_inlet_coords_info_ref:
                    linkage_flag_start = True
                    # ------by default, we set flow from the inlet node to the sewer node
                    from_name = rv_inlet_coords_info_ref[rv_start]
                    if rv_inlet_info[from_name]["ADDED"]:
                        rv_type = rv_inlet_info[from_name]["TYPE"]
                        rv_geom1 = rv_inlet_info[from_name]["GEOM1"]
                        rv_geom2 = rv_inlet_info[from_name]["GEOM2"]
                        rv_N = rv_inlet_info[from_name]["N"]
                else:
                    if rv_start in sw_rv_coords_info_ref:
                        from_name = sw_rv_coords_info_ref[rv_start]
                    elif rv_start in outfall_coords_info_ref.keys():
                        from_name = outfall_coords_info_ref[rv_start]
                        outfall_flag = True
                    else:
                        # ------save sewer node information
                        rv_name_tmp = "_".join([river_prefix, junction_prefix+str(rv_id)])
                        rv_node_coords_info[rv_name_tmp] = rv_start
                        rv_node_coords_info_ref[rv_start] = rv_name_tmp
                        if rv_type in ["CIRCULAR", "RECT_CLOSED"]:
                            rv_node_depth_info[rv_name_tmp] = sewer_depth * 3
                        else:
                            rv_node_depth_info[rv_name_tmp] = 0.0
                        #y_tmp, x_tmp = getRasterValLoc(rv_start[0], rv_start[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
                        node_ele[rv_name_tmp] = getRasterVal_bilinear(DEM_dta, rv_start[0], rv_start[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5])
                        rv_id += 1
                        from_name = rv_name_tmp

            if rv_end in rv_node_coords_info_ref.keys():
                to_name = rv_node_coords_info_ref[rv_end]
            else:
                if rv_end in rv_inlet_coords_info_ref.keys():
                    linkage_flag_end = True
                    to_name = rv_inlet_coords_info_ref[rv_end]
                else:
                    if rv_end in sw_rv_coords_info_ref:
                        to_name = sw_rv_coords_info_ref[rv_end]
                    elif rv_end in outfall_coords_info_ref.keys():
                        to_name = outfall_coords_info_ref[rv_end]
                        outfall_flag = True
                    else:
                        rv_name_tmp = "_".join([river_prefix, junction_prefix+str(rv_id)])
                        rv_node_coords_info[rv_name_tmp] = rv_end
                        rv_node_coords_info_ref[rv_end] = rv_name_tmp
                        if rv_type in ["CIRCULAR", "RECT_CLOSED"]:
                            rv_node_depth_info[rv_name_tmp] = sewer_depth * 3
                        else:
                            rv_node_depth_info[rv_name_tmp] = 0.0
                        #y_tmp, x_tmp = getRasterValLoc(rv_end[0], rv_end[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
                        node_ele[rv_name_tmp] = getRasterVal_bilinear(DEM_dta, rv_end[0], rv_end[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5])
                        rv_id += 1
                        to_name = rv_name_tmp

            # ------save river linkage information
            if outfall_prefix in from_name:
                to_name_f = from_name
                from_name_f = to_name
            elif outfall_prefix in to_name:
                to_name_f = to_name
                from_name_f = from_name
            else:
                if (linkage_flag_start and linkage_flag_end) or (not linkage_flag_start and not linkage_flag_end):
                    linkage_flag = False
                    if node_ele[from_name] > node_ele[to_name]:
                        to_name_f = to_name
                        from_name_f = from_name
                    else:
                        to_name_f = from_name
                        from_name_f = to_name
                elif linkage_flag_start:
                    from_name_f = from_name
                    to_name_f = to_name
                    linkage_flag = True
                #elif linkage_flag_end:
                else:
                    from_name_f = to_name
                    to_name_f = from_name
                    linkage_flag = True
            
            # ------check whether linkage information of outfalls should be updated
            if outfall_flag:
                outfall_numLinkEdge_info[to_name_f] += 1
                if linkage_flag:
                    outfall_linkNode_Info[to_name_f].append({"NAME": from_name_f,
                                                            "LINK_NAME": "_".join([inlet_prefix, river_prefix, conduit_prefix+str(rv_inlet_linkage_id)])})
                else:
                    outfall_linkNode_Info[to_name_f].append({"NAME": from_name_f,
                                                            "LINK_NAME": "_".join([river_prefix, conduit_prefix+str(rv_line_id)])})
                        
            if linkage_flag:
                rv_line_name_tmp = "_".join([inlet_prefix, river_prefix, conduit_prefix+str(rv_inlet_linkage_id)])
                rv_inlet_linkage_info[rv_line_name_tmp] = {}
                rv_inlet_linkage_info[rv_line_name_tmp]["FROM"] = from_name_f
                rv_inlet_linkage_info[rv_line_name_tmp]["TO"] = to_name_f
                rv_inlet_linkage_info[rv_line_name_tmp]["TYPE"] = rv_type
                rv_inlet_linkage_info[rv_line_name_tmp]["GEOM1"] = rv_geom1
                rv_inlet_linkage_info[rv_line_name_tmp]["GEOM2"] = rv_geom2 if rv_geom2 is not None else 0.0
                rv_inlet_linkage_info[rv_line_name_tmp]["N"] = rv_N
                rv_inlet_linkage_info[rv_line_name_tmp]["LENGTH"] = np.sqrt((rv_start[0]-rv_end[0])**2+(rv_start[1]-rv_end[1])**2)
                rv_inlet_linkage_id += 1
            else:
                rv_line_name_tmp = "_".join([river_prefix, conduit_prefix+str(rv_line_id)])
                rv_line_info[rv_line_name_tmp] = {}
                rv_line_info[rv_line_name_tmp]["FROM"] = from_name_f
                rv_line_info[rv_line_name_tmp]["TO"] = to_name_f
                rv_line_info[rv_line_name_tmp]["TYPE"] = rv_type
                rv_line_info[rv_line_name_tmp]["GEOM1"] = rv_geom1
                rv_line_info[rv_line_name_tmp]["GEOM2"] = rv_geom2 if rv_geom2 is not None else 0.0
                rv_line_info[rv_line_name_tmp]["N"] = rv_N
                rv_line_info[rv_line_name_tmp]["LENGTH"] = np.sqrt((rv_start[0]-rv_end[0])**2+(rv_start[1]-rv_end[1])**2)
                rv_line_id += 1

            link_graph.add_edge(from_name_f, to_name_f, 
                                NAME=rv_line_name_tmp, 
                                GEOM1=rv_geom1, 
                                GEOM2=rv_geom2 if rv_geom2 is not None else 0.0,
                                TYPE=rv_type)

    if len(river_added_outfall) > 0:
        num_rv_outfall = len(river_added_outfall)
        for i in range(0, num_rv_outfall):
            outfall_name_tmp = "_".join([outfall_prefix, river_prefix, junction_prefix+str(i)])
            coords_tmp = list(river_added_outfall[i].coords)[0]
            outfall_coords_info[outfall_name_tmp] = coords_tmp
            outfall_depth_info[outfall_name_tmp] = 0.0
            outfall_type_info[outfall_name_tmp] = "FIXED"
            outfall_coords_info_ref[coords_tmp] = outfall_name_tmp
            # y_tmp, x_tmp = getRasterValLoc(coords_tmp[0], coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5], h, w)
            node_ele[outfall_name_tmp] = getRasterVal_bilinear(DEM_dta, coords_tmp[0], coords_tmp[1], DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5])
    
    if len(river_added_pl)> 0:
        for rv_line in river_added_pl:
            rv_start, rv_end = rv_line.boundary
            rv_start = list(rv_start.coords)[0]
            rv_end = list(rv_end.coords)[0]
            from_name = rv_inlet_coords_info_ref[rv_start]
            rv_line_name_tmp = "_".join([inlet_prefix, river_prefix, conduit_prefix+str(rv_inlet_linkage_id)])
            rv_inlet_linkage_info[rv_line_name_tmp] = {}
            rv_inlet_linkage_info[rv_line_name_tmp]["FROM"] = from_name
            if rv_end in rv_node_coords_info_ref.keys():
                to_name = rv_node_coords_info_ref[rv_end]
            elif rv_end in outfall_coords_info_ref.keys():
                to_name = outfall_coords_info_ref[rv_end]
                outfall_numLinkEdge_info[to_name] += 1
                # `DIR = -1` indicates that the flow direction is from the sewer to the outfall
                outfall_linkNode_Info[to_name].append({"NAME": from_name, 
                                                       "LINK_NAME": rv_line_name_tmp})
            else:
                to_name = rv_inlet_coords_info_ref[rv_end]
            rv_inlet_linkage_info[rv_line_name_tmp]["TO"] = to_name
            rv_inlet_linkage_info[rv_line_name_tmp]["TYPE"] = rv_inlet_info[from_name]["TYPE"]
            rv_inlet_linkage_info[rv_line_name_tmp]["GEOM1"] = rv_inlet_info[from_name]["GEOM1"]
            rv_inlet_linkage_info[rv_line_name_tmp]["GEOM2"] = rv_inlet_info[from_name]["GEOM2"]
            rv_inlet_linkage_info[rv_line_name_tmp]["N"] = rv_inlet_info[from_name]["N"]
            rv_inlet_linkage_info[rv_line_name_tmp]["LENGTH"] = np.sqrt((rv_start[0]-rv_end[0])**2+(rv_start[1]-rv_end[1])**2)
            link_graph.add_edge(from_name, to_name, 
                                NAME=rv_line_name_tmp, 
                                GEOM1=rv_inlet_linkage_info[rv_line_name_tmp]["GEOM1"], 
                                GEOM2=rv_inlet_linkage_info[rv_line_name_tmp]["GEOM1"],
                                TYPE=rv_inlet_linkage_info[rv_line_name_tmp]["TYPE"],)
            rv_inlet_linkage_id  += 1
    
    # ---------adjust the linkage of outfalls if their linkage number is greater than 1
    # ------------******An outfall node is only permitted to have one link attached to it.******------
    num_outfallAux = 0
    for ofn in outfall_numLinkEdge_info.keys():
        if outfall_numLinkEdge_info[ofn] > 1:
            ofnAux_name = outfallAux_prefix + "_" + str(num_outfallAux)
            # ------------get the information of old outfall node
            ofn_x, ofn_y = outfall_coords_info[ofn]
            # ------------convert the outfall to a junction node
            node_ele[ofnAux_name] = getRasterVal_bilinear(DEM_dta, ofn_x, ofn_y, DEM_gt[0], DEM_gt[3], DEM_gt[1], DEM_gt[5]) - sewer_depth
            sw_node_coords_info[ofnAux_name] = outfall_coords_info[ofn]
            sw_node_depth_info[ofnAux_name] = sewer_depth
            sw_node_coords_info_ref[outfall_coords_info[ofn]] = ofnAux_name
            num_outfallAux += 1
            # ------------delete the old linkage and create the new linkage
            hydraulic_diameter_acc = 0.0
            length_acc = 0.0
            for ofn_linkInfo in outfall_linkNode_Info[ofn]:
                otherNode_name = ofn_linkInfo["NAME"]
                oflink_name = ofn_linkInfo["LINK_NAME"]
                # ---------------for existing linkages, we only need to change the FROM/TO node name
                edge_length = 0.0
                if oflink_name.startswith("_".join([inlet_prefix, sewer_prefix])):
                    sw_inlet_linkage_info[oflink_name]["TO"] = ofnAux_name
                    edge_length = sw_inlet_linkage_info[oflink_name]["LENGTH"]
                elif oflink_name.startswith(sewer_prefix):
                    sw_line_info[oflink_name]["TO"] = ofnAux_name
                    edge_length = sw_line_info[oflink_name]["LENGTH"]
                elif oflink_name.startswith("_".join([inlet_prefix, river_prefix])):
                    rv_inlet_linkage_info[oflink_name]["TO"] = ofnAux_name
                    edge_length = rv_inlet_linkage_info[oflink_name]["LENGTH"]
                elif oflink_name.startswith(river_prefix):
                    rv_line_info[oflink_name]["TO"] = ofnAux_name
                    edge_length = rv_line_info[oflink_name]["LENGTH"]

                # ------------get the old edge information
                edge_info = link_graph.get_edge_data(otherNode_name, ofn)
                # ------------create the new linkage on graph
                link_graph.remove_edge(otherNode_name, ofn)
                link_graph.add_edge(otherNode_name, ofnAux_name, **edge_info)
                
                # ------------calculate the hydraulic diameter information
                if edge_info["TYPE"] == "CIRCULAR":
                    diameter = edge_info["GEOM1"]
                elif edge_info["TYPE"] == "RECT_CLOSED":
                    diameter = 2 * edge_info["GEOM1"] * edge_info["GEOM2"] / (edge_info["GEOM1"] + edge_info["GEOM2"])
                else:
                    diameter = edge_info["GEOM1"]
                hydraulic_diameter_acc += diameter * edge_length
                length_acc += edge_length

            # ------------create the new linkage between old outfall node and new outfall node
            sw_line_geom1 = hydraulic_diameter_acc / length_acc
            # ---------------determine the length of the new linkage considering CFL condition
            sw_line_length = 10.0 * np.sqrt(9.81 * sw_line_geom1)
            sw_line_name_tmp = "_".join([sewer_prefix, conduit_prefix+str(sw_line_id)])
            sw_line_info[sw_line_name_tmp] = {}
            sw_line_info[sw_line_name_tmp]["FROM"] = ofnAux_name
            sw_line_info[sw_line_name_tmp]["TO"] = ofn
            sw_line_info[sw_line_name_tmp]["TYPE"] = "CIRCULAR"
            sw_line_info[sw_line_name_tmp]["GEOM1"] = sw_line_geom1
            sw_line_info[sw_line_name_tmp]["GEOM2"] = 0.0
            sw_line_info[sw_line_name_tmp]["N"] = 0.013
            sw_line_info[sw_line_name_tmp]["LENGTH"] = sw_line_length
            sw_line_id += 1

            link_graph.add_edge(ofnAux_name, ofn,
                                NAME=sw_line_name_tmp,
                                GEOM1=sw_line_geom1, GEOM2=0.0,
                                TYPE="CIRCULAR")
            # ------------create the newly inserted outfall node
            # ---------------we only need to change the coordiante of the old outfall 
            ofnNew_x, ofnNew_y = ofn_x, ofn_y + sw_line_length
            outfall_coords_info[ofn] = (ofnNew_x, ofnNew_y)

    # ------[4] add node and line information of culverts
    cl_line_id = 0
    cl_outfall_coords_info = {}
    cl_node_depth_info = {}
    cl_node_coords_info = {}
    cl_line_info = {}
    if culvert_gdf is not None:
        cl_graph, cl_node_geom, cl_gdf_exploded = getGraphFromGeoDataFrame(culvert_gdf, col_namelist=[link_type_col, link_geom1_col, link_geom2_col, ManningN_col, "THICKNESS", tauX_key, tauY_key, fromEle_key, toEle_key], sym="@", prefix=culvert_prefix)
        
        # ---------gather the namelist of outfalls whose degree is equal to 1
        cl_node_name = [outfall_prefix + "_" + n if cl_graph.degree[n] == 1 else n for n in cl_node_geom.keys()]

        # ---------add endpoints information by iterating over edges of culverts
        for ids, row in cl_gdf_exploded.iterrows():
            xs, ys = cl_node_geom[row["Pt1"]].coords[0]
            # ---------determine the name of nodes (for outfall node, a naming prefix should be added)
            if cl_graph.degree[row["Pt1"]] == 1:
                cl_sname = outfall_prefix + "_" + row["Pt1"]
            else:
                cl_sname = row["Pt1"]
            
            if cl_sname in cl_node_name:
                # ---------since an outfall node would be only searched for once during the iteration on edges,
                # ------------we can remove it to accelerate speed
                cl_node_name.remove(cl_sname)
                if outfall_prefix in cl_sname:
                    cl_outfall_coords_info[cl_sname] = [xs, ys]
                else:
                    cl_node_coords_info[cl_sname] = [xs, ys]
                node_ele[cl_sname] = row[fromEle_key]
                # ------------the depth of the node would be calculated as the averaging value of its owner edges 
                cl_node_depth_info[cl_sname] = []

            xe, ye = cl_node_geom[row["Pt2"]].coords[0]
            if cl_graph.degree[row["Pt2"]] == 1:
                cl_ename = outfall_prefix + "_" + row["Pt2"]
            else:
                cl_ename = row["Pt2"]
            if cl_ename in cl_node_name:
                cl_node_name.remove(cl_ename)
                if outfall_prefix in cl_ename:
                    cl_outfall_coords_info[cl_ename] = [xe, ye]
                else:
                    cl_node_coords_info[cl_ename] = [xe, ye]
                node_ele[cl_ename] = row[toEle_key]
                cl_node_depth_info[cl_ename] = []

            # cl_node_depth_info[cl_sname].append(row[link_geom1_col] + row["THICKNESS"])
            cl_node_depth_info[cl_ename].append(row[link_geom1_col])
            # cl_node_depth_info[cl_ename].append(row[link_geom1_col] + row["THICKNESS"])
            cl_node_depth_info[cl_ename].append(row[link_geom1_col])

            # ---------save edge information
            cl_line_name_tmp = "_".join([culvert_prefix, conduit_prefix+str(cl_line_id)])
            cl_line_info[cl_line_name_tmp] = {}
            cl_line_info[cl_line_name_tmp]["FROM"] = cl_sname
            cl_line_info[cl_line_name_tmp]["TO"] = cl_ename
            cl_line_info[cl_line_name_tmp]["TYPE"] = row[link_type_col]
            cl_line_info[cl_line_name_tmp]["GEOM1"] = row[link_geom1_col]
            cl_line_info[cl_line_name_tmp]["GEOM2"] = row[link_geom2_col]
            cl_line_info[cl_line_name_tmp]["N"] = row[ManningN_col]
            cl_line_info[cl_line_name_tmp]["LENGTH"] = np.sqrt((sw_start[0]-sw_end[0])**2+(sw_start[1]-sw_end[1])**2)
            cl_line_id += 1

            link_graph.add_edge(cl_sname, cl_ename, NAME=cl_line_name_tmp, 
                                GEOM1=cl_line_info[cl_line_name_tmp]["GEOM1"], 
                                GEOM2=cl_line_info[cl_line_name_tmp]["GEOM2"],
                                TYPE=cl_line_info[cl_line_name_tmp]["TYPE"],)
        
        for cln in cl_node_depth_info.keys():
            cl_node_depth_info[cln] = np.average(cl_node_depth_info[cln])
            
    # ------[5] adjust direction of links
    # ---------summarize node, link information
    swn_namelist = sorted(sw_node_coords_info.keys())
    otf_namelist = sorted(outfall_coords_info.keys())
    sw_rv_node_namelist = sorted(sw_rv_coords_info.keys())
    sw_inlet_namelist = sorted(sw_inlet_coords_info.keys())
    rvn_namelist = sorted(rv_node_coords_info.keys())
    rv_inlet_namelist = sorted(rv_inlet_coords_info.keys())
    cln_namelist = sorted(cl_node_coords_info.keys())
    cl_otf_namelist = sorted(cl_outfall_coords_info.keys())

    linkage_info = {**sw_line_info, **sw_inlet_linkage_info, **rv_line_info, **rv_inlet_linkage_info, **cl_line_info}
    '''
    node_info = {**sw_node_coords_info, **sw_rv_coords_info, **sw_inlet_coords_info, **rv_node_coords_info, **rv_inlet_coords_info, 
                    **outfall_coords_info, **cl_node_coords_info, **cl_outfall_coords_info}
    '''
    node_depth_info = {**sw_node_depth_info, **sw_rv_depth_info, **sw_inlet_info, **rv_node_depth_info, **rv_inlet_info, **outfall_depth_info, **cl_node_depth_info}
    
    # ---------re-estimate the depth of nodes based on geometry information of links
    node_depth_new = reset_swmm_NodeDepth(link_graph=link_graph, node_depth_info=node_depth_info, top_depth=sewer_depth,
                                            outfall_namelist=otf_namelist+cl_otf_namelist)
    
    # ---------save the new depth of nodes as json file for further use
    with open(os.path.join(save_prefix, "node_depth.json"), "w") as fp:
        json.dump(node_depth_new, fp, indent=4)

    # skip `outfall_coords_info` and `cl_outfall_coords_info` since the outfall node has no depth
    for pt in sw_node_coords_info.keys():
        #if not pt.startswith(outfallAux_prefix):
        sw_node_depth_info[pt] = node_depth_new[pt]["DEPTH"]
        node_ele[pt] = node_ele[pt] - node_depth_new[pt]["CHANGE"]
        # ------for outfallAux node, its invert elevation is inherited from its owner outfall node
        # ---------thus, we do not need to change its invert elevation
        # else:
        #     sw_node_depth_info[pt] = node_depth_new[pt]["DEPTH"]

    for pt in sw_rv_coords_info.keys():
        sw_rv_depth_info[pt] = node_depth_new[pt]["DEPTH"]
        node_ele[pt] = node_ele[pt] - node_depth_new[pt]["CHANGE"]

    for pt in sw_inlet_info.keys():
        sw_inlet_info[pt]["DEPTH"] = node_depth_new[pt]["DEPTH"]
        node_ele[pt] = node_ele[pt] - node_depth_new[pt]["CHANGE"]

    for pt in rv_inlet_info.keys():
        rv_inlet_info[pt]["DEPTH"] = node_depth_new[pt]["DEPTH"]
        node_ele[pt] = node_ele[pt] - node_depth_new[pt]["CHANGE"]

    for pt in rv_node_coords_info.keys():
        rv_node_depth_info[pt] = node_depth_new[pt]["DEPTH"]
        node_ele[pt] = node_ele[pt] - node_depth_new[pt]["CHANGE"]

    for pt in cl_node_coords_info.keys():
        cl_node_depth_info[pt] = node_depth_new[pt]["DEPTH"]
        node_ele[pt] = node_ele[pt] - node_depth_new[pt]["CHANGE"]

    # ---------re-set the FROM/TO information of links
    # skip `sw_inlet_linkage_info` and `rv_inlet_linkage_info` since their direction can not be changed
    for sw in sw_line_info.keys():
        from_tmp = sw_line_info[sw]["FROM"]
        to_tmp = sw_line_info[sw]["TO"]
        if outfall_prefix not in from_tmp and outfall_prefix not in to_tmp:
            if node_ele[from_tmp] < node_ele[to_tmp]:
                sw_line_info[sw]["FROM"] = to_tmp
                sw_line_info[sw]["TO"] = from_tmp
                link_graph.remove_edge(from_tmp, to_tmp)
                link_graph.add_edge(to_tmp, from_tmp, 
                                    NAME=sw,
                                    GEOM1=sw_line_info[sw]["GEOM1"],
                                    GEOM2=sw_line_info[sw]["GEOM2"],
                                    TYPE=sw_line_info[sw]["TYPE"])
    
    for rv in rv_line_info.keys():
        from_tmp = rv_line_info[rv]["FROM"]
        to_tmp = rv_line_info[rv]["TO"]
        if outfall_prefix not in from_tmp and outfall_prefix not in to_tmp:
            if node_ele[from_tmp] < node_ele[to_tmp]:
                rv_line_info[rv]["FROM"] = to_tmp
                rv_line_info[rv]["TO"] = from_tmp
                link_graph.remove_edge(from_tmp, to_tmp)
                link_graph.add_edge(to_tmp, from_tmp, 
                                    NAME=rv,
                                    GEOM1=rv_line_info[rv]["GEOM1"],
                                    GEOM2=rv_line_info[rv]["GEOM2"],
                                    TYPE=rv_line_info[rv]["TYPE"])
    
    for cl in cl_line_info.keys():
        from_tmp = cl_line_info[cl]["FROM"]
        to_tmp = cl_line_info[cl]["TO"]
        if outfall_prefix not in from_tmp and outfall_prefix not in to_tmp:
            if node_ele[from_tmp] < node_ele[to_tmp]:
                cl_line_info[cl]["FROM"] = to_tmp
                cl_line_info[cl]["TO"] = from_tmp
                link_graph.remove_edge(from_tmp, to_tmp)
                link_graph.add_edge(to_tmp, from_tmp, 
                                    NAME=cl,
                                    GEOM1=cl_line_info[cl]["GEOM1"],
                                    GEOM2=cl_line_info[cl]["GEOM2"],
                                    TYPE=cl_line_info[cl]["TYPE"])

    aux_outfall_namelist = [nname for nname in sw_node_coords_info.keys() if nname.startswith(outfallAux_prefix)]
    changed_info = setLinkDirection(link_graph=link_graph, outfall_namelist=otf_namelist+cl_otf_namelist,
                                    aux_outfall_namelist=aux_outfall_namelist,
                                    edge_name_attr="NAME", inlet_namelist=sw_inlet_namelist+rv_inlet_namelist,
                                    solver_config=solver_config)

    for lname in changed_info.keys():
        fname = changed_info[lname]["FROM"]
        tname = changed_info[lname]["TO"]
        if sewer_prefix in lname:
            if sw_inlet_linkage_prefix in lname:
                sw_inlet_linkage_info[lname]["FROM"] = fname
                sw_inlet_linkage_info[lname]["TO"] = tname
            else:
                sw_line_info[lname]["FROM"] = fname
                sw_line_info[lname]["TO"] = tname
        elif river_prefix in lname:
            if rv_inlet_linkage_prefix in lname:
                rv_inlet_linkage_info[lname]["FROM"] = fname
                rv_inlet_linkage_info[lname]["TO"] = tname
            else:
                rv_line_info[lname]["FROM"] = fname
                rv_line_info[lname]["TO"] = tname
        else:
            cl_line_info[lname]["FROM"] = fname
            cl_line_info[lname]["TO"] = tname
    
    # ------[6] write SWMM's .inp file
    start_date = simulation_settings["start_date"]
    start_time = simulation_settings["start_time"]
    end_date = simulation_settings["end_date"]
    end_time = simulation_settings["end_time"]
    report_step = simulation_settings["report_step"]
    
    save_prefix = str(save_prefix or '')
    with open(os.path.join(save_prefix, save_path), "w") as f:
        f.write("[TITLE]\n\n")
        # ---------options
        f.write("[OPTIONS]\n")
        f.write("FLOW_UNITS           CMS\n")           # ([C]ubic [M]eters per [S]econd)
        f.write("INFILTRATION         GREEN_AMPT\n")
        f.write("FLOW_ROUTING         DYNWAVE\n")
        f.write("START_DATE           {0}\n".format(start_date))
        f.write("START_TIME           {0}\n".format(start_time))
        f.write("REPORT_START_DATE    {0}\n".format(start_date))
        f.write("REPORT_START_TIME    {0}\n".format(start_time))
        f.write("END_DATE             {0}\n".format(end_date))
        f.write("END_TIME             {0}\n".format(end_time))
        f.write("SWEEP_START          1/1\n")
        f.write("SWEEP_END            12/31\n")
        f.write("DRY_DAYS             0\n")
        f.write("REPORT_STEP          {0}\n".format(report_step))
        # ------------WET_STEP is the time step length used to compute runoff from subcatchments
        # ---------------during periods of rainfall or when ponded water still remains on the surface.
        # ---------------The default is 0:05:00.
        f.write("WET_STEP             0:01:00\n")
        # ------------DRY_STEP is the time step length used for runoff computations (consisting essentially
        # ---------------of pollutant buildup) during periods when there is no rainfall and no ponded water.
        # ---------------The default is 1:00:00.
        f.write("DRY_STEP             0:05:00\n")
        f.write("ROUTING_STEP         5\n")
        f.write("ALLOW_PONDING        YES\n")
        # ------------INERTIAL_DAMPING indicates how the inertial terms in the Saint Venant 
        # ---------------momentum equation will be dropped under dynamic wave flow routing. [NONE/PARTIAL/FULL]
        f.write("INERTIAL_DAMPING     PARTIAL\n")
        # ------------VARIABLE_STEP is a safety factor applied to a variable time step related to CFL conditions
        f.write("VARIABLE_STEP        0.75\n")
        # ------------LENGTHENING_STEP s a time step, in seconds, used to lengthen conduits under dynamic wave routing
        f.write("LENGTHENING_STEP     0\n")
        # ------------SURCHARGE_METHOD selects which method will be used to handle surcharge conditions.
        # --------------- it appears that some of the instabilities that are intrinsic to the switching of the solution equations in EXTRAN 
        # --------------- when pressurization happens did not occur when SLOT was used.
        # --------------- ref to: Pachaly, R. L., Vasconcelos, J. G., Allasia, D. G., Tassi, R., & Bocchi, J. P. P. (2020). 
        # ------------------ Comparing SWMM 5.1 calculation alternatives to represent unsteady stormwater sewer flows. 
        # ------------------ Journal of Hydraulic Engineering, 146(7), 04020046.
        f.write("SURCHARGE_METHOD     SLOT\n")
        f.write("MIN_SURFAREA         0\n")
        f.write("NORMAL_FLOW_LIMITED  BOTH\n")
        f.write("SKIP_STEADY_STATE    NO\n")
        f.write("FORCE_MAIN_EQUATION  H-W\n")
        f.write("LINK_OFFSETS         DEPTH\n")
        f.write("MIN_SLOPE            0\n")

        f.write("MAX_TRIALS            16\n")
        f.write("THREADS            8\n\n")

        # ---------evaporation
        f.write("[EVAPORATION]\n")
        f.write(";;Type\tParameters\n")
        f.write(";;------------- ----------\n")
        f.write("CONSTANT\t0.0\n")
        f.write("DRY_ONLY\tNO\n\n")

        # ---------raingages
        f.write("[RAINGAGES]\n")
        f.write(";;\tRain\tTime\tSnow\tData\n")
        f.write(";;Name\tType\tIntrvl\tCatch\tSource\n")
        f.write(";;-------------- --------- ------ ------ ----------\n\n")

        # ---------junctions
        f.write("[JUNCTIONS]\n")
        f.write(";;\tInvert\tMax.\tInit.\tSurcharge\tPonded\n")
        f.write(";;Name\tElev.\tDepth\tDepth\tDepth\tArea\n")
        # ------------write the name, invert elevation, max depth, initial depth, surcharge depth, ponded area of junctions
        # ---------------sewer nodes
        NODEID_acc = 0
        swn_ele = {swn_name: node_ele[swn_name] for swn_name in swn_namelist}
        swn_content = "\n".join(["{0}\t{1}\t{2}\t0.0\t0.0\t{3}".format(swn_namelist[i], swn_ele[swn_namelist[i]], sw_node_depth_info[swn_namelist[i]], ponding_area) for i in range(0, len(swn_namelist))])
        if len(swn_namelist) > 0:
            f.write(swn_content + "\n")
        swn_id_ref = {swn_namelist[i]: i for i in range(0, len(swn_namelist))}
        NODEID_acc += len(swn_namelist)
        
        sw_rv_ele = {sw_rv_name: node_ele[sw_rv_name] for sw_rv_name in sw_rv_node_namelist}
        sw_rv_content = "\n".join(["{0}\t{1}\t{2}\t0.0\t0.0\t{3}".format(sw_rv_node_namelist[i], sw_rv_ele[sw_rv_node_namelist[i]], sw_rv_depth_info[sw_rv_node_namelist[i]], ponding_area) for i in range(0, len(sw_rv_node_namelist))])
        if len(sw_rv_node_namelist) > 0:
            f.write(sw_rv_content + "\n")
        sw_rv_id_ref = {sw_rv_node_namelist[i]: i+NODEID_acc for i in range(0, len(sw_rv_node_namelist))}
        NODEID_acc += len(sw_rv_node_namelist)

        # ------------------sewer inlet nodes
        sw_inlet_ele = {sw_inlet_name: node_ele[sw_inlet_name] for sw_inlet_name in sw_inlet_namelist}
        sw_inlet_content = "\n".join(["{0}\t{1}\t{2}\t0.0\t0.0\t{3}".format(sw_inlet_namelist[i], sw_inlet_ele[sw_inlet_namelist[i]], sw_inlet_info[sw_inlet_namelist[i]]["DEPTH"], ponding_area) for i in range(0, len(sw_inlet_namelist))])
        if len(sw_inlet_namelist) > 0:
            f.write(sw_inlet_content + "\n")
        sw_inlet_id_ref = {sw_inlet_namelist[i]: i+NODEID_acc for i in range(0, len(sw_inlet_namelist))}
        NODEID_acc += len(sw_inlet_namelist)

        # ---------------river nodes
        rvn_ele = {rvn_name: node_ele[rvn_name] for rvn_name in rvn_namelist}
        rvn_content = "\n".join(["{0}\t{1}\t{2}\t0.0\t0.0\t0.0".format(rvn_namelist[i], rvn_ele[rvn_namelist[i]], rv_node_depth_info[rvn_namelist[i]]) for i in range(0, len(rvn_namelist))])
        if len(rvn_namelist) > 0:
            f.write(rvn_content + "\n")
        rvn_id_ref = {rvn_namelist[i]: i+NODEID_acc for i in range(0, len(rvn_namelist))}
        NODEID_acc += len(rvn_namelist)

        # ------------------river inlet nodes
        rv_inlet_ele = {rv_inlet_name: node_ele[rv_inlet_name] for rv_inlet_name in rv_inlet_namelist}
        rv_inlet_content = "\n".join(["{0}\t{1}\t{2}\t0.0\t0.0\t{3}".format(rv_inlet_namelist[i], rv_inlet_ele[rv_inlet_namelist[i]], rv_inlet_info[rv_inlet_namelist[i]]["DEPTH"], ponding_area) for i in range(0, len(rv_inlet_namelist))])
        if len(rv_inlet_namelist) > 0:
            f.write(rv_inlet_content+"\n")
        rv_inlet_id_ref = {rv_inlet_namelist[i]: i+NODEID_acc for i in range(0, len(rv_inlet_namelist))}
        NODEID_acc += len(rv_inlet_namelist)

        # ---------------culvert nodes
        cln_ele = {cln_name: node_ele[cln_name] for cln_name in cln_namelist}
        cln_content = "\n".join(["{0}\t{1}\t{2}\t0.0\t0.0\t0.0".format(cln_name, cln_ele[cln_name], cl_node_depth_info[cln_name]) for cln_name in cln_namelist])
        if len(cln_namelist) > 0:
            f.write(cln_content + "\n")
        cln_id_ref = {cln_namelist[i]: i+NODEID_acc for i in range(0, len(cln_namelist))}
        NODEID_acc += len(cln_namelist)
        f.write("\n")

        # ---------outfalls
        f.write("[OUTFALLS]\n")
        f.write(";;Invert\tOutfall\tStage/Table\tTide\n")
        f.write(";;Name\tElev.\tType\tTime Series\tGate\n")
        
        # ------------write the invert elevation, outfall type (FREE/FIXED), Stage/Table TS ("\t"), Tide Gate ("NO") of outfalls
        otf_content = "\n".join(["{0}\t{1}\tFREE\t\t\tNO".format(otf_name, node_ele[otf_name]) if outfall_type_info[otf_name]=="FREE" 
                                    else "{0}\t{1}\tFIXED\t{2}\t\tNO".format(otf_name, node_ele[otf_name], node_ele[otf_name]) for otf_name in otf_namelist])
        if len(otf_namelist) > 0:
            f.write(otf_content + "\n")
        outfall_id_ref = {otf_namelist[i]: i+NODEID_acc for i in range(0, len(otf_namelist))}
        NODEID_acc += len(otf_namelist)
        
        cl_otf_content = "\n".join(["{0}\t{1}\tFIXED\t{2}\t\tNO".format(cl_otf_name, node_ele[cl_otf_name], node_ele[cl_otf_name]) for cl_otf_name in cl_otf_namelist])
        if len(cl_otf_namelist) > 0:
            f.write(cl_otf_content + "\n")
        cl_otf_id_ref = {cl_otf_namelist[i]: i+NODEID_acc for i in range(0, len(cl_otf_namelist))}
        NODEID_acc += len(cl_otf_namelist)
        f.write("\n")
        
        # ---------conduits
        f.write("[CONDUITS]\n")
        f.write(";;\tInlet\tOutlet\tManning\tInlet\tOutlet\tInit.\tMax.\n")
        f.write(";;Name\tNode\tNode\tLength\tN\tOffset\tOffset\tFlow\tFlow\n")
        f.write(";;-------------- ---------------- ---------------- ---------- ---------- ---------- ---------- ---------- ----------\n")
        # ------------write the name, inlet node's name, outlet node's name, length, Manning's N, inlet offset (0), outlet offset (0), initial flow (0), max flow (0) of conduits
        # ---------------sewer
        sw_namelist = sorted(sw_line_info.keys())
        sw_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t{4}\t0.0\t0.0\t0.0\t0.0".format(sw, sw_line_info[sw]["FROM"], sw_line_info[sw]["TO"], sw_line_info[sw]["LENGTH"], sw_line_info[sw]["N"]) for sw in sw_namelist])
        if len(sw_namelist) > 0:
            f.write(sw_content + "\n")

        sw_linkage_namelist = sorted(sw_inlet_linkage_info.keys())
        sw_linkage_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t{4}\t0.0\t0.0\t0.0\t0.0".format(sw, sw_inlet_linkage_info[sw]["FROM"], sw_inlet_linkage_info[sw]["TO"], sw_inlet_linkage_info[sw]["LENGTH"], sw_inlet_linkage_info[sw]["N"]) for sw in sw_linkage_namelist])
        if len(sw_linkage_namelist) > 0:
            f.write(sw_linkage_content + "\n")
        # ---------------river
        rv_namelist = sorted(rv_line_info.keys())
        rv_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t{4}\t0.0\t0.0\t0.0\t0.0".format(rv, rv_line_info[rv]["FROM"], rv_line_info[rv]["TO"], rv_line_info[rv]["LENGTH"], rv_line_info[rv]["N"]) for rv in rv_namelist])
        if len(rv_namelist) > 0:
            f.write(rv_content + "\n")

        rv_linkage_namelist = sorted(rv_inlet_linkage_info.keys())
        rv_linkage_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t{4}\t0.0\t0.0\t0.0\t0.0".format(rv, rv_inlet_linkage_info[rv]["FROM"], rv_inlet_linkage_info[rv]["TO"], rv_inlet_linkage_info[rv]["LENGTH"], rv_inlet_linkage_info[rv]["N"]) for rv in rv_linkage_namelist])
        if len(rv_linkage_namelist) > 0:
            f.write(rv_linkage_content+"\n")

        cl_namelist = sorted(cl_line_info.keys())
        cl_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t{4}\t0.0\t0.0\t0.0\t0.0".format(cl, cl_line_info[cl]["FROM"], cl_line_info[cl]["TO"], cl_line_info[cl]["LENGTH"], cl_line_info[cl]["N"]) for cl in cl_namelist])
        if len(cl_namelist) > 0:
            f.write(cl_content + "\n\n")

        # ---------X-axis sections of conduits
        f.write("[XSECTIONS]\n")
        f.write(";;Link\tShape\tGeom1\tGeom2\tGeom3\tGeom4\tBarrels\n")
        f.write(";;-------------- ------------ ---------------- ---------- ---------- ---------- ----------\n")
        # ------------write the link's name, shape (CIRCULAR/RECT_OPEN/RECT_CLOSED/IRREGULAR), Geom1-4, barrels (1) of X-axis sections.
        # ---------------Note that when shape is `RECT_OPEN/RECT_CLOSED`, Geom1 and Geom2 would be the depth and width, respectively.
        # ---------------Note that when shape is `IRREGULAR`, Geom1 would be `TransectXX`.
        # ---------------sewer
        sw_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t0.0\t0.0\t1".format(sw, sw_line_info[sw]["TYPE"], sw_line_info[sw]["GEOM1"], sw_line_info[sw]["GEOM2"]) for sw in sw_namelist])
        if len(sw_namelist) > 0:
            f.write(sw_content + "\n")

        sw_linkage_namelist = sorted(sw_inlet_linkage_info.keys())
        sw_linkage_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t0.0\t0.0\t1".format(sw, sw_inlet_linkage_info[sw]["TYPE"], sw_inlet_linkage_info[sw]["GEOM1"], sw_inlet_linkage_info[sw]["GEOM2"]) for sw in sw_linkage_namelist])
        if len(sw_linkage_content) > 0:
            f.write(sw_linkage_content + "\n")
        # ---------------river
        rv_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t0.0\t0.0\t1".format(rv, rv_line_info[rv]["TYPE"], rv_line_info[rv]["GEOM1"], rv_line_info[rv]["GEOM2"]) for rv in rv_namelist])
        if len(rv_namelist) > 0:
            f.write(rv_content + "\n")

        rv_linkage_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t0.0\t0.0\t1".format(rv, rv_inlet_linkage_info[rv]["TYPE"], rv_inlet_linkage_info[rv]["GEOM1"], rv_inlet_linkage_info[rv]["GEOM2"]) for rv in rv_linkage_namelist])
        if len(rv_linkage_namelist) > 0:
            f.write(rv_linkage_content+"\n")

        cl_content = "\n".join(["{0}\t{1}\t{2}\t{3}\t0.0\t0.0\t1".format(cl, cl_line_info[cl]["TYPE"], cl_line_info[cl]["GEOM1"], cl_line_info[cl]["GEOM2"]) for cl in cl_namelist])
        if len(cl_namelist) > 0:
            f.write(cl_content + "\n\n")

        # ---------transection definitions of conduits
        if trans_info is not None:
            f.write("[TRANSECTS]\n")
            for trans_name in trans_info.keys():
                num_station = len(trans_info[trans_name]["X"])
                # ------------write the Manning’s n of left, right overbank and main portion of channel.
                f.write("NC\t{0}\t{1}\t{2}\n".format(trans_info[trans_name]["ManningN_left"], trans_info[trans_name]["ManningN_right"], trans_info[trans_name]["ManningN_main"]))
                # ------------write the name, number of stations, station position which ends the left, right overbank portion of the channel.
                f.write("X1\t{0}\t{1}\t{2}\t{3}\t0\t0\t0\t0\t0\t0\n".format(trans_name, num_station, trans_info[trans_name]["X"][0], trans_info[trans_name]["X"][-1]))
                f.write("GR\t" + "\t".join(["{0}\t{1}".format(trans_info[trans_name]["ELEVATION"][i], trans_info[trans_name]["X"][i]) for i in range(0, num_station)]) + "\n")

        # ---------inflow definitions of conduits
        f.write("\n[INFLOWS]\n")
        f.write(";;\tParam\tUnits\tScale\tBaseline\tBaseline\n")
        f.write(";;Node\tParameter\tTime Series\tType\tFactor\tFactor\tValue\tPattern\n")
        f.write(";;-------------- ---------------- ---------------- -------- -------- -------- -------- --------\n\n")
        # ------------write the node's name, Parameter (Flow), TS's name, Parameter Type (FLOW), units factor (1.0), scale factor (1.0), baseline value (0.0), Baseline pattern ("\t") of junctions.
        
        # ---------TS definitions
        f.write("[TIMESERIES]\n")
        f.write(";;Name\tDate\tTime\tValue\n")
        f.write(";;-------------- ---------- ---------- ----------\n\n")
        # ------------write the TS's name, date (%-d/%-m/%Y), time (%M:%S), value of TSs.
        # ---------------Note that `\n` should added between different TSs.

        # ---------report settings
        f.write("[REPORT]\n")
        f.write("INPUT      YES\nCONTROLS   NO\nSUBCATCHMENTS ALL\nNODES ALL\nLINKS ALL\n\n")

        # ---------map settings
        f.write("[MAP]\n\n")

        # ---------coordinates of nodes
        f.write("[COORDINATES]\n")
        f.write(";;Node\tX-Coord\tY-Coord\n")
        f.write(";;-------------- ---------------- ----------------\n")
        # ------------write the name, X, Y of nodes.
        swn_content = "\n".join(["{0}\t{1}\t{2}".format(swn_name, sw_node_coords_info[swn_name][0], sw_node_coords_info[swn_name][1]) for swn_name in swn_namelist])
        if len(swn_namelist) > 0:
            f.write(swn_content + "\n")

        sw_rv_content = "\n".join(["{0}\t{1}\t{2}".format(sw_rv_name, sw_rv_coords_info[sw_rv_name][0], sw_rv_coords_info[sw_rv_name][1]) for sw_rv_name in sw_rv_node_namelist])
        if len(sw_rv_node_namelist) > 0:
            f.write(sw_rv_content + "\n")

        sw_inlet_content = "\n".join(["{0}\t{1}\t{2}".format(sw_inlet_name, sw_inlet_coords_info[sw_inlet_name][0], sw_inlet_coords_info[sw_inlet_name][1]) for sw_inlet_name in sw_inlet_namelist])
        if len(sw_inlet_namelist):
            f.write(sw_inlet_content + "\n")

        rvn_content = "\n".join(["{0}\t{1}\t{2}".format(rvn_name, rv_node_coords_info[rvn_name][0], rv_node_coords_info[rvn_name][1]) for rvn_name in rvn_namelist])
        if len(rvn_namelist) > 0:
            f.write(rvn_content + "\n")

        cln_content = "\n".join(["{0}\t{1}\t{2}".format(cln_name, cl_node_coords_info[cln_name][0], cl_node_coords_info[cln_name][1]) for cln_name in cln_namelist])
        if len(cln_namelist) > 0:
            f.write(cln_content + "\n")

        cl_otf_ele = {cl_otf_name: node_ele[cl_otf_name] for cl_otf_name in cl_otf_namelist}
        cl_otf_content = "\n".join(["{0}\t{1}\t{2}".format(cl_otf_name, cl_outfall_coords_info[cl_otf_name][0], cl_outfall_coords_info[cl_otf_name][1]) for cl_otf_name in cl_otf_namelist])
        if len(cl_otf_namelist) > 0:
            f.write(cl_otf_content + "\n")
        
        rv_inlet_content = "\n".join(["{0}\t{1}\t{2}".format(rv_inlet_name, rv_inlet_coords_info[rv_inlet_name][0], rv_inlet_coords_info[rv_inlet_name][1]) for rv_inlet_name in rv_inlet_namelist])
        if len(rv_inlet_namelist) > 0:
            f.write(rv_inlet_content + "\n")

        otf_ele = {otf_name: node_ele[otf_name] for otf_name in otf_namelist}
        otf_content = "\n".join(["{0}\t{1}\t{2}".format(otf_name, outfall_coords_info[otf_name][0], outfall_coords_info[otf_name][1]) for otf_name in otf_namelist])
        if len(otf_namelist) > 0:
            f.write(otf_content + "\n\n")

        f.write("[SYMBOLS]\n")
        f.write(";;Gage\tX-Coord\tY-Coord\n")
        f.write(";;-------------- ---------------- ----------------\n")
    
    # ------[5] save the generated objects into .shp files for further checking
    if shp_saved:
        with fiona.Env(OSR_WKT_FORMAT="WKT2_2018"):
            # ---------junctions
            junction_namelist = swn_namelist + sw_rv_node_namelist + rvn_namelist + otf_namelist + cln_namelist + cl_otf_namelist
            junction_coords_info = {**sw_node_coords_info, **sw_rv_coords_info, **rv_node_coords_info, **outfall_coords_info, **cl_node_coords_info, **cl_outfall_coords_info}
            junction_ele_info = {**swn_ele, **sw_rv_ele, **rvn_ele, **otf_ele, **cln_ele, **cl_otf_ele}
            junction_depth_info = {**sw_node_depth_info, **sw_rv_depth_info, **rv_node_depth_info, **outfall_depth_info, **cl_node_depth_info}
            junction_geom = [Point(junction_coords_info[jn][0], junction_coords_info[jn][1]) for jn in junction_namelist]
            junction_ele = [junction_ele_info[jn] for jn in junction_namelist]
            junction_depth = [junction_depth_info[jn] for jn in junction_namelist]
            junction_out_gdf = gpd.GeoDataFrame({"NAME": junction_namelist, "InvertEle": junction_ele, "DEPTH": junction_depth, "geometry": junction_geom})
            junction_out_gdf = junction_out_gdf.set_crs("epsg:{0}".format(save_epsg))
            junction_out_gdf.to_file(os.path.join(save_prefix, "swmm_junction.gpkg"))
            # ---------inlets
            inlet_namelist = sw_inlet_namelist + rv_inlet_namelist
            inlet_coords_info = {**sw_inlet_coords_info, **rv_inlet_coords_info}
            inlet_ele_info = {**sw_inlet_ele, **rv_inlet_ele}
            inlet_geom = [Point(inlet_coords_info[jn][0], inlet_coords_info[jn][1]) for jn in inlet_namelist]
            inlet_ele = [inlet_ele_info[jn] for jn in inlet_namelist]
            inlet_depth = [sw_inlet_info[jn]["DEPTH"] for jn in sw_inlet_namelist] + [rv_inlet_info[jn]["DEPTH"] for jn in rv_inlet_namelist]
            inlet_out_gdf = gpd.GeoDataFrame({"NAME": inlet_namelist, "InvertEle": inlet_ele, "DEPTH": inlet_depth, "geometry": inlet_geom})
            inlet_out_gdf = inlet_out_gdf.set_crs("epsg:{0}".format(save_epsg))
            inlet_out_gdf.to_file(os.path.join(save_prefix, "swmm_inlet.gpkg"))
            # ---------linkage
            node_coords_info = {**junction_coords_info, **inlet_coords_info}
            linkage_namelist = sw_namelist + sw_linkage_namelist + rv_namelist + rv_linkage_namelist + cl_namelist
            linkage_from = [linkage_info[rn]["FROM"] for rn in linkage_namelist]
            linkage_to = [linkage_info[rn]["TO"] for rn in linkage_namelist]
            linkage_type = [linkage_info[rn]["TYPE"] for rn in linkage_namelist]
            linkage_geom1 = [linkage_info[rn]["GEOM1"] for rn in linkage_namelist]
            linkage_geom2 = [linkage_info[rn]["GEOM2"] for rn in linkage_namelist]
            linkage_geom = [LineString([node_coords_info[linkage_info[rn]["FROM"]], node_coords_info[linkage_info[rn]["TO"]]]) for rn in linkage_namelist]
            linkage_out_gdf = gpd.GeoDataFrame({"NAME": linkage_namelist, "FROM": linkage_from, "TO": linkage_to, 
                                                "TYPE": linkage_type, "GEOM1": linkage_geom1, "GEOM2": linkage_geom2, "geometry": linkage_geom})
            linkage_out_gdf = linkage_out_gdf.set_crs("epsg:{0}".format(save_epsg))
            linkage_out_gdf.to_file(os.path.join(save_prefix, "swmm_link.gpkg"))
    
    if id_saved:
        node_id_info = {**swn_id_ref, **sw_rv_id_ref, **sw_inlet_id_ref, **rvn_id_ref, **rv_inlet_id_ref, **outfall_id_ref, **cln_id_ref, **cl_otf_id_ref}
        with open(os.path.join(save_prefix, "node_id.json"), "w") as f:
            json.dump(node_id_info, f, indent=4)
# ********* input formatter for SWMM (END) *********


if __name__ == "__main__":
    pass
