from typing import Union, List, Dict, Tuple
import os
import shutil
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.ops as sops
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint, Polygon
from functools import reduce
import pulp as lp
import networkx as nx
import swmmio
from swmmio.utils import modify_model

PULP_CBC_RANDOM_SEED = "116"

# ********* miscellaneous functions (START) *********
def testSubSeg_subproc(row: gpd.GeoSeries, seg_base: LineString, almost_equal_threshold=1e-3):
    line = row["geometry"]
    pt_s = Point(line.coords[0])
    pt_s_distance = pt_s.distance(seg_base)

    pt_e = Point(line.coords[1])
    pt_e_distance = pt_e.distance(seg_base)

    if pt_s_distance + pt_e_distance < almost_equal_threshold:
        return True
    else:
        return False
    

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
        line_segs.to_file(save_path, engine="pyogrio")
    
    return line_segs


# ---Split the drainage lines at intersections with other types of drainage lines
def splitLineByOtherType(row: gpd.GeoSeries, geom_whole: Union[LineString, MultiLineString], node_whole: MultiPoint):
    geom_target = row["geometry"]
    geom_other = geom_whole.difference(geom_target)
    geom_intersect = geom_target.intersection(geom_other)
    if not geom_intersect.is_empty:
        if isinstance(geom_intersect, Point):
            pt_split = geom_intersect
        else:
            pt_split = MultiPoint([g for g in geom_intersect.geoms if isinstance(g, Point)])
        pt_intersect = pt_split.intersection(node_whole)
        geom_target = sops.split(geom_target, pt_intersect)
        geom_target = MultiLineString([g for g in geom_target.geoms if isinstance(g, LineString)])
    
    return geom_target


def insertInlet(row: gpd.GeoSeries, inlet_spacing=50.0, inlet_min_spacing=10.0, mask_geoms=None):
    line_geom = row["geometry"]
    line_length = line_geom.length

    num_seg = max(int(line_length / inlet_spacing), 1)

    if num_seg > 1:
        seg_pt_loc = np.arange(0, line_length, inlet_spacing)
        if seg_pt_loc[-1] > line_length - inlet_min_spacing and seg_pt_loc[-1] < line_length:
            seg_pt_loc = seg_pt_loc[:-1]

        if seg_pt_loc[-1] < line_length:
            seg_pt_list = [line_geom.interpolate(d) for d in seg_pt_loc] + [Point(line_geom.coords[-1])]
        else:
            seg_pt_list = [line_geom.interpolate(d) for d in seg_pt_loc]
        
        # ---------exclude the points which are located in the mask
        if mask_geoms is not None:
            seg_pt_list = [pt for pt in seg_pt_list if not pt.intersects(mask_geoms)]

        split_geom = MultiLineString([LineString([seg_pt_list[i], seg_pt_list[i+1]]) for i in range(0, len(seg_pt_list)-1)])
    else:
        split_geom = line_geom

    return split_geom


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


def reset_swmm_NodeDepth(link_graph: nx.DiGraph, node_depth_info: Dict, top_depth=0.6, outfall_keywords=None, outfall_namelist=None):
    """Estimate the direction of river/sewer network based on DEM using 0/1 Programming.

    Parameters
    ----------

    link_graph: networkx.Graph
        Undirected graph which indicates the structure of the river/sewer network.
    node_info: dict
        Information of nodal depth.
    top_depth: list
        Depth between the surface and the top of underground links.
    
    """
    sym = "@"

    edge_attr_ref = {}
    pt_link_ref = {}
    for fname, tname, dta in link_graph.edges(data=True):
        vname = fname+sym+tname
        edge_attr_ref[vname] = {}
        edge_attr_ref[vname]["GEOM1"] = dta["GEOM1"]
        edge_attr_ref[vname]["GEOM2"] = dta["GEOM2"]
        edge_attr_ref[vname]["TYPE"] = dta["TYPE"]
        if fname in pt_link_ref.keys():
            pt_link_ref[fname].append(vname)
        else:
            pt_link_ref[fname] = [vname]
        if tname in pt_link_ref.keys():
            pt_link_ref[tname].append(vname)
        else:
            pt_link_ref[tname] = [vname]

    if outfall_namelist is None:
        if outfall_keywords is None:
            outfall_namelist = [nname for (nname, v) in link_graph.out_degree() if v==0]
            print("Outfall nodes would be selected as nodes whose out_degree is zero")
        else:
            outfall_namelist = [n for n in node_depth_info.keys() if outfall_keywords in n]

    node_depth_new = {}
    nonoutfall_pt = set(pt_link_ref.keys()).difference(set(outfall_namelist))
    for pt in nonoutfall_pt:
        node_depth_ini = node_depth_info[pt]
        if isinstance(node_depth_ini, dict):
            node_depth_ini = node_depth_ini["DEPTH"]
        link_crown_depth_all = [edge_attr_ref[vname]["GEOM1"] + top_depth if edge_attr_ref[vname]["TYPE"] in ["CIRCULAR", "RECT_CLOSED"] else 0.0 for vname in pt_link_ref[pt]]
        link_crown_depth = max(link_crown_depth_all)
        node_depth_new[pt] = {}
        node_depth_new[pt]["DEPTH"]= max(node_depth_ini, link_crown_depth)
        node_depth_new[pt]["MinDepth"] = max([d - 0.5 * top_depth if d > 0 else 0.0 for d in link_crown_depth_all])
        # print(pt, node_depth_ini, link_crown_depth, node_depth_new[pt]["DEPTH"])
        node_depth_new[pt]["CHANGE"] = node_depth_new[pt]["DEPTH"] - node_depth_ini
    
    for pt in outfall_namelist:
        node_depth_new[pt] = {}
        node_depth_new[pt]["DEPTH"]= 0.0
        node_depth_new[pt]["MinDepth"] = 0.0
        node_depth_new[pt]["CHANGE"] = 0.0

    return node_depth_new
    

# ---Artificial Spatial Discretization for drainage lines
# ---ref to: Pachaly, R. L., Vasconcelos, J. G., Allasia, D. G., Tassi, R., & Bocchi, J. P. P. (2020). 
# ------Comparing SWMM 5.1 calculation alternatives to represent unsteady stormwater sewer flows. Journal of Hydraulic Engineering, 146(7), 04020046.
def adaptiveSplitLine(row, alpha=0.02, mask_geoms=None):
    line_type = row["XSECTION"]
    line_geom = row["geometry"]

    if line_type == "CIRCULAR":
        diameter = row["GEOM1"]
    elif line_type == "RECT_CLOSED":
        diameter = 2 * row["GEOM1"] * row["GEOM2"] / (row["GEOM1"] + row["GEOM2"])
    else:
        diameter = row["GEOM1"]

    num_seg = max(int(alpha * line_geom.length / diameter), 1)

    if num_seg > 1:
        dx = line_geom.length / num_seg
        seg_pt_list = [line_geom.interpolate(d) for d in np.arange(0, line_geom.length, dx)][:-1] + [Point(line_geom.coords[-1])]

        # ---------exclude the points which are located in the mask
        if mask_geoms is not None:
            seg_pt_list = [pt for pt in seg_pt_list if not pt.intersects(mask_geoms)]

        split_geom = MultiLineString([LineString([seg_pt_list[i], seg_pt_list[i+1]]) for i in range(0, len(seg_pt_list)-1)])
    else:
        split_geom = line_geom

    return split_geom


def split_drainage_line_with_inlet(input_sewer_path: str, output_sewer_path: str, output_inlet_path: str, inlet_spacing=50, simplified_tolerance=0.1, geom_scale_factor=1.0, nonInlet_node_path=None, mask_boundary_path=None, inlet_min_spacing=10.0, inlet_depth=0.6):
    """split the sewer lines with road inlets placed by specified spacings.

    Parameters
    ----------

    input_sewer_path: str
        Input path of the sewer network.
    output_sewer_path: str
        Output path of the discretized sewer network.
    output_inlet_path: str
        Output path of the generated inlets.
    inlet_spacing: float
        Spacing of the road inlets.
        The default is `50`.
    simplified_tolerance: float
        Tolerance distance used in Douglas-Peucker simplification of input sewer.
        Ref to: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html.
        The default is `0.1`.
    geom_scale_factor: float
        Scale factor of the geometry size of input sewer.
        The target unit should be [m].
        The default is `1.0`.
    nonInlet_node_path: str
        Input path of the non-inlet nodes which would be excluded from the generated inlets.
        The default is `None`.
    mask_boundary_path: str
        Input path of the mask boundary which would be used to exclude the generated inlets, i.e., no inlets would be placed in this region.
        The default is `None`.
    inlet_min_spacing: float
        Minimum spacing of the road inlets.
    inlet_depth: float
        Initial estimated depth of the road inlets.
    
    """
    sym = "@"
    attribute_col = ["GEOM1", "GEOM2", "XSECTION", "ROUGHNESS"]
    line_gdf = gpd.read_file(input_sewer_path)

    if "GEOM2" not in line_gdf.columns:
        line_gdf["GEOM2"] = 0.0

    if "XSECTION" not in line_gdf.columns:
        line_gdf["XSECTION"] = "CIRCULAR"

    if "ROUGHNESS" not in line_gdf.columns:
        line_gdf["ROUGHNESS"] = 0.013

    line_gdf["GEOM1"] = line_gdf["GEOM1"] * geom_scale_factor
    line_gdf["GEOM2"] = line_gdf["GEOM2"] * geom_scale_factor

    if mask_boundary_path is not None:
        mask_boundary = gpd.read_file(mask_boundary_path)
        mask_geoms = mask_boundary.unary_union
    else:
        mask_geoms = None

    # ------locate the points where some type of line shared with other types of lines
    
    # ------dissolve the lines with same attributes
    line_gdf_ds = line_gdf.dissolve(by=attribute_col, as_index=False)
    # ------merge dissolved lines
    line_gdf_ds["geometry"] = line_gdf_ds.apply(lambda row: sops.linemerge(row["geometry"]) if isinstance(row["geometry"], MultiLineString) else row["geometry"], axis=1)
    geom_whole = sops.unary_union([g for g in line_gdf_ds["geometry"]])

    pt_cr_whole = line_gdf_ds["geometry"].apply(lambda g: [cr for cr in g.coords] if isinstance(g, LineString) else [cr for sg in g.geoms for cr in sg.coords])
    pt_cr_whole = np.concatenate(pt_cr_whole, axis=0)

    pt_cr_whole = np.unique(pt_cr_whole, axis=0)

    pt_whole = MultiPoint(pt_cr_whole)
    line_gdf_ds["geometry"] = line_gdf_ds.apply(lambda row: splitLineByOtherType(row, geom_whole, pt_whole), axis=1)
    # ------do simplifications
    line_gdf_exploded = explodeLineSeg(ini_gdf=line_gdf_ds, simplified=True, tolerance=simplified_tolerance)

    # ------we may need to add functions to combine close intermediate points in the coming version
    print(f"Add rainwater inlets by the spacing of {inlet_spacing} m.")
    line_gdf_exploded["geometry"] = line_gdf_exploded.apply(lambda row: insertInlet(row, inlet_spacing=inlet_spacing, inlet_min_spacing=inlet_min_spacing, mask_geoms=mask_geoms), axis=1)
    line_gdf_exploded = explodeLineSeg(ini_gdf=line_gdf_exploded, simplified=False)

    _, line_node_geom, _ = getGraphFromGeoDataFrame(line_gdf_exploded, col_namelist=attribute_col, simplified=False, sym=sym, prefix="IL")
    inlet_name = []
    inlet_geom = []
    for pt_name in line_node_geom.keys():
        inlet_name.append(pt_name)
        inlet_geom.append(line_node_geom[pt_name])
    inlet_gdf = gpd.GeoDataFrame({"geometry": inlet_geom, "name": inlet_name}, crs=line_gdf.crs)
    if nonInlet_node_path is not None:
        node_known = gpd.read_file(nonInlet_node_path)
        inlet_gdf = gpd.overlay(inlet_gdf, node_known, how="difference")
    # ------set the (dummy) attributes for the inlet points
    inlet_gdf["DEPTH"] = inlet_depth
    inlet_gdf["XSECTION"] = "CIRCULAR"
    inlet_gdf["GEOM1"] = 0.3
    inlet_gdf["GEOM2"] = 0.0
    inlet_gdf["ROUGHNESS"] = 0.013
    inlet_gdf.to_file(output_inlet_path, engine="pyogrio")

    # ------add artificial spatial discretization (ASD) for drainage lines
    line_gdf_exploded["geometry"] = line_gdf_exploded.apply(lambda row: adaptiveSplitLine(row, alpha=0.02, mask_geoms=mask_geoms), axis=1)
    line_gdf_exploded = explodeLineSeg(ini_gdf=line_gdf_exploded, simplified=False)

    line_gdf_exploded.drop(columns=[col for col in line_gdf_exploded.columns if col not in attribute_col + ["geometry"]], inplace=True)
    line_gdf_exploded.to_file(output_sewer_path, engine="pyogrio")
# ********* miscellaneous functions (END) *********


# ********* sewer reconstructor (START) *********
def setLinkDirection(link_graph: nx.DiGraph, outfall_namelist: list, edge_name_attr=None, inlet_namelist=None, aux_outfall_namelist=None, fixed_dir_info=None, solver_config=None):
    """Estimate the direction of river/sewer network based on DEM using 0/1 Programming.

    Parameters
    ----------

    link_graph: networkx.Graph
        Undirected graph which indicates the structure of the river/sewer network.
    outfall_namelist: list
        Namelist of outfalls which serves as the starting points of inflows.
    edge_name_attr: float
        Name of the edge attribute in `link_graph` which indicates the name of the edge.
        If no name is given, the edge would be named as the join of the name of start and end points.
        The default is `None`.
    inlet_namelist: list
        Namelist of inlet nodes which serves as inflow points which would be excluded from bridge points during cycling analysis.
        The default is `None`.
    aux_outfall_namelist: list
        Namelist of auxiliary outfalls which also serves as outfalls.
    fixed_dir_info: dict
        Dictionary which stores the `InletNode` and `OutletNode` of edges which are fixed.
        The default is `None`.
    solver_config: dict
        Input dict which describe the solver settings of link direction derivation, for example:
            {
                "solver": "cbc",
                "license": "",
                "path": "",
                "name": "test",
                "save_prefix": "",
            }
    
    """
    if fixed_dir_info is not None and edge_name_attr is None:
        raise ValueError("The name of edges must be given if fixed_namelist is not None.")
    else:
        if fixed_dir_info is None:
            fixed_dir_info = {}

    if aux_outfall_namelist is None:
        aux_outfall_namelist = []

    sym = "@"

    node_namelist = [n for n in link_graph.nodes]
    link_graph_undirected = link_graph.to_undirected()

    sol_old = {}
    var_name_ref = {}
    # -----store the name of links which contain the current point
    pt_link_ref = {}
    fixed_dir_sol = {}
    for fname, tname, dta in link_graph.edges(data=True):
        vname = fname+sym+tname
        
        sol_old[vname] = 1
        if edge_name_attr is not None:
            lname = dta[edge_name_attr]
        else:
            lname = fname+tname

        var_name_ref[vname] = lname
        if lname in fixed_dir_info:
            if fname == fixed_dir_info[lname]["InletNode"]:
                fixed_dir_sol[vname] = 1
            else:
                fixed_dir_sol[vname] = 0

        if fname in pt_link_ref.keys():
            pt_link_ref[fname].append(vname)
        else:
            pt_link_ref[fname] = [vname]

        if tname in pt_link_ref.keys():
            pt_link_ref[tname].append(vname)
        else:
            pt_link_ref[tname] = [vname]
        
    var_namelist = sorted(sol_old.keys())

    non_outfall_namelist = sorted([n for n in node_namelist if n not in outfall_namelist])
    
    # ------find edges linked to cycles/loops in the graph
    # ---------ref to: https://stackoverflow.com/questions/31034730/graph-analysis-identify-loop-paths
    cyc_info_ref = {}

    # cycle_list = list(nx.cycle_basis(link_graph_undirected))
    cycle_list = list(nx.simple_cycles(link_graph_undirected))
    cycle_cut_OuterIntersect = []
    cycle_cut_OuterIntersect_flag = []
    cycleInnerEdge = []
    cycleInnerEdge_flag = []

    for cyc in cycle_list:
        cycOuterEdge = []
        cycOuterEdge_flag = []
        num_pt = len(cyc)
        for pt in cyc:
            outer_linked_edge = [vname for vname in pt_link_ref[pt] if not all([pt in cyc for pt in vname.split(sym)])]
            if len(outer_linked_edge) > 0:
                outer_linked_edge_flag = [1 if vname.split(sym)[0]==pt else 0 for vname in outer_linked_edge]
                cycOuterEdge += outer_linked_edge
                cycOuterEdge_flag += outer_linked_edge_flag
        # --------record the number of outer bridges of the cycle where the point lie
        for pt in cyc:
            cyc_info_ref[pt] = len(cycOuterEdge)
        # --------record the edges of the cycle
        cycle_cut_OuterIntersect.append(cycOuterEdge)
        cycle_cut_OuterIntersect_flag.append(cycOuterEdge_flag)
        
        cycleInnerEdge_tmp = []
        cycleInnerEdge_flag_tmp = []
        for i in range(0, num_pt-1):
            fname = cyc[i]
            tname = cyc[i+1]
            if fname+sym+tname in pt_link_ref[fname]:
                cycleInnerEdge_tmp.append(fname+sym+tname)
                cycleInnerEdge_flag_tmp.append(1)
            else:
                cycleInnerEdge_tmp.append(tname+sym+fname)
                cycleInnerEdge_flag_tmp.append(0)
        if cyc[-1]+sym+cyc[0] in pt_link_ref[cyc[-1]]:
            cycleInnerEdge_tmp.append(cyc[-1]+sym+cyc[0])
            cycleInnerEdge_flag_tmp.append(1)
        else:
            cycleInnerEdge_tmp.append(cyc[0]+sym+cyc[-1])
            cycleInnerEdge_flag_tmp.append(0)
        
        cycleInnerEdge.append(cycleInnerEdge_tmp)
        cycleInnerEdge_flag.append(cycleInnerEdge_flag_tmp)
        
    # ------construct the 0/1 linear programming model for the solution of direction
    lp_prob = lp.LpProblem("LinkDirection", lp.LpMinimize)

    var_aux = [vname+"Aux" for vname in var_namelist]
    var_rec = [vname+"Rec" for vname in var_namelist]
    d_all = var_aux + var_rec
    lp_var = lp.LpVariable.dicts("var", d_all, cat="Integer", lowBound=0, upBound=1)

    # ---------define the objective function
    lp_prob += lp.lpSum([lp_var[v] for v in var_aux])

    # ------------[1] auxiliary variable constraints
    for vname in var_namelist:
        lp_prob += lp_var[vname+"Rec"] + lp_var[vname+"Aux"] >= sol_old[vname]
        lp_prob += lp_var[vname+"Rec"] - lp_var[vname+"Aux"] <= sol_old[vname]
    
    # ------------[2] in_degree constraints for outfall nodes
    for node_name in outfall_namelist:
        link_list_tmp = pt_link_ref[node_name]
        for vname in link_list_tmp:
            fname, tname = vname.split(sym)
            if fname == node_name:
                lp_prob += 1 - lp_var[vname+"Rec"] == 1
                other_node_name = tname
            else:
                lp_prob += lp_var[vname+"Rec"] == 1
                other_node_name = fname
            # ---------------if the other node is an auxliary outfall node,
            # ------------------it is also an outfall node previously
            # ---------------------we should also add the out_degree constraints for the other node
            if other_node_name in aux_outfall_namelist:
                link_list_other_tmp = pt_link_ref[other_node_name]
                for vname_other in link_list_other_tmp:
                    fname_other, tname_other = vname_other.split(sym)
                    if fname_other == other_node_name and tname_other != node_name:
                        lp_prob += 1 - lp_var[vname_other+"Rec"] == 1
                    elif tname_other == other_node_name and fname_other != node_name:
                        lp_prob += lp_var[vname_other+"Rec"] == 1
    
    # ------------[3] fixed direction constraints
    for vname in fixed_dir_sol.keys():
        lp_prob += lp_var[vname+"Rec"] == fixed_dir_sol[vname]
    
    # ------------[4] out_degree constraints for non-outfall nodes
    for node_name in non_outfall_namelist:
        link_list_tmp = pt_link_ref[node_name]
        node_flag_tmp = [1 if vname.split(sym)[0]==node_name else 0 for vname in link_list_tmp]
        lp_prob += lp.lpSum([lp_var[link_list_tmp[i]+"Rec"] if node_flag_tmp[i] == 1 else 1 - lp_var[link_list_tmp[i]+"Rec"] for i in range(0, len(link_list_tmp))]) >= 1
        # ------------[5] in-degree constraints for non-source nodes which are not contained in the cycles whose number of outer bridges is greater than 1 (i.e., non-island nodes)
        # ---------------assure the point are not a source
        if len(link_list_tmp) > 1:
            if node_name in cyc_info_ref.keys():
                if cyc_info_ref[node_name] > 1:
                    lp_prob += lp.lpSum([1-lp_var[link_list_tmp[i]+"Rec"] if node_flag_tmp[i] == 1 else lp_var[link_list_tmp[i]+"Rec"] for i in range(0, len(link_list_tmp))]) >= 1
            else:
                lp_prob += lp.lpSum([1-lp_var[link_list_tmp[i]+"Rec"] if node_flag_tmp[i] == 1 else lp_var[link_list_tmp[i]+"Rec"] for i in range(0, len(link_list_tmp))]) >= 1
    
    num_cyc_set = len(cycle_cut_OuterIntersect)
    
    for sid in range(0, num_cyc_set):
        # ------------[6] out_degree constraints for cycles/loops in the graph
        cyc_edge = cycle_cut_OuterIntersect[sid]
        cyc_edge_flag = cycle_cut_OuterIntersect_flag[sid]
        if len(cyc_edge) > 0:
            lp_prob += lp.lpSum([lp_var[cyc_edge[i]+"Rec"] if cyc_edge_flag[i] == 1 else 1 - lp_var[cyc_edge[i]+"Rec"] for i in range(0, len(cyc_edge))]) >= 1
        # ------------[7] no recirculation for cycles/loops in the graph
        cyc_edge = cycleInnerEdge[sid]
        print(f"****** Cycle {sid} ******", [e for e in cyc_edge])
        cyc_edge_flag = cycleInnerEdge_flag[sid]
        lp_prob += lp.lpSum([lp_var[cyc_edge[i]+"Rec"] if cyc_edge_flag[i] == 1 else 1 - lp_var[cyc_edge[i]+"Rec"] for i in range(0, len(cyc_edge))]) <= len(cyc_edge)-1
        lp_prob += lp.lpSum([lp_var[cyc_edge[i]+"Rec"] if cyc_edge_flag[i] == 1 else 1 - lp_var[cyc_edge[i]+"Rec"] for i in range(0, len(cyc_edge))]) >= 1
        
    # ---------solve the LP problem
    if solver_config is None or solver_config["solver"] == "cbc":
        sl_used = lp.PULP_CBC_CMD(options=["randomC", PULP_CBC_RANDOM_SEED])
    elif solver_config["solver"] == "gurobi":
        from pulp import GUROBI_CMD
        os.environ["GRB_LICENSE_FILE"] = solver_config["license"]
        sl_used = GUROBI_CMD(path=solver_config["path"])
    else:
        print("Solver not supported!")
        print(solver_config)
        print("Using default solver: PULP_CBC_CMD")
        sl_used = lp.PULP_CBC_CMD(options=["randomC", PULP_CBC_RANDOM_SEED])

    if solver_config is not None and "name" in solver_config:
        lp_name = solver_config["name"]
        if "save_prefix" in solver_config:
            lp_path = os.path.join(solver_config["save_prefix"], f"{lp_name}_dir.lp")
        else:
            lp_path = f"{lp_name}_dir.lp"
        lp_prob.writeLP(lp_path)

    lp_prob.solve(sl_used)

    changed_info = {}
    if lp_prob.status == 1:
        for vname in var_namelist:
            fname_old, tname_old = vname.split(sym)
            if lp_var[vname+"Rec"].varValue == 0:
                lname = var_name_ref[vname]
                changed_info[lname] = {}
                changed_info[lname]["FROM"] = tname_old
                changed_info[lname]["TO"] = fname_old
    
    return changed_info


def reconstructNodeDepth_twoNodes(link_graph: nx.DiGraph, slope_min=None, slope_min_ref=None, slope_min_factor=1.0, slope_max=2.0, slope_max_ref=None, fixed_namelist=None, fixed_ele_info=None, delta_max=10.0, weighted=None, depth_limited=False, outfall_relaxed=False, solver_config=None):
    """Reconstruct the invert elevation and depth of nodes of river/sewer network with given minimum slope constraints and known values using Linear Programming.

    Parameters
    ----------

    link_graph: networkx.DiGraph
        Directed graph which indicates the structure of the river/sewer network.
        Its edge attributes should include `length` (length of the edge).
        Its node attributes should include `invert_ele` (invert elevation of the node) and `crest_ele` (crest elevation of the node).
    inlet_namelist: list
        Namelist of inlet nodes which serves as inflow points.
    outfall_namelist: list
        Namelist of outfall nodes which serves as outflow points of inflows.
    slope_min : float
        Minimum allowable pipe slope [m/m] used if `slope_min_ref` is not provided.
    slope_min_ref : dict or None
        Dictionary mapping pipe diameter [mm] to minimum allowable slope; overrides `slope_min`.
    slope_min_factor : float
        Factor used to scale the slope_min_ref values.
        The default is `1.0`.
    slope_max : float
        Maximum allowable pipe slope [m/m] used if `slope_max_ref` is not provided.
    slope_max_ref : dict or None
        Dictionary mapping pipe diameter [mm] to maximum allowable slope; overrides `slope_max`.
    fixed_namelist: list
        Namelist of nodes with fixed invert elevation values.
        The default is `None`.
    fixed_ele_info: dict
        Dictionary mapping node names to fixed invert elevation values.
        The default is `None`.
    delta_max: float
        Maximum variation of elevation value allowed after the optimization. 
        The default is `10.0`.
    weighted: str
        Weighted methods for penalty function of each nodes in the network which can be chosen from: 
            `harmonic` which stands for weighting by the harmonic average of the length of edges linked to the node.
            `arithmetic` which stands for weighting by the arithmetic average of the length of edges linked to the node.
            `None` which stands for equally weighting.
        The default is `None`.
    depth_limited: bool
        Flag for depth limitation to constrain the depth of nodes.
        The default is `False`.
    outfall_relaxed : bool, optional
        Whether to allow elevation adjustments at outfall nodes if necessary for feasibility.
    solver_config : dict or None, optional
        Input dict which describe the solver settings of link direction derivation, for example:
            {
                "solver": "cbc",
                "license": "",
                "path": "",
                "name": "test"
            }
    
    """
    if slope_min is None and slope_min_ref is None:
        raise ValueError("Either slope_min or slope_min_ref should be given.")
    elif slope_min is not None and slope_min_ref is None:
        print("slope_min is given, slope_min is set to be {0} * {1}.".format(slope_min, slope_min_factor))
        slope_min_ref_used = {0: slope_min_factor * slope_min}
    else:
        slope_min_ref_used = {k: v * slope_min_factor for k, v in slope_min_ref.items()}

    if slope_max_ref is None:
        slope_max_ref_used = {0: slope_max}
        print("slope_max is set to be {0}.".format(slope_max))
    else:
        slope_max_ref_used = slope_max_ref

    # ------set the slope constraints for different types of links
    d_used_smin = []
    i_min_used = []
    for k, v in slope_min_ref_used.items():
        d_used_smin.append(k)
        i_min_used.append(v)
    d_used_smin = np.array(d_used_smin)
    d_max_smin = np.max(d_used_smin)
    i_min_used = np.array(i_min_used)
    i_min_smin = np.min(i_min_used)

    d_used_smax = []
    i_max_used = []
    for k, v in slope_max_ref_used.items():
        d_used_smax.append(k)
        i_max_used.append(v)
    d_used_smax = np.array(d_used_smax)
    d_max_smax = np.max(d_used_smax)
    i_max_used = np.array(i_max_used)
    i_min_smax = np.min(i_max_used)

    def get_i_min(d, d_g, i_min_g, d_g_max, i_min_min):
        if d >= d_g_max:
            return i_min_min
        else:
            return np.interp(d, d_g, i_min_g)

    # ------only name of the fixed-elevation nodes is given
    if fixed_namelist is not None:
        fixed_invEle_info = {n: link_graph.nodes[n]["invert_ele"] for n in fixed_namelist}
    else:
        fixed_namelist = []
        fixed_invEle_info = {}

    if fixed_ele_info is not None:
        fixed_invEle_info = {k: v for k, v in fixed_ele_info.items()}
        fixed_namelist = list(fixed_invEle_info.keys())

    # ------change the link_graph attributes according to `fixed_invEle_info`
    for nname in fixed_invEle_info.keys():
        depth_tmp = link_graph.nodes[nname]["crest_ele"] - link_graph.nodes[nname]["invert_ele"]
        link_graph.nodes[nname]["invert_ele"] = fixed_invEle_info[nname]
        link_graph.nodes[nname]["crest_ele"] = link_graph.nodes[nname]["invert_ele"] + depth_tmp

    edge_length_info = nx.get_edge_attributes(link_graph, "length")
    
    # ------get the elevation of nodes to be reconstructed
    # ---------get the interior nodes
    node_depth_ref = {}
    for nname in link_graph.nodes():
        node_depth_ref[nname] = {}
        node_depth_ref[nname]["invert_ele"] = link_graph.nodes[nname]["invert_ele"]
        node_depth_ref[nname]["crest_ele"] = link_graph.nodes[nname]["crest_ele"]
    
    # ------construct the linear programming model for the solution of depth
    lp_prob = lp.LpProblem("DepthReconstruction", lp.LpMinimize)

    # ---------define the decision variable
    var_namelist = sorted(node_depth_ref.keys())
    var_aux = [vname+"Aux" for vname in var_namelist]
    var_rec = [vname+"Rec" for vname in var_namelist]
    d_all = var_aux + var_rec
    lp_var = lp.LpVariable.dicts("var", d_all, cat="Continuous")

    # ---------define the objective function
    if outfall_relaxed:
        penalty_factor = 1e6
    else:
        penalty_factor = 1.0

    if weighted == "harmonic":
        node_length_acc = {n: 0.0 for n in link_graph.nodes}
        node_length_count = {n: 0.0 for n in link_graph.nodes}
        for u, v, dta in link_graph.edges(data=True):
            node_length_acc[u] += 1.0 / dta["length"]
            node_length_count[u] += 1
            node_length_acc[v] += 1.0 / dta["length"]
            node_length_count[v] += 1
        for n in node_length_acc.keys():
            if node_length_acc[n] == 0:
                print("Warning: node {0} has no linked edges!".format(n))
        node_length_avg = {n: node_length_count[n] / node_length_acc[n] for n in node_length_acc.keys()}
        normalizer = sum(node_length_avg.values())
        lp_prob += lp.lpSum([lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer * penalty_factor if vname in fixed_namelist \
                              else lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer for vname in var_namelist])
    elif weighted == "arithmetic":
        node_length_acc = {n: 0.0 for n in link_graph.nodes}
        node_length_count = {n: 0.0 for n in link_graph.nodes}
        for u, v, dta in link_graph.edges(data=True):
            node_length_acc[u] += dta["length"]
            node_length_count[u] += 1
            node_length_acc[v] += dta["length"]
            node_length_count[v] += 1
        node_length_avg = {n: node_length_acc[n] / node_length_count[n] for n in node_length_acc.keys()}
        normalizer = sum(node_length_avg.values())
        lp_prob += lp.lpSum([lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer * penalty_factor if vname in fixed_namelist \
                             else lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer for vname in var_namelist])
    else:
        lp_prob += lp.lpSum([lp_var[vname+"Aux"] * penalty_factor if vname in fixed_namelist else lp_var[vname+"Aux"] for vname in var_namelist])

    # ---------define the constraints
    # ------------[1] the slope of sewer/river should be larger than slope_min
    for edges in link_graph.edges():
        i_sname, i_ename = edges
        elapsed_length = edge_length_info[(i_sname, i_ename)]

        d_tmp = link_graph.get_edge_data(i_sname, i_ename)["D"]
        slope_min_tmp = get_i_min(d_tmp, d_used_smin, i_min_used, d_max_smin, i_min_smin)
        slope_max_tmp = get_i_min(d_tmp, d_used_smax, i_max_used, d_max_smax, i_min_smax)
        lp_prob += lp.lpSum([lp_var[i_sname+"Rec"], lp_var[i_ename+"Rec"] * (-1)]) / elapsed_length >= slope_min_tmp
        lp_prob += lp.lpSum([lp_var[i_sname+"Rec"], lp_var[i_ename+"Rec"] * (-1)]) / elapsed_length <= slope_max_tmp
                        
    # ------------[2] the adjusted invert elevation of point should not be higher than ground elevation
    if depth_limited:
        for vname in node_depth_ref.keys():
            d_min = link_graph.nodes[vname]["depth_min"]
            lp_prob += lp_var[vname+"Rec"] - link_graph.nodes[vname]["crest_ele"] <= -d_min
    
    # ------------[3] auxiliary variable constraints
    for vname in node_depth_ref.keys():
        lp_prob += lp_var[vname+"Rec"] + lp_var[vname+"Aux"] >= link_graph.nodes[vname]["invert_ele"]
        lp_prob += lp_var[vname+"Rec"] - lp_var[vname+"Aux"] <= link_graph.nodes[vname]["invert_ele"]
    
    # ------------[5] fixed invert elevation constraints
    if not outfall_relaxed:
        for vname in fixed_invEle_info.keys():
            lp_prob += lp_var[vname+"Rec"] == fixed_invEle_info[vname]

    # ---------solve the LP problem
    if solver_config is None or solver_config["solver"] == "cbc":
        sl_used = lp.PULP_CBC_CMD(options=["randomC", PULP_CBC_RANDOM_SEED])
    elif solver_config["solver"] == "gurobi":
        from pulp import GUROBI_CMD
        os.environ["GRB_LICENSE_FILE"] = solver_config["license"]
        sl_used = GUROBI_CMD(path=solver_config["path"])
    else:
        print("Solver not supported!")
        print(solver_config)
        print("Using default solver: PULP_CBC_CMD")
        sl_used = lp.PULP_CBC_CMD(options=["randomC", PULP_CBC_RANDOM_SEED])

    if solver_config is not None and "name" in solver_config:
        lp_name = solver_config["name"]
        if "save_prefix" in solver_config:
            lp_path = os.path.join(solver_config["save_prefix"], f"{lp_name}_depth.lp")
        else:
            lp_path = f"{lp_name}_depth.lp"
        lp_prob.writeLP(lp_path)
    
    lp_prob.solve(sl_used)

    depth_adjusted_info = {}
    if lp_prob.status == 1:
        for vname in node_depth_ref.keys():
            depth_adjusted_info[vname] = {}
            invert_ele_ini = node_depth_ref[vname]["invert_ele"]
            invert_ele_rec = lp_var[vname+"Rec"].varValue
            adjust_val = invert_ele_ini - invert_ele_rec

            depth_ini = node_depth_ref[vname]["crest_ele"] - invert_ele_ini
            depth_rec = node_depth_ref[vname]["crest_ele"] - invert_ele_rec
            # ---------if the adjusted depth is too high, 
            # ------------or if the adjusted depth is negative (i.e., adjusted crest elevation exceeds given reference),
            # ------------we only modify the invert elevation and keep depth unchanged (crest elevation would be changed in this case)
            if np.abs(adjust_val) >= delta_max or depth_rec <= 0:
                depth_adjusted_info[vname]["invert_ele"] = round(invert_ele_rec, 6)
                depth_adjusted_info[vname]["depth"] = round(depth_ini, 6)
            # ---------if the node is on the ground, 
            # ------------we directly modify the invert elevation and keep its depth to zero
            elif depth_ini < 1e-6:
                depth_adjusted_info[vname]["invert_ele"] = round(invert_ele_rec, 6)
                depth_adjusted_info[vname]["depth"] = 0.0
            # ---------Otherwise we adjust the depth and invert elevation simultaneously and
            # ------------crest elevation would be unchanged in this case, which is our desired solution :)
            else:
                depth_adjusted_info[vname]["invert_ele"] = round(invert_ele_rec, 6)
                depth_adjusted_info[vname]["depth"] = round(depth_rec, 6)
    else:
        print("Linear Programming failed !!!")
    
    return depth_adjusted_info


def reconstructNodeDepth(link_graph: nx.DiGraph, inlet_namelist: list, outfall_namelist: list, slope_min=None, slope_min_ref=None, slope_min_factor=1.0, slope_max_ref=None, slope_max=2.0, fixed_namelist=None, delta_max=10.0, num_pt_min=3, group_length=50.0, stride_length=10.0, weighted=None, depth_limited=False, outfall_relaxed=False, solver_config=None):
    """Reconstruct the invert elevation and depth of nodes of river/sewer network with given minimum slope constraints and known values using Linear Programming.

    Parameters
    ----------

    link_graph: networkx.DiGraph
        Directed graph which indicates the structure of the river/sewer network.
        Its edge attributes should include `length` (length of the edge).
        Its node attributes should include `invert_ele` (invert elevation of the node) and `crest_ele` (crest elevation of the node).
    inlet_namelist: list
        Namelist of inlet nodes which serves as inflow points.
    outfall_namelist: list
        Namelist of outfall nodes which serves as outflow points of inflows.
    slope_min : float
        Minimum allowable pipe slope [m/m] used if `slope_min_ref` is not provided.
    slope_min_ref : dict or None
        Dictionary mapping pipe diameter [mm] to minimum allowable slope; overrides `slope_min`.
    slope_min_factor : float
        Factor used to scale the slope_min_ref values.
        The default is `1.0`.
    slope_max : float
        Maximum allowable pipe slope [m/m] used if `slope_max_ref` is not provided.
    slope_max_ref : dict or None
        Dictionary mapping pipe diameter [mm] to maximum allowable slope; overrides `slope_max`.
    fixed_namelist: list
        Namelist of nodes with fixed invert elevation values.
        The default is `None`.
    delta_max : float, optional
        Maximum allowable elevation change between adjusted and original node elevations.
        The default is `10.0`.
    num_pt_min : int, optional
        Minimum number of consecutive nodes to use when estimating slope.
    group_length : float, optional
        Minimum length [m] of node groups when using the path-based adjustment algorithm.
    stride_length : float, optional
        Stride length [m] to shift the node group window during processing.
    weighted: str
        Weighted methods for penalty function of each nodes in the network which can be chosen from: 
            `harmonic` which stands for weighting by the harmonic average of the length of edges linked to the node.
            `arithmetic` which stands for weighting by the arithmetic average of the length of edges linked to the node.
            `None` which stands for equally weighting.
        The default is `None`.
    depth_limited: bool
        Flag for depth limitation to constrain the depth of nodes.
        The default is `False`.
    outfall_relaxed : bool, optional
        Whether to allow elevation adjustments at outfall nodes if necessary for feasibility.
    solver_config : dict or None, optional
        Input dict which describe the solver settings of link direction derivation, for example:
            {
                "solver": "cbc",
                "license": "",
                "path": "",
                "name": "test"
            }
    
    """

    if slope_min is None and slope_min_ref is None:
        raise ValueError("Either slope_min or slope_min_ref should be given.")
    elif slope_min is not None and slope_min_ref is None:
        print("slope_min is given, slope_min is set to be {0} * {1}.".format(slope_min, slope_min_factor))
        slope_min_ref_used = {0: slope_min_factor * slope_min}
    else:
        slope_min_ref_used = {k: v * slope_min_factor for k, v in slope_min_ref.items()}

    if slope_max_ref is None:
        slope_max_ref_used = {0: slope_max}
        print("slope_max is set to be {0}.".format(slope_max))
    else:
        slope_max_ref_used = slope_max_ref

    # ------set the slope constraints for different types of links
    d_used = []
    i_min_used = []
    for k, v in slope_min_ref_used.items():
        d_used.append(k)
        i_min_used.append(v)
    d_used = np.array(d_used)
    d_max = np.max(d_used)
    i_min_used = np.array(i_min_used)
    i_min = np.min(i_min_used)

    d_used_smax = []
    i_max_used = []
    for k, v in slope_max_ref_used.items():
        d_used_smax.append(k)
        i_max_used.append(v)
    d_used_smax = np.array(d_used_smax)
    d_max_smax = np.max(d_used_smax)
    i_max_used = np.array(i_max_used)
    i_min_smax = np.min(i_max_used)

    def get_i_min(d, d_g, i_min_g, d_g_max, i_min_min):
        if d >= d_g_max:
            return i_min_min
        else:
            return np.interp(d, d_g, i_min_g)

    if fixed_namelist is not None:
        fixed_invEle_info = {n: link_graph.nodes[n]["invert_ele"] for n in fixed_namelist}
    else:
        fixed_namelist = []
        fixed_invEle_info = {}

    link_graph_rev = link_graph.reverse(copy=True)

    edge_length_info = nx.get_edge_attributes(link_graph, "length")
    
    # ------select a successor node of inlet nodes as the start point if:
    # ---------1) its ancestor (inlet node) has no further ancestor
    # ---------2) all of its other ancestors have no further ancestors
    source_pair = []
    in_degree_ref = {nname: v for (nname, v) in link_graph.in_degree()}
    startNode_namelist = []
    for iname in inlet_namelist:
        if in_degree_ref[iname] == 0:
            inlet_neigh = [n for n in link_graph.neighbors(iname)]
            for sname in inlet_neigh:
                # ------------store the information of links between successor nodes and their ancestors (inlet nodes) 
                source_pair.append([iname, sname])
                if all([in_degree_ref[n]==0 for n in link_graph_rev.neighbors(sname)]):
                    startNode_namelist.append(sname)
    
    startNode_namelist = np.unique(startNode_namelist)
    # ------get the elevation of nodes to be reconstructed
    # ---------get the interior nodes
    node_depth_ref = {}
    path_ref = {}
    for sname in startNode_namelist:
        path_ref[sname] = {}
        for oname in outfall_namelist:
            if nx.has_path(link_graph, sname, oname) and sname != oname:
                path_ref[sname][oname] = [path for path in nx.all_simple_paths(link_graph, sname, oname)]
                node_list_tmp = np.unique(reduce(lambda x, y: x+y, path_ref[sname][oname]))
                for n in node_list_tmp:
                    # ---------we fix the invert elevation of outfalls and nodes whose full_depth is zero
                    if n not in node_depth_ref.keys():
                        node_depth_ref[n] = {}
                        node_depth_ref[n]["invert_ele"] = link_graph.nodes[n]["invert_ele"]
                        node_depth_ref[n]["crest_ele"] = link_graph.nodes[n]["crest_ele"]
    # ---------get the nodes at the boundary
    for spair in source_pair:
        i_sname, i_ename = spair
        if i_sname not in node_depth_ref:
            node_depth_ref[i_sname] = {}
            node_depth_ref[i_sname]["invert_ele"] = link_graph.nodes[i_sname]["invert_ele"]
            node_depth_ref[i_sname]["crest_ele"] = link_graph.nodes[i_sname]["crest_ele"]
        if i_ename not in node_depth_ref:
            node_depth_ref[i_ename] = {}
            node_depth_ref[i_ename]["invert_ele"] = link_graph.nodes[i_ename]["invert_ele"]
            node_depth_ref[i_ename]["crest_ele"] = link_graph.nodes[i_ename]["crest_ele"]
    
    # ------construct the linear programming model for the solution of depth
    lp_prob = lp.LpProblem("DepthReconstruction", lp.LpMinimize)

    # ---------define the decision variable
    var_namelist = sorted(node_depth_ref.keys())
    var_aux = [vname+"Aux" for vname in var_namelist]
    var_rec = [vname+"Rec" for vname in var_namelist]
    d_all = var_aux + var_rec
    lp_var = lp.LpVariable.dicts("var", d_all, cat="Continuous")

    # ---------define the objective function
    if outfall_relaxed:
        penalty_factor = 1e6
    else:
        penalty_factor = 1.0

    if weighted == "harmonic":
        node_length_acc = {n: 0.0 for n in link_graph.nodes}
        node_length_count = {n: 0.0 for n in link_graph.nodes}
        for u, v, dta in link_graph.edges(data=True):
            node_length_acc[u] += 1.0 / dta["length"]
            node_length_count[u] += 1
            node_length_acc[v] += 1.0 / dta["length"]
            node_length_count[v] += 1
        for n in node_length_acc.keys():
            if node_length_acc[n] == 0:
                print("Warning: node {0} has no linked edges!".format(n))
        node_length_avg = {n: node_length_count[n] / node_length_acc[n] for n in node_length_acc.keys()}
        normalizer = sum(node_length_avg.values())
        lp_prob += lp.lpSum([lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer * penalty_factor if vname in fixed_namelist \
                              else lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer for vname in var_namelist])
    elif weighted == "arithmetic":
        node_length_acc = {n: 0.0 for n in link_graph.nodes}
        node_length_count = {n: 0.0 for n in link_graph.nodes}
        for u, v, dta in link_graph.edges(data=True):
            node_length_acc[u] += dta["length"]
            node_length_count[u] += 1
            node_length_acc[v] += dta["length"]
            node_length_count[v] += 1
        node_length_avg = {n: node_length_acc[n] / node_length_count[n] for n in node_length_acc.keys()}
        normalizer = sum(node_length_avg.values())
        lp_prob += lp.lpSum([lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer * penalty_factor if vname in fixed_namelist \
                             else lp_var[vname+"Aux"] * node_length_avg[vname] / normalizer for vname in var_namelist])
    else:
        lp_prob += lp.lpSum([lp_var[vname+"Aux"] * penalty_factor if vname in fixed_namelist else lp_var[vname+"Aux"] for vname in var_namelist])

    # ---------define the constraints
    # ------------[1] the slope of sewer/river should be larger than slope_min
    for spair in source_pair:
        i_sname, i_ename = spair
        elapsed_length = edge_length_info[(i_sname, i_ename)]

        d_tmp = link_graph.get_edge_data(i_sname, i_ename)["D"]
        slope_min_tmp = get_i_min(d_tmp, d_used, i_min_used, d_max, i_min)
        slope_max_tmp = get_i_min(d_tmp, d_used_smax, i_max_used, d_max_smax, i_min_smax)
        lp_prob += lp.lpSum([lp_var[i_sname+"Rec"], lp_var[i_ename+"Rec"] * (-1)]) / elapsed_length >= slope_min_tmp
        lp_prob += lp.lpSum([lp_var[i_sname+"Rec"], lp_var[i_ename+"Rec"] * (-1)]) / elapsed_length <= slope_max_tmp

    for sname in startNode_namelist:
        for oname in path_ref[sname].keys():
            # ---------------add the montonicity constraints based on Linear Regression Estimator
            for path_tmp in path_ref[sname][oname]:
                i_sid = 0
                i_eid = 1
                num_path_pt = len(path_tmp)
                while (i_eid <= num_path_pt - 1):
                    num_pt_min_tmp = max(min(num_path_pt, num_pt_min), 2)
                    i_sname = path_tmp[i_sid]
                    i_ename = path_tmp[i_eid]

                    elapsed_pt = [i_sname, i_ename]
                    elapsed_length_acc = edge_length_info[(i_sname, i_ename)]
                    elapsed_length = [0, elapsed_length_acc]
                    
                    # ---------------initialize the flag which indicates whether start point is to be changed
                    i_sid_next = i_eid
                    if elapsed_length_acc >= stride_length:
                        spt_change_flag = True
                    else:
                        spt_change_flag = False

                    while (elapsed_length_acc < group_length) and (len(elapsed_pt) < num_pt_min_tmp):
                        # ---------------check whether the ending cursor has reached the ending of the path
                        if i_eid == num_path_pt - 1:
                            break
                        i_eid += 1
                        i_ename_new = path_tmp[i_eid]
                        elapsed_length_tmp = edge_length_info[(i_ename, i_ename_new)]
                        elapsed_length_acc += elapsed_length_tmp
                        # ---------------once the accumulated length exceeds `stride_length`
                        # ------------------switch the flag to True and records the next starting point
                        if (elapsed_length_acc >= stride_length) and (not spt_change_flag):
                            i_sid_next = i_eid

                        elapsed_length.append(elapsed_length_acc)
                        elapsed_pt.append(i_ename_new)
                        i_ename = i_ename_new

                    num_pt = len(elapsed_pt)

                    if num_pt >= 2:
                        if any([path_tmp[i_sid+i] in node_depth_ref.keys() for i in range(0, num_pt)]):
                            x_sum = sum(elapsed_length)
                            denominator = num_pt * sum([x**2 for x in elapsed_length]) - x_sum ** 2
                            
                            slope_min_tmp = [get_i_min(link_graph.get_edge_data(elapsed_pt[i], elapsed_pt[i+1])["D"], d_used, i_min_used, d_max, i_min) for i in range(0, num_pt-1)]
                            slope_min_tmp = np.median(slope_min_tmp)
                            slope_max_tmp = [get_i_min(link_graph.get_edge_data(elapsed_pt[i], elapsed_pt[i+1])["D"], d_used_smax, i_max_used, d_max_smax, i_min_smax) for i in range(0, num_pt-1)]
                            slope_max_tmp = np.median(slope_max_tmp)
                            lp_prob += lp.lpSum([(num_pt*elapsed_length[i] - x_sum) * lp_var[elapsed_pt[i]+"Rec"] for i in range(0, num_pt)]) / denominator <= -slope_min_tmp
                            lp_prob += lp.lpSum([(num_pt*elapsed_length[i] - x_sum) * lp_var[elapsed_pt[i]+"Rec"] for i in range(0, num_pt)]) / denominator >= -slope_max
                            # ------additional constraints for ending points of the group
                            # ---------in case that the start point is lower than the ending point
                            i_sname = elapsed_pt[0]
                            i_ename = elapsed_pt[-1]
                            s_ele = lp_var[i_sname+"Rec"]
                            e_ele = lp_var[i_ename+"Rec"]
                            lp_prob += (s_ele - e_ele) / elapsed_length_acc >= slope_min_tmp
                            lp_prob += (s_ele - e_ele) / elapsed_length_acc <= slope_max
                            
                        i_sid = i_sid_next
                        i_eid = i_sid_next + 1
                        
    if depth_limited:
        for vname in node_depth_ref.keys():
            d_min = link_graph.nodes[vname]["depth_min"]
            lp_prob += lp_var[vname+"Rec"] - link_graph.nodes[vname]["crest_ele"] <= -d_min
    
    # ------------[3] auxiliary variable constraints
    for vname in node_depth_ref.keys():
        lp_prob += lp_var[vname+"Rec"] + lp_var[vname+"Aux"] >= link_graph.nodes[vname]["invert_ele"]
        lp_prob += lp_var[vname+"Rec"] - lp_var[vname+"Aux"] <= link_graph.nodes[vname]["invert_ele"]
    
    # ------------[5] fixed invert elevation constraints
    if not outfall_relaxed:
        for vname in fixed_invEle_info.keys():
            lp_prob += lp_var[vname+"Rec"] == fixed_invEle_info[vname]

    # ---------solve the LP problem
    if solver_config is None or solver_config["solver"] == "cbc":
        sl_used = lp.PULP_CBC_CMD(options=["randomC", PULP_CBC_RANDOM_SEED])
    elif solver_config["solver"] == "gurobi":
        from pulp import GUROBI_CMD
        os.environ["GRB_LICENSE_FILE"] = solver_config["license"]
        sl_used = GUROBI_CMD(path=solver_config["path"])
    else:
        print("Solver not supported!")
        print(solver_config)
        print("Using default solver: PULP_CBC_CMD")
        sl_used = lp.PULP_CBC_CMD(options=["randomC", PULP_CBC_RANDOM_SEED])

    if solver_config is not None and "name" in solver_config:
        lp_name = solver_config["name"]
        if "save_prefix" in solver_config:
            lp_path = os.path.join(solver_config["save_prefix"], f"{lp_name}_depth.lp")
        else:
            lp_path = f"{lp_name}_depth.lp"

    lp_prob.solve(sl_used)

    depth_adjusted_info = {}
    if lp_prob.status == 1:
        for vname in node_depth_ref.keys():
            depth_adjusted_info[vname] = {}
            invert_ele_ini = node_depth_ref[vname]["invert_ele"]
            invert_ele_rec = lp_var[vname+"Rec"].varValue
            adjust_val = invert_ele_ini - invert_ele_rec

            depth_ini = node_depth_ref[vname]["crest_ele"] - invert_ele_ini
            depth_rec = node_depth_ref[vname]["crest_ele"] - invert_ele_rec
            # ---------if the adjusted depth is too high, 
            # ------------or if the adjusted depth is negative (i.e., adjusted crest elevation exceeds given reference),
            # ------------we only modify the invert elevation and keep depth unchanged (crest elevation would be changed in this case)
            if np.abs(adjust_val) >= delta_max or depth_rec <= 0:
                depth_adjusted_info[vname]["invert_ele"] = round(invert_ele_rec, 6)
                depth_adjusted_info[vname]["depth"] = round(depth_ini, 6)
            # ---------if the node is on the ground, 
            # ------------we directly modify the invert elevation and keep its depth to zero
            elif depth_ini < 1e-6:
                depth_adjusted_info[vname]["invert_ele"] = round(invert_ele_rec, 6)
                depth_adjusted_info[vname]["depth"] = 0.0
            # ---------Otherwise we adjust the depth and invert elevation simultaneously and
            # ------------crest elevation would be unchanged in this case, which is our desired solution :)
            else:
                depth_adjusted_info[vname]["invert_ele"] = round(invert_ele_rec, 6)
                depth_adjusted_info[vname]["depth"] = round(depth_rec, 6)
    else:
        print("Linear Programming failed !!!")
    
    return depth_adjusted_info


def adjust_swmm_LinkDirection(out_path: str, inp_path: str, fixed_shp_path=None, inlet_keywords=None, inlet_namelist=None, outfall_keywords=None, outfall_namelist=None, aux_outfall_keywords=None, aux_outfall_namelist=None, solver_config=None):
    """
    Estimate the gravitational flow direction of the sewer network considering necessray topological constraints.

    Parameters
    ----------
    out_path : str
        Output path of the updated SWMM input file with corrected link directions.
    inp_path : str
        Input Path of the original SWMM input file to be processed.
    fixed_shp_path : str or None, optional
        Input Path to a shapefile containing digitized lines indicating fixed flow directions; used to enforce link orientation.
    inlet_keywords : str or None, optional
        Keyword used to identify inlet node names if `inlet_namelist` is not provided.
    inlet_namelist : list of str or None, optional
        Namelist of inlet nodes.
        If not provided, nodes with zero in-degree are inferred as inlets.
    outfall_keywords : str or None, optional
        Keyword used to identify outfall node names if `outfall_namelist` is not provided.
    outfall_namelist : list of str or None, optional
        Namelist of outfalls.
        If not provided, nodes with zero out-degree are inferred as outfalls.
    aux_outfall_keywords : str or None, optional
        Keyword used to identify auxiliary outfalls.
    aux_outfall_namelist : list of str or None, optional
        Namelist of auxiliary outfalls.
    solver_config : dict or None, optional
        Input dict which describe the solver settings of link direction derivation, for example:
                {
                    "solver": "cbc",
                    "license": "",
                    "path": "",
                    "name": "test",
                    "save_prefix": ""
                }
    """
    
    bf_eps = 1e-2
    
    new_inp_path = out_path
    shutil.copyfile(inp_path, new_inp_path)

    sim_case = swmmio.Model(new_inp_path)
    node_df = sim_case.nodes.dataframe
    link_df = sim_case.links.dataframe

    if fixed_shp_path is not None:
        coords_df = sim_case.inp.coordinates
    else:
        coords_df = None

    # ------construct the linkage graph (formulated by directed graph)
    link_graph = nx.DiGraph()
    
    node_info = [(n, {"invert_ele": node_df.loc[n]["InvertElev"], "crest_ele": node_df.loc[n]["InvertElev"] + node_df.loc[n]["MaxDepth"]}) if not np.isnan(node_df.loc[n]["MaxDepth"]) else \
                    (n, {"invert_ele": node_df.loc[n]["InvertElev"], "crest_ele": node_df.loc[n]["InvertElev"]}) for n in node_df.index]
    link_graph.add_nodes_from(node_info)

    # ------------add edges with the information of length
    if fixed_shp_path is None:
        for lname, row in link_df.iterrows():
            from_pt = row["InletNode"]
            to_pt = row["OutletNode"]
            link_graph.add_edge(from_pt, to_pt, INLET=from_pt, OUTLET=to_pt, NAME=lname)
    else:
        fixed_line_gdf = gpd.read_file(fixed_shp_path)
        link_geom = []
        link_namelist = []
        link_fromNode = []
        link_toNode = []
        for lname, row in link_df.iterrows():
            from_pt = row["InletNode"]
            to_pt = row["OutletNode"]
            link_graph.add_edge(from_pt, to_pt, INLET=from_pt, OUTLET=to_pt, NAME=lname)
            link_geom.append(LineString([coords_df.loc[from_pt], coords_df.loc[to_pt]]))
            link_namelist.append(lname)
            link_fromNode.append(from_pt)
            link_toNode.append(to_pt)
        link_gdf = gpd.GeoDataFrame(data={"NAME": link_namelist, 
                                          "geometry": link_geom, 
                                          "InletNode": link_fromNode, "OutletNode": link_toNode}, geometry="geometry", crs=fixed_line_gdf.crs)

    if inlet_namelist is None:
        if inlet_keywords is None:
            inlet_namelist = [nname for (nname, v) in link_graph.in_degree() if v==0]
            print("Inlet nodes would be selected as nodes whose in_degree is zero")
        else:
            inlet_namelist = [n for n in node_df.index if inlet_keywords in n]

    if outfall_namelist is None:
        if outfall_keywords is None:
            outfall_namelist = [nname for (nname, v) in link_graph.out_degree() if v==0]
            print("Outfall nodes would be selected as nodes whose out_degree is zero")
        else:
            outfall_namelist = [n for n in node_df.index if outfall_keywords in n]
    
    if aux_outfall_namelist is None:
        if aux_outfall_keywords is None:
            aux_outfall_namelist = []
            print("No auxiliary outfall nodes would be selected")
        else:
            aux_outfall_namelist = [n for n in node_df.index if aux_outfall_keywords in n]
    
    # ------if `fixed_shp_path` is not None, 
    # ---------then the direction of some links would be fixed according to digitization order of lines in shapefile
    if fixed_shp_path is not None:
        # fixed_line_geom = [g for g in fixed_line_gdf.geometry if isinstance(g, LineString)]
        # ------------explode the LineString to LineString segments
        fixed_line_geom = []
        for g in fixed_line_gdf.geometry:
            if isinstance(g, LineString):
                # ------------create segments of LineString
                for i in range(0, len(g.coords)-1):
                    fixed_line_geom.append(LineString([g.coords[i], g.coords[i+1]]))
            elif isinstance(g, MultiLineString):
                for subg in g.geoms:
                    for i in range(0, len(subg.coords)-1):
                        fixed_line_geom.append(LineString([subg.coords[i], subg.coords[i+1]]))
            
        fixed_seg_gdf = gpd.GeoDataFrame(data={"geometry": fixed_line_geom}, geometry="geometry", crs=fixed_line_gdf.crs)
        # ------------match the SWMM's link with the fixed-direction line geometry
        # ---------------to set corresponding direction of SWMM's link
        # ------------[1] only consider the links which intersects the buffering zone of `fixed_seg_gdf`
        fixed_seg_gdf_bf = fixed_seg_gdf.copy(deep=True)
        fixed_seg_gdf_bf["geometry"] = fixed_seg_gdf_bf["geometry"].buffer(bf_eps)
        link_gdf_isct = gpd.sjoin(link_gdf, fixed_seg_gdf_bf, how="inner", predicate="intersects")

        # ------------[2] for each intersected link, test whether it is nearly on the fixed-direction line
        fixed_dir_info = {}
        for seg in fixed_seg_gdf.geometry:
            xf_seg, yf_seg = seg.coords[0]
            xt_seg, yt_seg = seg.coords[1]
            n_vec_seg = np.array([xt_seg - xf_seg, yt_seg - yf_seg])

            subseg_res = link_gdf_isct.apply(lambda x: testSubSeg_subproc(x, seg, bf_eps), axis=1)
            link_fixed_gdf = link_gdf_isct[subseg_res]
            for ids, row in link_fixed_gdf.iterrows():
                lname = row["NAME"]
                xf_link, yf_link = row["geometry"].coords[0]
                xt_link, yt_link = row["geometry"].coords[1]

                n_vec_link = np.array([xt_link - xf_link, yt_link - yf_link])
                # ---------------[3] if the direction of link is not consistent with the direction of fixed-direction line
                # ---------------then reverse the direction of link
                fixed_dir_info[lname] = {}
                if np.dot(n_vec_seg, n_vec_link) < 0:
                    fixed_dir_info[lname]["InletNode"] = row["OutletNode"]
                    fixed_dir_info[lname]["OutletNode"] = row["InletNode"]
                else:
                    fixed_dir_info[lname]["InletNode"] = row["InletNode"]
                    fixed_dir_info[lname]["OutletNode"] = row["OutletNode"]
    else:
        fixed_dir_info = None

    changed_info = setLinkDirection(link_graph=link_graph, 
                                    outfall_namelist=outfall_namelist,
                                    edge_name_attr="NAME",
                                    inlet_namelist=inlet_namelist,
                                    aux_outfall_namelist=aux_outfall_namelist,
                                    fixed_dir_info=fixed_dir_info,
                                    solver_config=solver_config)

    new_link_df = link_df.copy(deep=True)
    for lname in changed_info.keys():
        fname_tmp = changed_info[lname]["FROM"]
        tname_tmp = changed_info[lname]["TO"]
        new_link_df.loc[lname, "InletNode"] = fname_tmp
        new_link_df.loc[lname, "OutletNode"] = tname_tmp

    export_df = pd.DataFrame(data={
        "Name": list(new_link_df.index),
        "InletNode": new_link_df["InletNode"],
        "OutletNode": new_link_df["OutletNode"],
        "Length": new_link_df["Length"],
        "Roughness": new_link_df["Roughness"],
        "InOffset": new_link_df["InOffset"],
        "OutOffset": new_link_df["OutOffset"],
        "InitFlow": new_link_df["InitFlow"],
        "MaxFlow": new_link_df["MaxFlow"],
    })

    export_df["id_pseudo"] = ""
    export_df.set_index("id_pseudo", inplace=True)
    new_sim_case = modify_model.replace_inp_section(new_inp_path, modified_section_header="CONDUITS",  new_data=export_df)


def adjust_swmm_NodeDepth(out_path: str, inp_path: str, slope_min=0.002, slope_min_ref=None, slope_max=2.0, slope_max_ref=None, depth_limit_path=None, inlet_keywords=None, inlet_namelist=None, outfall_keywords=None, outfall_namelist=None, delta_max=10.0, num_pt_min=2, group_length=50.0, stride_length=10.0, weighted=None, outfall_relaxed=False, solver_config=None):
    """Estimate the nodal depth of the sewer network constrained by the minimum and maximum allowable slopes

    Parameters
    ----------

    out_path : str
        Output path of the updated SWMM input file with corrected nodal elevations.
    inp_path : str
        Input Path of the original SWMM input file to be processed.
    slope_min : float, optional
        Minimum allowable pipe slope [m/m] used if `slope_min_ref` is not provided.
    slope_min_ref : dict or None, optional
        Dictionary mapping pipe diameter [mm] to minimum allowable slope; overrides `slope_min`.
    slope_max : float, optional
        Maximum allowable pipe slope [m/m] used if `slope_max_ref` is not provided.
    slope_max_ref : dict or None, optional
        Dictionary mapping pipe diameter [mm] to maximum allowable slope; overrides `slope_max`.
    depth_limit_path : str or None, optional
        Path to a JSON file specifying per-node minimum depth constraints.
    inlet_keywords : str or None, optional
        Keyword used to identify inlet node names if `inlet_namelist` is not provided.
    inlet_namelist : list of str or None, optional
        Namelist of inlet nodes.
        If not provided, nodes with zero in-degree are inferred as inlets.
    outfall_keywords : str or None, optional
        Keyword used to identify outfall node names if `outfall_namelist` is not provided.
    outfall_namelist : list of str or None, optional
        Namelist of outfalls.
        If not provided, nodes with zero out-degree are inferred as outfalls.
    delta_max : float, optional
        Maximum allowable elevation change between adjusted and original node elevations.
    num_pt_min : int, optional
        Minimum number of consecutive nodes to use when estimating slope. The default is `2`.
    group_length : float, optional
        Minimum length [m] of node groups when using the path-based adjustment algorithm.
        Used only if `num_pt_min` is greater than `2`.
    stride_length : float, optional
        Stride length [m] to shift the node group window during processing.
        Used only if `num_pt_min` is greater than `2`.
    weighted : str, optional
        Weighted methods for penalizing node depth adjustments.
    outfall_relaxed : bool, optional
        Whether to allow elevation adjustments at outfall nodes if necessary for feasibility.
    solver_config : dict or None, optional
        Input dict which describe the solver settings of link direction derivation, for example:
            {
                "solver": "cbc",
                "license": "",
                "path": "",
                "name": "test",
                "save_prefix": ""
            }
    
    """
    if slope_min_ref is None:
        slope_min_ref = {0: slope_min}

    if slope_max_ref is None:
        slope_max_ref = {0: slope_max}

    new_inp_path = out_path
    new_dir = os.path.dirname(new_inp_path)
    shutil.copyfile(inp_path, new_inp_path)

    sim_case = swmmio.Model(new_inp_path)
    node_df = sim_case.nodes.dataframe
    link_df = sim_case.links.dataframe
    outfall_df = sim_case.inp.outfalls

    # ------read the depth limit information
    if depth_limit_path is not None:
        with open(depth_limit_path, "r") as f:
            depth_limit_info = json.load(f)

    # ------construct the linkage graph (formulated by directed graph)
    link_graph = nx.DiGraph()
    # ------------add nodes with the information of invert elevation, crest elevation
    if depth_limit_path is not None:
        node_info = [(n, {"invert_ele": node_df.loc[n]["InvertElev"], "depth_min": depth_limit_info[n]["MinDepth"], "crest_ele": node_df.loc[n]["InvertElev"] + node_df.loc[n]["MaxDepth"]}) if not np.isnan(node_df.loc[n]["MaxDepth"]) else \
                        (n, {"invert_ele": node_df.loc[n]["InvertElev"], "depth_min": depth_limit_info[n]["MinDepth"], "crest_ele": node_df.loc[n]["InvertElev"]}) for n in node_df.index]
        depth_limited = True
    else:
        node_info = [(n, {"invert_ele": node_df.loc[n]["InvertElev"], "crest_ele": node_df.loc[n]["InvertElev"] + node_df.loc[n]["MaxDepth"]}) if not np.isnan(node_df.loc[n]["MaxDepth"]) else \
                        (n, {"invert_ele": node_df.loc[n]["InvertElev"], "crest_ele": node_df.loc[n]["InvertElev"]}) for n in node_df.index]
        depth_limited = False
    link_graph.add_nodes_from(node_info)
    
    # ------------add edges with the information of length
    for lname, row in link_df.iterrows():
        from_pt = row["InletNode"]
        to_pt = row["OutletNode"]
        if row["Shape"] == "CIRCULAR":
            diameter = row["Geom1"]
        elif row["Shape"] == "RECT_CLOSED":
            diameter = 2 * row["Geom1"] * row["Geom2"] / (row["Geom1"] + row["Geom2"])
        else:
            diameter = row["Geom1"]
        # ---------convert diameter from m to mm
        link_graph.add_edge(from_pt, to_pt, length=row["Length"], D=diameter * 1000.0)

    if inlet_namelist is None:
        if inlet_keywords is None:
            inlet_namelist = [nname for (nname, v) in link_graph.in_degree() if v==0]
            print("Inlet nodes would be selected as nodes whose in_degree is zero")
        else:
            inlet_namelist = [n for n in node_df.index if inlet_keywords in n]

    if outfall_namelist is None:
        if outfall_keywords is None:
            outfall_namelist = [nname for (nname, v) in link_graph.out_degree() if v==0]
            print("Outfall nodes would be selected as nodes whose out_degree is zero")
        else:
            outfall_namelist = [n for n in node_df.index if outfall_keywords in n]
    
    fixed_namelist = [n for n in node_df.index if np.isnan(node_df.loc[n]["MaxDepth"])]
    if num_pt_min == 2:
        print("The slope estimation would be based on 2 nodes in the graph")
        depth_adjusted_info = reconstructNodeDepth_twoNodes(link_graph,
                                                            slope_min_ref=slope_min_ref, 
                                                            slope_max_ref=slope_max_ref, 
                                                            fixed_namelist=fixed_namelist, 
                                                            delta_max=delta_max,
                                                            weighted=weighted, 
                                                            depth_limited=depth_limited, 
                                                            outfall_relaxed=outfall_relaxed,
                                                            solver_config=solver_config)
    else:
        print(f"The slope estimation would be based on a minimum of {num_pt_min} nodes in the graph")
        depth_adjusted_info = reconstructNodeDepth(link_graph, inlet_namelist, outfall_namelist, 
                                                    slope_min_ref=slope_min_ref, 
                                                    slope_max_ref=slope_max_ref, 
                                                    fixed_namelist=fixed_namelist, 
                                                    delta_max=delta_max, 
                                                    num_pt_min=num_pt_min, 
                                                    group_length=group_length, 
                                                    stride_length=stride_length, 
                                                    weighted=weighted, 
                                                    depth_limited=depth_limited, 
                                                    outfall_relaxed=outfall_relaxed,
                                                    solver_config=solver_config)

    new_node_df = node_df.copy(deep=True)
    outfall_info = {"Name": [], "InvertElev": [], "OutfallType": [], "StageOrTimeseries": []}
    for vname in depth_adjusted_info:
        if vname not in outfall_namelist:
            new_node_df.loc[vname, "InvertElev"] = depth_adjusted_info[vname]["invert_ele"]
            new_node_df.loc[vname, "MaxDepth"] = depth_adjusted_info[vname]["depth"]
        else:
            outfall_info["Name"].append(vname)
            outfall_info["InvertElev"].append(depth_adjusted_info[vname]["invert_ele"])
            outfall_info["OutfallType"].append(outfall_df.loc[vname]["OutfallType"])
            outfall_info["StageOrTimeseries"].append(outfall_df.loc[vname]["StageOrTimeseries"])

    mask = ~np.isnan(new_node_df["MaxDepth"])
    export_df = pd.DataFrame(data={
        "Name": list(new_node_df[mask].index),
        "InvertElev": new_node_df[mask]["InvertElev"].values,
        "MaxDepth": new_node_df[mask]["MaxDepth"].values,
        "InitDepth": new_node_df[mask]["InitDepth"].values,
        "SurchargeDepth": new_node_df[mask]["SurchargeDepth"].values,
        "PondedArea": new_node_df[mask]["PondedArea"].values
    })

    export_df["id_pseudo"] = ""
    export_df.set_index("id_pseudo", inplace=True)
    new_sim_case = modify_model.replace_inp_section(new_inp_path, modified_section_header="JUNCTIONS",  new_data=export_df)

    if len(outfall_info["Name"]) > 0:
        export_df = pd.DataFrame(data=outfall_info)
        export_df["id_pseudo"] = ""
        export_df.set_index("id_pseudo", inplace=True)
        new_sim_case = modify_model.replace_inp_section(new_inp_path, modified_section_header="OUTFALLS",  new_data=export_df)
    
    with open(os.path.join(new_dir, "sol.json"), "w") as f:
        json.dump(depth_adjusted_info, f, indent=4)
# ********* sewer reconstructor (END) *********


if __name__ == "__main__":
    # ------according to GB 50014-2021
    # ---------the space of rainwater inlets should between 25 m and 50 m (5.7.3)
    # ---------and spacing can be larger than 50 m if the slope is larger than 2% 

    base_shp_dir = "/home/lyy/project/IUHM_GPU/Yinchuan_case/tmp/sewer"
    output_shp_dir = "/home/lyy/project/IUHM_GPU/Yinchuan_case/tmp/sewer/output"

    if not os.path.exists(output_shp_dir):
        os.makedirs(output_shp_dir)

    for spacing in [30, 40, 50, 60, 70, 100]:
        split_drainage_line_with_inlet(input_sewer_path=os.path.join(base_shp_dir, "yc_whole_sewer_clean.gpkg"),
                                       output_sewer_path=os.path.join(output_shp_dir, f"yc_whole_sewer_D{spacing}.gpkg"), 
                                       output_inlet_path=os.path.join(output_shp_dir, f"yc_whole_inlet_D{spacing}.gpkg"),
                                       inlet_spacing=spacing,
                                       simplified_tolerance=0.1, 
                                       geom_scale_factor=1.0,
                                       nonInlet_node_path=os.path.join(base_shp_dir, "yc_whole_outfall.gpkg"),
                                       mask_boundary_path=os.path.join(base_shp_dir, "yc_whole_sewer_mask.gpkg"))