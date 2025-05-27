import os
import argparse
import json

from input_formatter import shp2swmm
from sewer_reconstructor import split_drainage_line_with_inlet, adjust_swmm_LinkDirection, adjust_swmm_NodeDepth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWMM reconstructor")
    parser.add_argument("--slope_config_path", type=str, help="Path to slope configuration file")
    parser.add_argument("--solver_config_path", type=str, help="Path to LP solver configuration file")
    parser.add_argument("--swmm_config_path", type=str, help="Path to SWMM configuration file")
    args = parser.parse_args()

    slope_config_path = args.slope_config_path
    lp_config_path = args.solver_config_path
    swmm_config_path = args.swmm_config_path

    with open(slope_config_path, "r") as f:
        slope_config = json.load(f)
        slope_max_ref = {int(k): float(v) for k, v in slope_config["slope_max"].items()}
        slope_min_ref = {int(k): float(v) for k, v in slope_config["slope_min"].items()}
    
    with open(lp_config_path, "r") as f:
        solver_config = json.load(f)

    if "outfall_relaxed" in solver_config:
        outfall_relaxed = solver_config["outfall_relaxed"]
        if outfall_relaxed == "False":
            outfall_relaxed = False
        else:
            outfall_relaxed = True
    else:
        outfall_relaxed = True

    with open(swmm_config_path, "r") as f:
        swmm_config = json.load(f)    

    inlet_spacing = swmm_config["parameter"]["inlet_spacing"]
    DEM_path = swmm_config["parameter"]["DEM_path"]

    swmm_geometry_settings = swmm_config["geometry_settings"]
    if "fixed_direction" in swmm_geometry_settings:
        swmm_fixedDir_path = swmm_geometry_settings["fixed_direction"]
    else:
        swmm_fixedDir_path = None

    swmm_simulation_settings = swmm_config["simulation_settings"]
    swmm_case_epsg_code = swmm_config["simulation_settings"]["EPSG_code"]
    swmm_inp_file = swmm_config["simulation_settings"]["input_file"]

    swmm_inp_folder = os.path.dirname(swmm_inp_file)
    swmm_inp_basename = os.path.basename(swmm_inp_file)

    SWMM_SHP_SAVED = False

    # [1] check the input files
    if not os.path.exists(swmm_inp_folder):
        os.makedirs(swmm_inp_folder)

    ini_sewer_path = swmm_geometry_settings["sewer"]
    if not os.path.exists(ini_sewer_path):
        raise FileNotFoundError(f"Input sewer shapefile not found: {ini_sewer_path}")
    else:
        print(f"Input sewer shapefile found: {ini_sewer_path}")

    outfall_path = swmm_geometry_settings["outfall"]
    if not os.path.exists(outfall_path):
        raise FileNotFoundError(f"Input outfall shapefile not found: {outfall_path}")
    else:
        print(f"Input outfall shapefile found: {outfall_path}")

    if "masked_area" in swmm_geometry_settings:
        masked_area_path = swmm_geometry_settings["masked_area"]
        if not os.path.exists(masked_area_path):
            if len(masked_area_path) > 0:
                print(f"Input masked area shapefile not found: {masked_area_path}")
            masked_area_path = None
    else:
        masked_area_path = None

    if swmm_fixedDir_path is not None:
        if not os.path.exists(swmm_fixedDir_path):
            if len(swmm_fixedDir_path) > 0:
                print(f"Input fixed direction shapefile not found: {swmm_fixedDir_path}")
            swmm_fixedDir_path = None

    # [2] generate rainwater inlets with a given spacing
    output_sewer_path = os.path.join(swmm_inp_folder, f"_sewer_ILS{int(inlet_spacing)}.gpkg")
    output_inlet_path = os.path.join(swmm_inp_folder, f"_inlet_ILS{int(inlet_spacing)}.gpkg")

    print("=> Generating inlets with spacing: ", inlet_spacing, "m")

    split_drainage_line_with_inlet(input_sewer_path=ini_sewer_path,
                                    output_sewer_path=output_sewer_path, 
                                    output_inlet_path=output_inlet_path,
                                    inlet_spacing=inlet_spacing,
                                    nonInlet_node_path=outfall_path,
                                    mask_boundary_path=masked_area_path,
                                    inlet_depth=0.6)

    print("=> Output sewer shapefile: ", output_sewer_path)
    print("=> Output inlet shapefile: ", output_inlet_path)

    # [2] generate the SWMM input file
    ini_swmm_path = os.path.join(swmm_inp_folder, swmm_inp_basename.replace(".inp", "_INI.inp"))
    swmm_geometry_settings["road_inlet"] = output_inlet_path
    swmm_geometry_settings["sewer"] = output_sewer_path

    shp2swmm(shp_info=swmm_geometry_settings, 
                save_path=swmm_inp_basename.replace(".inp", "_INI.inp"), 
                DEM_path=DEM_path,
                simulation_settings=swmm_simulation_settings,
                shp_saved=SWMM_SHP_SAVED, 
                save_prefix=swmm_inp_folder,
                save_epsg=swmm_case_epsg_code,
                solver_config=solver_config)

    print("=> Initial SWMM input file generated: ", ini_swmm_path)

    # [3] estimate the link direction
    redirect_swmm_path = os.path.join(swmm_inp_folder, swmm_inp_basename.replace(".inp", "_REDIRECT.inp"))
    adjust_swmm_LinkDirection(out_path=redirect_swmm_path,
                                inp_path=ini_swmm_path,
                                fixed_shp_path=swmm_fixedDir_path, 
                                inlet_keywords="IL", 
                                outfall_keywords="OF",
                                aux_outfall_keywords="OAux",
                                solver_config=solver_config)
    
    print("=> Redirected SWMM input file generated: ", redirect_swmm_path)

    # [4] estimate the node depth
    final_swmm_path = os.path.join(swmm_inp_folder, swmm_inp_basename)
    adjust_swmm_NodeDepth(out_path=final_swmm_path,
                            inp_path=redirect_swmm_path, 
                            depth_limit_path=os.path.join(swmm_inp_folder, "node_depth.json"),
                            slope_min_ref=slope_min_ref, 
                            slope_max_ref=slope_max_ref, 
                            inlet_keywords="IL", outfall_keywords="OF", 
                            num_pt_min=2,
                            weighted=None, 
                            outfall_relaxed=outfall_relaxed,
                            solver_config=solver_config)
    
    print("=> Final SWMM input file generated: ", final_swmm_path)
    print("=> SWMM reconstruction completed.")
