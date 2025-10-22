import subprocess
# learning process
from train.learn_pytorch3d import fit_mesh_via_rendering
from train.pytorch3d_uv import fit_texture_via_rendering
from train.pytorch3d_uv_neuraltexture import fit_texture_via_rendering_neuraltexture
from train.uv_neuraltexture_sh import uv_neuraltexture_sh
# visual anagrams
from train.visual_anagrams_2d import visual_anagrams_2d
from train.visual_anagrams_2d_sd import visual_anagrams_2d_sd
from train.visual_anagrams_2d_sd_depth import visual_anagrams_2d_sd_depth
# factorized diffusion
from train.factorized_diffusion_2d import factorized_diffusion_2d
from train.factorized_diffusion_2d_sd import factorized_diffusion_2d_sd
from train.factorized_diffusion_2d_depth import factorized_diffusion_2d_depth
from train.visual_anagrams_3d import visual_anagrams_3d
# illusion diffusion
from train.illusion_diffusion import IllusionDiffusion
from train.illusion_diffusion_depth import IllusionDiffusionDepth
# PTDiffusion
from train.PTDiffusion_2d import PTDiffusion_2d
from train.PTDiffusion_2d_class import PTDiffusion_2d_class
from train.PTDiffusion_2d_to_3d import PTDiffusion_2d_to_3d
from train.PTDiffusion_3d import PTDiffusion_3d
from train.PTDiffusion_3d_RGB_optimize import PTDiffusion_3d_RGB_optimize
from train.PTDiffusion_3d_RGB_optimize_wo_VSD import PTDiffusion_3d_RGB_optimize_wo_VSD
# Instruct_Tex2Tex
from train.IP2P_2d import IP2P_2d
from train.IP2P_PTD_2d import IP2P_PTD_2d
from train.Instruct_Tex2Tex import Instruct_Tex2Tex
from train.Instruct_Tex2Tex_2_secrets import Instruct_Tex2Tex_2_secrets
from train.Tex2Tex_GaussTrap import Tex2Tex_GaussTrap
from train.Instruct_Tex2Tex_GaussTrap import Instruct_Tex2Tex_GaussTrap
# Histogram
from train.histogram_2d import histogram_2d

# baseline
from train.baseline import baseline
from train.baseline_scannetpp import baseline_scannetpp
from train.inference_baseline_scannetpp import inference_baseline_scannetpp

# ScanNet++
from utils.scannetpp_dataloader import scannetpp_dataloader
from utils.scannetpp_datasaver import scannetpp_datasaver
from utils.scannetpp_datasaver_chunk import scannetpp_datasaver_chunk
from utils.scannetpp_large_scene_datasaver_chunk import scannetpp_large_scene_datasaver_chunk
from utils.scannetpp_large_scene_datasaver_semantic_chunk import scannetpp_large_scene_datasaver_semantic_chunk
from utils.render_video_scannetpp import render_video_scannetpp
from utils.render_video_nerfstudio import render_video_nerfstudio


if __name__ == '__main__':
    # Run the data preparation script
    # subprocess.run([
    #     "blender",
    #     "--background",
    #     "--python",
    #     "./data/data_prepare.py"
    # ])

    # Run the smart UV project script
    # subprocess.run([
    #     "blender",
    #     "--background",
    #     "--python",
    #     "./utils/smart_uv_project.py"
    # ])

    # Run the Learn PyTorch3D script
    # fit_mesh_via_rendering()
    # fit_texture_via_rendering()
    # fit_texture_via_rendering_neuraltexture()
    # uv_neuraltexture_sh()

    # Run the Illusion Diffusion script
    # IllusionDiffusion()
    # IllusionDiffusionDepth()

    # Run the Visual Anagrams 2D script
    # visual_anagrams_2d()
    # visual_anagrams_2d_sd()
    # visual_anagrams_2d_sd_depth()
    
    # Run the Factorized Diffusion 2D script
    # factorized_diffusion_2d()
    # factorized_diffusion_2d_sd()
    # factorized_diffusion_2d_depth()

    # Run the PTDiffusion 3D script
    # PTDiffusion_2d()
    # PTDiffusion_2d_class()
    # PTDiffusion_2d_to_3d()
    # PTDiffusion_3d()
    # PTDiffusion_3d_RGB_optimize()
    # PTDiffusion_3d_RGB_optimize_wo_VSD()

    # Run the Instruct_Tex2Tex script
    # IP2P_2d()
    # IP2P_PTD_2d()
    # Instruct_Tex2Tex()
    # Instruct_Tex2Tex_2_secrets()
    # Tex2Tex_GaussTrap()
    # Instruct_Tex2Tex_GaussTrap()

    # Run the Histogram script
    # histogram_2d()

    # Run the ScanNet++ scripts
    # scannetpp_dataloader()
    # scannetpp_datasaver()
    # scannetpp_datasaver_chunk()
    # scannetpp_large_scene_datasaver_chunk()
    # scannetpp_large_scene_datasaver_semantic_chunk()

    # baseline
    # baseline()
    # baseline_scannetpp()
    inference_baseline_scannetpp()

    # Run the ScanNet++ mesh video rendering script
    # render_video_scannetpp()
    # render_video_nerfstudio()

    # Run the Visual Anagrams 3D script
    # visual_anagrams_3d()