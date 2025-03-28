# PoseProbe 🌟  
**Generic Objects as Pose Probes for Few-shot View Synthesis**  (accepted by <span style="color: #FF5733;">**IEEE TCSVT 2025**</span>) 

🚀 ​**Official implementation of PoseProbe** using <span style="color: #FFC300;">**PyTorch**</span>.  
✨ ​**Full Code is coming soon!** If you find this repository useful to your research or work, it is greatly appreciated to ​**star** this repository and ​**cite** our paper.  

---


## 📌 Project Links  
- ​[Project Page](https://zhirui-gao.github.io/PoseProbe.github.io/)
- ​[Arxiv Paper](https://arxiv.org/pdf/2408.16690) 

---


## 🎯 Abstract  

Radiance fields, including NeRFs and 3D Gaussians, demonstrate great potential in high-fidelity rendering and scene reconstruction, while they require a substantial number of posed images as input. COLMAP is frequently employed for preprocessing to estimate poses. However, COLMAP necessitates a large number of feature matches to operate effectively, and struggles with scenes characterized by sparse features, large baselines, or few-view images. We aim to tackle few-view NeRF reconstruction using only 3 to 6 unposed scene images, freeing from COLMAP initializations. Inspired by the idea of calibration boards in traditional pose calibration, we propose a novel approach of utilizing everyday objects, commonly found in both images and real life, as **pose probes**. By initializing the probe object as a cube shape, we apply a dual-branch volume rendering optimization (object NeRF and scene NeRF) to constrain the pose optimization and jointly refine the geometry. PnP matching is used to initialize poses between images incrementally, where only a few feature matches are enough. PoseProbe achieves state-of-the-art performance in pose estimation and novel view synthesis across multiple datasets in experiments. We demonstrate its effectiveness, particularly in few-view and large-baseline scenes where COLMAP struggles. In ablations, using different objects in a scene yields comparable performance, showing that PoseProbe is robust to the choice of probe objects. Our project page is available at [here](https://zhirui-gao.github.io/PoseProbe.github.io/) 


## 🔑 Key Features  
- <span style="color: #FF5733;">**Few-view Reconstruction**</span>: Works with only ​**3 to 6 unposed images**.  
- <span style="color: #FF5733;">**No COLMAP Dependency**</span>: Completely bypasses COLMAP initialization.  
- <span style="color: #FF5733;">**Everyday Objects as Probes**</span>: Utilizes common objects for pose estimation.  
- <span style="color: #FF5733;">**Dual-branch Optimization**</span>: Jointly optimizes object and scene geometry.  

---


## 📚 Reference  
If you find this code useful for your research, please use the following BibTeX entry:  
```bibtex
@article{gao2024generic,
  title={Generic Objects as Pose Probes for Few-Shot View Synthesis},
  author={Gao, Zhirui and Yi, Renjiao and Zhu, Chenyang and Zhuang, Ke and Chen, Wei and Xu, Kai},
  journal={arXiv preprint arXiv:2408.16690},
  year={2024}
}
