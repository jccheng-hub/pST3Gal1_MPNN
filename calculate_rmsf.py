#!/usr/bin/env python
import glob
import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis import rms

# Index-1 because bioemu start at 0
# fixed = "44 45 46 47 48 49 50 51 87 88 89 90 91 92 93 94 110 111 112 113 114 115 116 131 132 133 134 135 136 137 138 139 149 150 151 152 153 155 156 171 172 173 174 175 176 195 203 208 209 210 211 212 213 214 215 216 217 218 229 230 231 232 233 234 235 236 238 239 240 241 242 243 244 245 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 268 279 280 281 283"

outcsv="rmsf_data.csv"

all_base = glob.glob("bioemu_ensembles/*/*.pdb")
all_traj = glob.glob("bioemu_ensembles/*/*.xtc")

all_base.sort()
all_traj.sort()

full_list = []
for i in range(len(all_base)):
    base = all_base[i]
    traj = all_traj[i]
    print(f"Analyzing {base} with {traj}...")
   
    u = mda.Universe(base, traj)
    aligner = align.AlignTraj(u, u, select="backbone", in_memory=True).run()
    u_select = u.select_atoms(f"protein and name CA")
    # u_select = u.select_atoms(f"protein and name CA and resid {fixed}")
    rmsf_analysis = rms.RMSF(u_select).run()
    
    resids = u_select.resids
    rmsf_vals = rmsf_analysis.results.rmsf
    
    if i == 0:
        resids += 1
        str_arr = resids.astype(str)
        str_arr = np.insert(str_arr, 0, ["resid"])
        # str_arr = np.insert(str_arr, 0, ["path", "frames"])
        full_list.append(str_arr)

    str_arr = rmsf_vals.astype(str)
    str_arr = np.insert(str_arr, 0, [base])
    # str_arr = np.insert(str_arr, 0, [base, u.trajectory.n_frames])
    full_list.append(str_arr)

final_matrix = np.column_stack(full_list)
np.savetxt(outcsv, final_matrix, delimiter = ",", fmt="%s")
