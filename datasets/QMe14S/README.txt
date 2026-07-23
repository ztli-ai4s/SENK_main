## Details of MD_{file_idx}.h5 File
The MD_x. h5 files store the dynamic configurations and corresponding properties from the MD simulations, each single file containing 1 million configurations for 10 thousands molecules. 
Data Fields:
edge_index: The graph's edge index, type is torch.tensor.
pos: Atomic position data, type is torch.tensor.
z: Atomic number, type is torch.tensor.
dipole: Molecular dipole moment, type is torch.tensor.
force: Atomic force , type is torch.tensor.
atomization_energy: Atomization energy, stored as a float.
smile: The SMILES representation of the molecule, type is str.
Data Object:
Each group of data will be converted into a Data object, containing the following fields:
Data(
    edge_index=edge_index,
    pos=pos,
    z=z,
    dipole=dipole,
    force=force,
    atomization_energy=atomization_energy,
    smile=smile
)

## Details of QMe14S_single_point.h5 File
The QMe14S_single_point. h5 file stores the 10 configurations randomly selected from 100 dynamic configurations and the corresponding dynamic properties. 
Data Fields:
edge_index: The graph's edge index, type is torch.tensor.
pos: Atomic position data, type is torch.tensor.
z: Atomic number, type is torch.tensor.
atomization_energy: Atomization energy, stored as a float.
dipole: Molecule dipole moment, type is torch.tensor.
polar: Molecule polarizability, type is torch.tensor.
dedipole: Molecule dipole derivatives, type is torch.tensor.
Hi: The atomic part of the Hessian matrix, type is torch.tensor.
Hij: The interatomic part of the Hessian matrix, type is torch.tensor.
smile: The SMILES representation of the molecule, type is str.
r2:  Electronic spatial extent, stored as a float.
quadrupole: Molecule quadrupole moment, type is torch.tensor.
octapole: Molecule octupole moment, type is torch.tensor.
force: Atomic force, type is torch.tensor.
Data Object:
Each group of data will be converted into a Data object, containing the following fields:
Data(
    edge_index=edge_index,
    pos=pos,
    z=z,
    atomization_energy=atomization_energy,
    dipole=dipole,
    polar=polar,
    dedipole=dedipole,
    Hi=Hi,
    Hij=Hij,
    smile=smile,
    r2=r2,
    quadrupole=quadrupole,
    octapole=octapole,
    force=force
)

## Details of OPT_186102.h5 File
OPT_186102. h5 file contains the optimized geometries and corresponding static properties of the 186,102 molecules. 
Data Fields:
edge_index: The graph's edge index, type is torch.tensor.
pos: Atomic position data, type is torch.tensor.
smile: The SMILES representation of the molecule, type is str.
z: Atomic number, type is torch.tensor.
quadrupole: Molecule quadrupole moment, type is torch.tensor.
octapole: Molecule octupole moment, type is torch.tensor.
npacharge: Natural Population Charge, type is torch.tensor.
dipole: Molecule dipole moment, type is torch.tensor.
polar: Molecule polarizability, type is torch.tensor.
hyperpolar: First hyperpolarizability, type is torch.tensor.
Hij: The interatomic part of the Hessian matrix, type is torch.tensor.
Hii: The atomic part of the Hessian matrix, type is torch.tensor.
dedipole: Dipole Derivatives, type is torch.tensor.
depolar: Polarizability Derivatives, type is torch.tensor.
Data Object:
Each group of data will be converted into a Data object, containing the following fields:
Data(
    edge_index=edge_index,
    pos=pos,
    smile=smile,
    z=z,
    quadrupole=quadrupole,
    octapole=octapole,
    npacharge=npacharge,
    dipole=dipole,
    polar=polar,
    hyperpolar=hyperpolar,
    Hij=Hij,
    Hii=Hii,
    dedipole=dedipole,
    depolar=depolar
)

## Details of Hessian_opt.h5 File
Hessian_opt.h5 file contains the Hessian matrix of the equilibrium (optimized).
Data Fields:
pos: Atomic position, type is torch.tensor.
smiles: The SMILES representation of the molecule, type is str.
hessian: Hessian matrix data, type is torch.tensor.
Data Object:
Each group of data will be converted into a Data object, containing the following fields:
Data(
    pos=pos,
    smiles=smiles,
    hessian=hessian
)

## Details of Hessian_single_point.h5 File
Hessian_opt.h5 file contains the Hessian matrix of the nonequilibrium (dynamic).
Data Fields:
pos: Atomic position data, type is torch.tensor.
z: Atomic number, type is torch.tensor.
hessian: Hessian matrix data, type is torch.tensor.
smile: The SMILES representation of the molecule, type is str.
Data Object:
Each group of data will be converted into a Data object, containing the following fields:
Data(
    pos=pos,
    z=z,
    hessian=hessian,
    smile=smile
)

## Details of NMR.h5 File
NMR.h5 file containsthe nuclear shielding tensor and their isotropic values.
Data Fields:
pos: Atomic position data, type is torch.tensor.
z: Atomic number, type is torch.tensor.
nst: Nuclear Shielding Tensor, type is torch.tensor.
nst_iso: Isotropic value of the Nuclear Shielding Tensor, type is torch.tensor.
smile: The SMILES representation of the molecule, type is str.
Data Object:
Each group of data will be converted into a Data object, containing the following fields:
Data(
    pos=pos,
    z=z,
    nst=nst,
    nst_iso=nst_iso,
    smile=smile
)

## Details of IR_broaden.zip
The broaden IR spectra of the optimized molecules, the wave number is 500-4000cm-1.
line0:The SMILES representation of the molecule
line1-line(n+1):
	colomn0: Atomic number
	colomn1-colomn3: The Cartesian coordinates of the molecule
line(n+3): IR intensity after Lorentz broadening with a half-width of 15 cm⁻1

## Details of Raman_broaden.zip:
The broaden Raman spectra of the optimized molecules, the wave number is 500-4000cm-1.
line0: The SMILES representation of the molecule
line1-line(n+1):
	colomn0: Atomic number
	colomn1-colomn3: The Cartesian coordinates of the molecule
line(n+3): Raman intensity after Lorentz broadening with a half-width of 10 cm⁻1