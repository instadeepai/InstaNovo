# Reference: Natively Supported Modifications

InstaNovo has been trained to recognize a set of common post-translational modifications (PTMs). This document lists the modifications that are natively supported by the pre-trained models.

| Amino Acid                  | Single Letter | Modification            | Mass Delta (Da) | Unimod ID                                                                   |
| --------------------------- | ------------- | ----------------------- | --------------- | --------------------------------------------------------------------------- |
| Methionine                  | M             | Oxidation               | +15.9949        | [UNIMOD:35](https://www.unimod.org/modifications_view.php?editid1=35)   |
| Cysteine                    | C             | Carboxyamidomethylation | +57.0215        | [UNIMOD:4](https://www.unimod.org/modifications_view.php?editid1=4)     |
| Asparagine, Glutamine       | N, Q          | Deamidation             | +0.9840         | [UNIMOD:7](https://www.unimod.org/modifications_view.php?editid1=7)     |
| Serine, Threonine, Tyrosine | S, T, Y       | Phosphorylation         | +79.9663        | [UNIMOD:21](https://www.unimod.org/modifications_view.php?editid1=21)   |
| N-terminal                  | -             | Ammonia Loss            | -17.0265        | [UNIMOD:385](https://www.unimod.org/modifications_view.php?editid1=385) |
| N-terminal                  | -             | Carbamylation           | +43.0058        | [UNIMOD:5](https://www.unimod.org/modifications_view.php?editid1=5)     |
| N-terminal                  | -             | Acetylation             | +42.0106        | [UNIMOD:1](https://www.unimod.org/modifications_view.php?editid1=1)     |

The residue configuration can be found in the [`instanovo/configs/residues/extended.yaml`](https://github.com/instadeepai/InstaNovo/blob/main/instanovo/configs/residues/extended.yaml) file.
