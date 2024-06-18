"""
Reference:
- graphein.protein.utils.filter_dataframe
- graphein.protein.edges.distance.compute_distmat
"""

# basic
import math
import shutil
import tempfile
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, MMCIFParser
from biopandas.pdb import PandasPdb
from pymol import cmd


# ----------------------------------------
# General utils
# ----------------------------------------
def timestamp() -> str:
    # create time stamp in format 2024May21-123045
    return time.strftime("%Y%b%d-%H%M%S", time.localtime())


def print_msg(msg: str, level: str = "INFO") -> None:
    level = level.upper()
    assert level in ["INFO", "ERROR", "WARNING", "DEBUG"]
    if level == "INFO":
        level = "INFO "
    print(f"{level} {timestamp()}: {msg}")


def deduplicate_res_pairs(res_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Deduplicate residue pairs. E.g. (A:ASN:54, B:TRP:403) and (B:TRP:403, A:ASN:54) are the same.

    Args:
        res_pairs (List[Tuple[str, str]]): List of residue pairs.

    Raises:
        ValueError: _description_

    Returns:
        List[Tuple[str, str]]: Deduplicated residue pairs.
    """
    # deduplicate
    res_pairs_set = set(frozenset(pair) for pair in res_pairs)
    # convert back to tuples
    res_pairs_nr = sorted(tuple(sorted(pair)) for pair in res_pairs_set)
    return res_pairs_nr


def filter_interaction(
    res_pairs: List[Tuple[str, str]], chain_label_1: str, chain_label_2: str
) -> List[Tuple[str, str]]:
    """
    Filter the interaction pairs by chain label.

    Args:
        res_pairs (List[Tuple[str, str]]): List of interaction residue pairs.
        chain_label_1 (str): Chain label 1.
        chain_labels_2 (str): Chain label 2.

    Returns:
        List[Tuple[str, str]]: Filtered interaction residue pairs.
    """
    # if chain labels are provided, filter by chain labels
    return sorted(
        (res1, res2)
        for res1, res2 in res_pairs
        if res1.split(":")[0] != res2.split(":")[0]
        and res1.split(":")[0] in (chain_label_1, chain_label_2)
        and res2.split(":")[0] in (chain_label_1, chain_label_2)
    )


@contextmanager
def timing_context(label: str = "Block") -> Generator[None, Any, None]:
    """Context manager for timing the execution of a code block with enhanced logging.

    Args:
        label (str): A label for the code block to be timed.

    Yields:
        None: This context manager yields nothing and only prints execution time details.
    """
    start_time = time.time()
    print_msg(f"{label} started.", "INFO")
    try:
        yield
    finally:
        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)
        print_msg(f"{label} ended.", "INFO")
        print_msg(f"Total execution time for {label}: {mins:.0f}m {secs:.2f}s", "INFO")


def three_to_one(aa_code: str) -> str:
    """
    Maps a three-letter amino acid code to a one-letter code or 'X' for unknown codes.

    Args:
        aa_code (str): The three-letter code of the amino acid.

    Returns:
        str: The one-letter code of the amino acid, or 'X' if the code is unknown.
    """
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    # Ensure that the input is uppercase to match dictionary keys
    aa_code_upper = aa_code.upper()

    # Return the mapped one-letter code or 'X' for unknown codes
    return aa_map.get(aa_code_upper, "X")


# ----------------------------------------
# Output formatting
# ----------------------------------------


# a helper function to convert the result into json format
def to_json(
    contacts_df: pd.DataFrame,
    interaction_annotation_dict: Dict[Tuple[str, str], List[str]],
):
    """
    E.g. dataframe
    residue_1	residue_2	distance
    H:ASN:54::CB  N:TRP:403::CZ2  4.318736
    H:ASN:54::CB    N:THR:401::O  3.360026

    interaction_annotation_dict e.g.
    {
        ('H:ASN:54:', 'N:TRP:403:'): ['HBond']
    }

    Output:
    [
        {
            key: 1                       // row number,
            name: "ASN H 54:TRP N 403",  // space separated
            distance: 3.908668           // distance
            loci1: { seq_id: '54' , chain_id: 'H', comp_id: 'ASN' },
            loci2: { seq_id: '403', chain_id: 'N', comp_id: 'TRP' },
            "distance": 3.908668,
            "interaction": "HBond"
        },
        {
            key: 2                         // row number,
            name: "ASN H 54A:TRP N 403A",  // space separated
            distance: 3.908668             // distance
            loci1: { seq_id: '54A' , chain_id: 'H', comp_id: 'ASN' },
            loci2: { seq_id: '403A', chain_id: 'N', comp_id: 'TRP' },
            "interaction": ""
        }
    ]

    Args:
        df (pd.DataFrame): A DataFrame containing the residue pairs and distances.
        interaction_annotation_dict (Dict[Tuple[str, str], List[str]]): A dictionary containing the interaction annotation.
            e.g. {('H:ASN:54:', 'N:TRP:403:'): ['HBond']}

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the residue pairs and distances.
    """
    # a shorthand for the interaction annotation dict
    itfd = interaction_annotation_dict
    out = []
    for i, row in contacts_df.iterrows():
        chain_label_1, res_name_1, res_num_1, insert_1, atom_name_1 = row[
            "residue_1"
        ].split(":")
        chain_label_2, res_name_2, res_num_2, insert_2, atom_name_2 = row[
            "residue_2"
        ].split(":")
        # join the residue number and insertion
        res_lab_1 = "".join([res_num_1, insert_1])
        res_lab_2 = "".join([res_num_2, insert_2])
        # map the interaction type
        it_str = ",".join(
            itfd.get(
                (
                    f"{chain_label_1}:{res_name_1}:{res_num_1}:{insert_1}",
                    f"{chain_label_2}:{res_name_2}:{res_num_2}:{insert_2}",
                ),
                [],
            )
        )  # NOTE: in future, need to handle multiple interaction types, use comma is not difficult for frontend to filter
        # add to the output
        out.append(
            {
                "key": i + 1,
                "name": f"{res_name_1} {chain_label_1} {res_lab_1}:{res_name_2} {chain_label_2} {res_lab_2}",
                "distance": row["distance"],
                "loci1": {
                    "seq_id": res_lab_1,
                    "chain_id": chain_label_1,
                    "comp_id": res_name_1,
                },
                "loci2": {
                    "seq_id": res_lab_2,
                    "chain_id": chain_label_2,
                    "comp_id": res_name_2,
                },
                "interaction": it_str,
            }
        )

    return out


# a helper function to convert the result into csv format
def to_csv(
    contacts_df: pd.DataFrame,
    interaction_annotation_dict: Dict[Tuple[str, str], List[str]],
):
    """
    E.g. dataframe
    atom_1          atom_2          distance
    H:ASN:54::CB    N:TRP:403::CZ2  4.318736
    H:ASN:54::CB    N:THR:401::O    3.360026

    interaction_annotation_dict e.g.
    {
        ('H:ASN:54:', 'N:TRP:403:'): ['HBond']
    }

    Output:
    residue_1,residue_2,distance,interaction
    ASN H 54:TRP N 403,3.908668,HBond
    ASN H 54A:TRP N 403A,3.908668,''  // empty string if no interaction

    Args:
        df (pd.DataFrame): A DataFrame containing the residue pairs and distances.
        interaction_annotation_dict (Dict[Tuple[str, str], List[str]]): A dictionary containing the interaction annotation.
            e.g. {('H:ASN:54:', 'N:TRP:403:'): ['HBond']}

    Returns:
        pd.DataFrame: A DataFrame containing the residue pairs, distances, and interactions.
    """
    # a shorthand for the interaction annotation dict
    itfd = interaction_annotation_dict
    out = {
        "residue_1": [],
        "residue_2": [],
        "distance": [],
        "chain_1": [],
        "chain_2": [],
        "interaction": [],
    }
    for i, row in contacts_df.iterrows():
        chain_label_1, res_name_1, res_num_1, insert_1, atom_name_1 = row[
            "atom_1"
        ].split(":")
        chain_label_2, res_name_2, res_num_2, insert_2, atom_name_2 = row[
            "atom_2"
        ].split(":")
        # join the residue number and insertion
        res_lab_1 = "".join([res_num_1, insert_1])
        res_lab_2 = "".join([res_num_2, insert_2])

        # map the interaction type
        it_str = ",".join(
            itfd.get(
                (
                    f"{chain_label_1}:{res_name_1}:{res_num_1}:{insert_1}",
                    f"{chain_label_2}:{res_name_2}:{res_num_2}:{insert_2}",
                ),
                ["Contact"],
            )
        )  # NOTE: in future, need to handle multiple interaction types, use comma is not difficult for frontend to filter

        # add to the output
        out["residue_1"].append(f"{chain_label_1}:{res_name_1}:{res_num_1}:{insert_1}")
        out["residue_2"].append(f"{chain_label_2}:{res_name_2}:{res_num_2}:{insert_2}")
        out["chain_1"].append(chain_label_1)
        out["chain_2"].append(chain_label_2)
        out["distance"].append(round(row["distance"], 2))
        out["interaction"].append(it_str)

    # to pd.DataFrame
    out = pd.DataFrame(out)

    return out


# ----------------------------------------
# Structure File Processing
# ----------------------------------------
def _add_node_id_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add node_id column to the dataframe in the format of [chain_id]:[residue_name]:[residue_number]:[insertion]

    Args:
        df (pd.DataFrame): A DataFrame containing the PDB data. Returned by PandasPdb.df["ATOM"]

    Returns:
        pd.DataFrame: updated DataFrame with the node_id column added.
    """
    df["node_id"] = (
        df["chain_id"]
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].astype(str)
        + ":"
        + df["insertion"]
    )
    df["node_id"] = df["node_id"].str.replace(r":\s*$", "", regex=True)
    return df


def convert_mmcif_to_pdb(input_file: Union[Path, str], output_file: Union[Path, str]):
    """Convert a structure from mmCIF format to PDB format using BioPython.

    Args:
        input_file (str): The path to the input mmCIF file.
        output_file (str): The path where the output PDB file will be saved.
    """
    parser = MMCIFParser()
    structure = parser.get_structure("MMCIF", str(input_file))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_file))
    print_msg(f"Converted {input_file} to {output_file}", "info")


def _load_abdb_pdb_as_df(pdb_fp: Path) -> PandasPdb:
    # parse it as df
    ppdb = PandasPdb().read_pdb(str(pdb_fp))
    # convert to dataframe
    atom_df = ppdb.df["ATOM"]
    # add node_id in the format of [chain_id]:[residue_name]:[residue_number]:[insertion]
    atom_df["node_id"] = (
        atom_df["chain_id"]
        + ":"
        + atom_df["residue_name"]
        + ":"
        + atom_df["residue_number"].astype(str)
        + ":"
        + atom_df["insertion"]
    )
    # remove the tailing space and colon in the node_id if insertion is empty
    atom_df["node_id"] = atom_df["node_id"].str.replace(r":\s*$", "", regex=True)
    # update the atom df
    ppdb.df["ATOM"] = atom_df

    return ppdb


def load_pdb_with_non_standard_extension(pdb_fp: Path) -> PandasPdb:
    # get the file name
    file_name = pdb_fp.name
    # get the file name without extension
    file_name_no_ext = file_name.split(".")[0]

    struct = None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # copy the file to the temp directory with a .pdb extension
            new_pdb_fp = Path(temp_dir) / f"{file_name_no_ext}.pdb"
            shutil.copy(pdb_fp, new_pdb_fp)
            # parse
            struct = _load_abdb_pdb_as_df(new_pdb_fp)
    except Exception as e:
        print(e)
        return
    return struct


def parse_pdb_file_as_atom_df(pdb_fp: Union[Path, str]) -> pd.DataFrame:
    # 使用 biopandas 处理 PDB 文件
    ppdb = PandasPdb().read_pdb(str(pdb_fp))
    atom_df = ppdb.df["ATOM"]
    process_atom_df(atom_df)  # update columns
    return atom_df


# ----------------------------------------
# Geometry related functions
# ----------------------------------------
def calculate_min_residue_distances(
    atom_df: pd.DataFrame, chain_label_1: str, chain_label_2: str, thr: float = 4.5
) -> pd.DataFrame:
    """
    Find contacts between two chains in a PDB file.
    A contact is defined as a pair of residues from the two chains that have at
    least one pair of non-hydrogen atoms within 4.5 angstroms.

    Args:
        atom_df (pd.DataFrame): A DataFrame containing the PDB data.
            Returned by PandasPdb.df["ATOM"]
        chain_label_1 (str): The label of the first chain.
        chain_label_2 (str): The label of the second chain.

    Returns:
        pd.DataFrame: A DataFrame with the minimum distances between each pair
        of residues from the two chains.
    """
    # if node_id is not present, add it
    if "node_id" not in atom_df.columns:
        atom_df = _add_node_id_col(atom_df)

    # assert chain labels are valid
    assert (
        chain_label_1 in atom_df["chain_id"].unique()
    ), f"Invalid chain label {chain_label_1}"
    assert (
        chain_label_2 in atom_df["chain_id"].unique()
    ), f"Invalid chain label {chain_label_2}"

    # Filter atoms by chain label
    chain_1_atoms = atom_df.query(f'chain_id == "{chain_label_1}"')
    chain_2_atoms = atom_df.query(f'chain_id == "{chain_label_2}"')

    # Get coordinates
    coords_1 = chain_1_atoms[["x_coord", "y_coord", "z_coord"]].values
    coords_2 = chain_2_atoms[["x_coord", "y_coord", "z_coord"]].values

    # Calculate distance matrix
    diff = coords_1[:, np.newaxis, :] - coords_2[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    # find distances less than 4.5 angstroms
    atm_idx_1, atm_idx_2 = np.where(dist_matrix < thr)

    # map back to get residue pairs
    res_pairs: pd.DataFrame = (
        pd.DataFrame(
            {
                "atom_1": chain_1_atoms.iloc[atm_idx_1]["node_id"].values,
                "atom_2": chain_2_atoms.iloc[atm_idx_2]["node_id"].values,
                "residue_1": chain_1_atoms.iloc[atm_idx_1]["node_id"]
                .apply(lambda x: x.rsplit(":", 1)[0])
                .values,
                "residue_2": chain_2_atoms.iloc[atm_idx_2]["node_id"]
                .apply(lambda x: x.rsplit(":", 1)[0])
                .values,
                "distance": dist_matrix[atm_idx_1, atm_idx_2],
            }
        )
        .drop_duplicates()
        .sort_values("residue_1")
        .reset_index(drop=True)
    )

    # inlcude only the pair with the minimum distance
    # 1. add columns id_1, id_2 by remove the last two elements from atom_1, atom_2 e.g. H:ASN:54::CB -> H:ASN:54: (DO NOT remove the last colon, it is for insertion)
    # 2. group by id_1, id_2, and get the minimum distance
    res_pairs["id_1"] = res_pairs["residue_1"].str.replace(r":\w+$", "", regex=True)
    res_pairs["id_2"] = res_pairs["residue_2"].str.replace(r":[\w\']+$", "", regex=True)
    res_pairs = res_pairs.groupby(["id_1", "id_2"]).min().reset_index()
    res_pairs.drop(columns=["id_1", "id_2"], inplace=True)

    return res_pairs


def process_atom_df(atom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the atom dataframe"
    - add atom name to node_id e.g. H:PRO:52 -> H:PRO:52::
    - add empty insertion if not present in node_id e.g. H:PRO:52:CA -> H:PRO:52::CA

    Args:
        atom_df (pd.DataFrame): _description_

    Returns:
        None: this function modifies the input DataFrame in place.
    """
    atom_df["node_id"] = (
        atom_df["chain_id"]
        + ":"
        + atom_df["residue_name"]
        + ":"
        + atom_df["residue_number"].astype(str)
        + ":"
        + atom_df["insertion"]
        + ":"
        + atom_df["atom_name"]
    )
    atom_df["residue_id"] = (
        atom_df["chain_id"]
        + ":"
        + atom_df["residue_name"]
        + ":"
        + atom_df["residue_number"].astype(str)
        + ":"
        + atom_df["insertion"]
    )


def filter_dataframe(
    dataframe: pd.DataFrame,
    by_column: str,
    list_of_values: List[Any],
    boolean: bool,
) -> pd.DataFrame:
    """
    Filter function for DataFrame.

    Filters the DataFrame such that the ``by_column`` values have to be
    in the ``list_of_values`` list if ``boolean == True``, or not in the list
    if ``boolean == False``.

    Args:
        dataframe (pd.DataFrame): atomic dataframe
        by_column (str): denoting column of DataFrame to filter.
        list_of_values (List[Any]): List of values to filter with.
        boolean (bool): indicates whether to keep or exclude matching
            ``list_of_values``. ``True`` -> in list, ``False`` -> not in list.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = dataframe.copy()
    df = df[df[by_column].isin(list_of_values) == boolean]
    df.reset_index(inplace=True, drop=True)

    return df


def atom_pairwise_distmat(coords: np.ndarray) -> np.ndarray:
    """
    Calculate the pairwise Euclidean distance between atoms.

    Args:
        coords (np.ndarray): Array of atom coordinates. Shape (n_atoms, 3).

    Raises:
        ValueError: If the input is not a 2D array with 3 columns.

    Returns:
        np.ndarray: Pairwise distance matrix.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "Input must be an Nx3 array where N is the number of points and 3 are their x, y, z coordinates."
        )

    # Compute the difference matrix between each pair of points
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Compute the Euclidean distance using the norm along axis -1
    dist_matrix = np.linalg.norm(diff, axis=-1)

    return dist_matrix


def compute_distmat(atom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Euclidean distances between every atom.

    Design choice: passed in a ``pd.DataFrame`` to enable easier testing on

    Args:
        pdb_df (pd.DataFrame): Dataframe containing protein structure. Must contain columns ``["x_coord", "y_coord", "z_coord"]``.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: pd.Dataframe of Euclidean distance matrix.
    """
    if not pd.Series(["x_coord", "y_coord", "z_coord"]).isin(atom_df.columns).all():
        raise ValueError(
            "Dataframe must contain columns ['x_coord', 'y_coord', 'z_coord']"
        )
    eucl_dists = atom_pairwise_distmat(
        atom_df[["x_coord", "y_coord", "z_coord"]].to_numpy()
    )
    eucl_dists = pd.DataFrame(eucl_dists)
    eucl_dists.index = atom_df.index
    eucl_dists.columns = atom_df.index

    return eucl_dists


def get_interacting_atoms(angstroms: float, distmat: pd.DataFrame) -> np.ndarray:
    """
    Find the atoms that are within a particular radius of one another.

    Args:
        angstroms (float): the radius in angstroms.
        distmat (pd.DataFrame): the distance matrix.

    Returns:
        np.ndarray: The indices of the atoms that are within the radius of one another.
    """
    return np.where(distmat <= angstroms)


def remove_intra_chain_interactions(id_pairs: List[Tuple[str, str]]):
    return sorted(
        [
            (res1, res2)
            for res1, res2 in id_pairs
            if res1.split(":")[0] != res2.split(":")[0]
        ],
        # key=lambda x: (x[0][0].lower() not in ["h", "l"], x[0], x[1]),
    )


def itf_atom_to_resis(itf_df: pd.DataFrame, interacting_atoms: np.ndarray):
    interacting_atom_pairs = [
        (
            itf_df.iloc[i].node_id,
            itf_df.iloc[j].node_id,
        )  # TODO: new node_id add atom name
        for i, j in np.array(interacting_atoms).T
        if i != j
    ]
    interacting_res_pairs = sorted(
        set(
            [
                (itf_df.iloc[i].residue_id, itf_df.iloc[j].residue_id)
                for i, j in np.array(interacting_atoms).T
                if i != j
            ]
        )
    )
    return interacting_atom_pairs, interacting_res_pairs


def get_atomic_interaction_core(
    atom_df: pd.DataFrame, atom_name_set: List[str], dist_thr: float
):
    """adapted from graphein.protein.edges.distance.add_hbond_interaction"""
    itf_df = filter_dataframe(
        dataframe=atom_df,
        by_column="atom_name",
        list_of_values=atom_name_set,
        boolean=True,
    )
    if len(itf_df.index) > 0:
        distmat = compute_distmat(itf_df)
        interacting_atoms = get_interacting_atoms(angstroms=dist_thr, distmat=distmat)

        # convert to interacting resis
        interacting_atom_pairs, interacting_res_pairs = itf_atom_to_resis(
            itf_df, interacting_atoms
        )

        # remove intra-chain interactions
        interacting_res_pairs_nr = remove_intra_chain_interactions(
            interacting_res_pairs
        )
        interacting_atom_pairs_nr = remove_intra_chain_interactions(
            interacting_atom_pairs
        )
        return {
            "inter-chain-res-pairs": interacting_res_pairs_nr,
            "inter-chain-atom-pairs": interacting_atom_pairs_nr,
        }
    print("No interactions found")
    return {"inter-chain-res-pairs": [], "inter-chain-atom-pairs": []}


def get_hydrogen_bonds(
    atom_df: pd.DataFrame,
) -> Dict[str, List[Tuple[str, str]]]:
    DIST_THR = {"common": 3.5, "sulphur": 4.0}
    HBOND_ATOMS = [
        # Protein specific atoms
        "ND",  # histidine and asparagine
        "NE",  # glutamate, tryptophan, arginine, histidine
        "NH",  # arginine
        "NZ",  # lysine
        "OD1",
        "OD2",
        "OE",
        "OG",
        "OH",
        "SD",  # cysteine
        "SG",  # methionine
        "N",
        "O",
        # RNA specific atoms
        "O2'",
        "O3'",
        "O4'",
        "O5'",
        "O2P",
        "O1P",
        "N1",
        "N2",
        "N3",
        "N4",
        "N6",
        "N7",
        "N9",
    ]

    hbond_dict_common = get_atomic_interaction_core(
        atom_df=atom_df, atom_name_set=HBOND_ATOMS, dist_thr=DIST_THR["common"]
    )
    hbond_dict_sulphur = get_atomic_interaction_core(
        atom_df, atom_name_set=["SD", "SG"], dist_thr=DIST_THR["sulphur"]
    )
    # merge the two dict
    hbond_dict = {}
    for k in ["inter-chain-res-pairs", "inter-chain-atom-pairs"]:
        hbond_dict[k] = hbond_dict_common[k] + hbond_dict_sulphur[k]
    # deduplicate: e.g. (A:ASN:54, B:TRP:403) and (B:TRP:403, A:ASN:54) are the same
    hbond_dict["inter-chain-res-pairs"] = deduplicate_res_pairs(
        hbond_dict["inter-chain-res-pairs"]
    )
    hbond_dict["inter-chain-atom-pairs"] = deduplicate_res_pairs(
        hbond_dict["inter-chain-atom-pairs"]
    )
    return hbond_dict


def get_ionic_interactions(
    atom_df: pd.DataFrame,
):
    """
    adapted from graphein.protein.edges.distance.add_ionic_interaction

    NOTE: this function only detects positively and negatively charged residues, and not all ionic interactions e.g. ionic interaction with backbone atoms

    """
    # IONIC_RESIS: List[str] = ["ARG", "LYS", "HIS", "ASP", "GLU", "A", "C", "G", "U"]
    # """Residues and RNA bases capable of forming ionic interactions."""

    # POS_AA: List[str] = ["HIS", "LYS", "ARG", "A", "C", "G", "U"]
    # """Positively charged amino acids and RNA bases."""

    # NEG_AA: List[str] = ["GLU", "ASP", "A", "C", "G", "U"]
    # """Negatively charged amino acids and RNA bases."""

    DIST_THR = 6.0
    IONIC_ATOMS = [
        # Protein specific atoms
        "N",
        "O",
        "ND",  # histidine and asparagine
        "NE",  # glutamate, tryptophan, arginine, histidine
        "NH",  # arginine
        "NZ",  # lysine
        "OD1",
        "OD2",
        "OE",
        "OG",
        "OH",
        "SD",  # cysteine
        "SG",  # methionine
        # Nucleic acid atoms
        "N1",
        "N2",
        "N3",
        "N4",
        "N6",
        "N7",
        "N9",
        "O2",
        "O4",
        "O6",
        "OP1",
        "OP2",
        "O3'",
        "O4'",
        "O5'",
    ]

    # Get ionic interactions
    ionic_dict = get_atomic_interaction_core(
        atom_df=atom_df, atom_name_set=IONIC_ATOMS, dist_thr=DIST_THR
    )

    # deduplicate: e.g. (A:ASN:54, B:TRP:403) and (B:TRP:403, A:ASN:54) are the same
    ionic_dict["inter-chain-res-pairs"] = deduplicate_res_pairs(
        ionic_dict["inter-chain-res-pairs"]
    )
    ionic_dict["inter-chain-atom-pairs"] = deduplicate_res_pairs(
        ionic_dict["inter-chain-atom-pairs"]
    )

    return ionic_dict


# atom_df = parse_pdb_file_as_atom_df("example/4xln.pdb")
# l = get_ionic_interactions(atom_df)
# l['inter-chain-atom-pairs']

def find_contacts_between_two_chains(
    atom_df: pd.DataFrame,
    chain_label_1: str,
    chain_label_2: str,
    thr: float = 4.5,
):
    """
    Main function to calculate the minimum distances between residues in two chains of a PDB file.

    Args:
        atom_df (pd.DataFrame): A DataFrame containing the PDB atomic data.
        chain_label_1 (str): The label of the first chain.
        chain_label_2 (str): The label of the second chain.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the residue pairs and distances.
    """
    # filter atom_df to include only the chains of interest
    atom_df = filter_dataframe(
        atom_df, "chain_id", [chain_label_1, chain_label_2], boolean=True
    )

    # find hydrogen bonds
    hbonds: Dict[str, List[Tuple[str]]] = get_hydrogen_bonds(atom_df)
    print_msg(f"Hydrogen bonds: {hbonds}", "debug")
    # prepare annotation dict
    interaction_annotation_dict = {}
    if hbonds["inter-chain-res-pairs"]:
        # residue pairs with hydrogen bonds
        interaction_annotation_dict = {
            k: ["HBond"] for k in hbonds["inter-chain-res-pairs"]
        }

    # find contacts
    atom_contacts: pd.DataFrame = calculate_min_residue_distances(
        atom_df, chain_label_1, chain_label_2, thr=thr
    )

    return {
        "contacts": atom_contacts,
        "interaction_annotation_dict": interaction_annotation_dict,
        "hbond_atom_pairs": hbonds["inter-chain-atom-pairs"],
    }


def create_chain_pairs(
    receptor_chain_labels: List[str], ligand_chain_labels: List[str]
):
    """
    Get all possible chain pairs.
    e.g. receptor_chain_labels = ["H", "L"], ligand_chain_labels = ["A", "B"]
    chain_pairs = [("H", "A"), ("H", "B"), ("L", "A"), ("L", "B")]

    Args:
        receptor_chain_labels (List[str]): Receptor chain labels.
        ligand_chain_labels (List[str]): Ligand chain labels.

    Returns:
        List[Tuple[str, str]]: List of chain pairs.
    """
    chain_pairs: List[Tuple[str, str]] = sorted(
        set(
            sorted(
                tuple([r, l])
                for r in receptor_chain_labels
                for l in ligand_chain_labels
                if r != l
            )
        )
    )
    return chain_pairs


def create_chain_pairs_one_to_multi(
    query_chain_label: str, target_chain_labels: List[str]
):
    """
    Get all possible chain pairs.
    e.g. query_chain_label = "H", target_chain_labels = ["A", "B"]
    chain_pairs = [("H", "A"), ("H", "B")]

    Args:
        query_chain_label (str): Query chain label.
        target_chain_labels (List[str]): Target chain labels.

    Returns:
        List[Tuple[str, str]]: List of chain pairs.
    """
    chain_pairs: List[Tuple[str, str]] = sorted(
        set(
            tuple(sorted([query_chain_label, t]))
            for t in target_chain_labels
            if query_chain_label != t
        )
    )
    return chain_pairs


def create_txt_content(
    chain_labels_1: List[str],
    chain_labels_2: List[str],
    atom_df: pd.DataFrame,
    interface_residues_by_chain: Dict[str, List[str]],
    step: int = 50,
) -> str:
    # round step up to the nearest 10
    step = 10 * math.ceil(step / 10)
    content = "Interface residues:\n"
    for chain_label in chain_labels_1 + chain_labels_2:
        print_msg(
            f"Interface residues for chain {chain_label}: {interface_residues_by_chain[chain_label]}",
            "debug",
        )
        # e.g. chain D
        content += f"Chain: {chain_label}\n"
        # write its sequence read from atom_df
        three_letter_list = (
            atom_df.query(f'chain_id == "{chain_label}"')
            .drop_duplicates("residue_id")
            .residue_name.to_list()
        )
        if set(three_letter_list) < {"A", "C", "U", "G", "T"}:
            # RNA
            if "U" in three_letter_list:
                chain_type = "RNA"
            elif "T" in three_letter_list:
                chain_type = "DNA"
            else:
                chain_type = "Nucleic"
        else:
            chain_type = "Protein"
        content += f"Type: {chain_type}\n"
        # add context: i.e. comparing chain set 1 and chain set 2, so that
        # the user knows which chain set is being compared
        # chain pairs that involve chain_label
        if chain_label in chain_labels_1:
            chain_pairs = create_chain_pairs_one_to_multi(chain_label, chain_labels_2)
        else:
            chain_pairs = create_chain_pairs_one_to_multi(chain_label, chain_labels_1)
        # e.g. chain_pairs: (A, D), (B, D), (C, D), (D, E), (D, F), (D, G), (D, H), (D, I)
        # convert to (D, ABCDEFGHI)
        other_chains = "".join([i for p in chain_pairs for i in p if i != chain_label])
        content += f"Context: ({chain_label}, {other_chains})\n"

        # Add sequence
        if chain_type == "Protein":
            aa1_list = [three_to_one(aa3) for aa3 in three_letter_list]
            # seq
            seq = "".join(aa1_list)
        else:
            seq = "".join(three_letter_list)

        # Add interface mask, create a binary mask for interface residues
        chain_df = (
            atom_df.query(f'chain_id == "{chain_label}"')
            .drop_duplicates("residue_id")
            .reset_index(drop=True)
        )
        # got indices of interface residues
        interface_res_indices = chain_df.query(
            "residue_id in @interface_residues_by_chain[@chain_label]"
        ).index
        # create a binary mask
        mask = np.zeros(len(chain_df), dtype=bool)
        mask[interface_res_indices] = True
        # convert False to '-' and True to '*'
        mask_str = "".join(["*" if m else "-" for m in mask])
        # add interface residue count
        content += f"Interface: {mask.sum():<3} residues\n"

        # add to the content wrap e.g. 100 characters a line
        # add a line above to show 10, 20, ... 100 using 1 - 9, 0
        for i in range(0, len(seq), step):
            s = " ".join(textwrap.wrap(seq[i : i + step], 10))
            j = " ".join(textwrap.wrap(mask_str[i : i + step], 10))
            # if not the last step
            if i + step < len(seq):
                content += f"Sequence : {i+1:>4}  {s}  {i+1+step:<4}\n"
                content += f"Interface: {i+1:>4}  {j}  {i+1+step:<4}\n"
            else:
                content += f"Sequence : {i+1:>4}  {s}  {len(seq):<4}\n"
                content += f"Interface: {i+1:>4}  {j}  {len(seq):<4}\n"

        # add empty line
        content += "\n\n"

    return content


# ----------------------------------------
# PyMOL
# ----------------------------------------
# TODO 1: clean this function
# TODO 2: add function to select ionic interactions
def select_interface(
    selection: str = "all",
    chain_labels_1: Optional[Union[str, List[str]]] = None,
    chain_labels_2: Optional[Union[str, List[str]]] = None,
    dist_cutoff: Union[float, str] = 4.5,
    save_results: bool = False,
    pdb_file: str = None,
) -> None:
    """
    Find the interface residues between two chain sets in a PDB file and create selections in PyMOL.

    Args:
        selection (str): a selection-expression e.g. chain A and resi 1-10
        pdb_file (str, optional): Path to the corresponding pdb file. Defaults to None.
            If None, the object_name will be used to save a temporary pdb file.
        chain_labels_1 (str, optional): chain label(s) of the first chain set.
            Acceptable values: None, "*", "A", "AB", ["A", "B"]
            Defaults to None.
            None and "*" will be converted to all chain labels in the structure.
        chain_labels_2 (str, optional): chain label(s) of the second chain set.
            Acceptable values: None, "*", "A", "AB", ["A", "B"]
            Defaults to None.
            None and "*" will be converted to all chain labels in the structure.
        dist_cutoff (Union[float, str], optional): distance threshold.
            Defaults to 4.5.
        save_results (bool, optional): Whether to save the results to file(s).
            Defaults to False.

    Returns:
        None
    """
    # correct types
    pdb_file = Path(pdb_file) if pdb_file else None
    chain_labels_1 = list(chain_labels_1) if chain_labels_1 else None
    chain_labels_2 = list(chain_labels_2) if chain_labels_2 else None
    dist_cutoff = float(dist_cutoff) if dist_cutoff else 4.5
    # save a temporary pdb file if pdb_file is None
    if not pdb_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            cmd.save(
                filename=f"{temp_dir}/{selection}.pdb",
                selection=selection,
                format="pdb",
            )
            pdb_file = Path(f"{temp_dir}/{selection}.pdb")
            atom_df = parse_pdb_file_as_atom_df(pdb_file)
    else:
        atom_df = parse_pdb_file_as_atom_df(pdb_file)

    # all chain labels
    chain_labels = atom_df["chain_id"].unique().tolist()
    print_msg(
        f"All chain labels in the structure: {chain_labels}({type(chain_labels)})",
        "debug",
    )

    # if neither receptor_chain_labels nor ligand_chain_labels are provided, do it pairwise
    if not chain_labels_1 or chain_labels_1 == ["*"]:
        chain_labels_1 = chain_labels
    if not chain_labels_2 or chain_labels_2 == ["*"]:
        chain_labels_2 = chain_labels
    chain_pairs = create_chain_pairs(chain_labels_1, chain_labels_2)
    print_msg(f"Chain pairs: {chain_pairs}", "info")

    # iterate over all chain pairs
    final_df_residue = pd.DataFrame()
    final_df_atom = pd.DataFrame()
    for chain_label_1, chain_label_2 in chain_pairs:
        with timing_context(
            f"Find contacts between {chain_label_1} and {chain_label_2}"
        ):
            results = find_contacts_between_two_chains(
                atom_df=atom_df,
                chain_label_1=chain_label_1,
                chain_label_2=chain_label_2,
                thr=dist_cutoff,
            )
            print_msg(f"Results:\n{results}", "debug")
            # residue pairs with hydrogen bonds to csv
            contacts_csv = to_csv(
                contacts_df=results["contacts"],
                interaction_annotation_dict=results["interaction_annotation_dict"],
            )
            print_msg(f"Contacts CSV:\n{contacts_csv}", "debug")
            # atom pairs with hydrogen bonds to csv
            data = {}
            data["atom_1"] = [x[0] for x in results["hbond_atom_pairs"]]
            data["atom_2"] = [x[1] for x in results["hbond_atom_pairs"]]
            data["chain_1"] = [x.split(":")[0] for x in data["atom_1"]]
            data["chain_2"] = [x.split(":")[0] for x in data["atom_2"]]
            final_df_atom = pd.concat([final_df_atom, pd.DataFrame(data)])
            # add to the final dataframe
            final_df_residue = pd.concat([final_df_residue, contacts_csv])

    print_msg(f"Final dataframe: {final_df_residue}", "debug")

    # [x] TODO: Create selection in pymol by chain
    # e.g. selName = f"{c1}{c2}-{c1}" means select interface residues between chain c1 and c2 in chain c1
    for c1, c2 in chain_pairs:
        # query interface residues between D and E from final_df
        sub_df_residue = final_df_residue.query(f"chain_1=='{c1}' & chain_2=='{c2}'")
        sub_df_atom = final_df_atom.query(f"chain_1=='{c1}' & chain_2=='{c2}'")

        def _create_selection_for_one_chain(
            c: str, chain1: str, chain2: str, col_name: str
        ):
            assert col_name in ["residue_1", "residue_2"]
            # create selection string
            sel_name = f"interface-{chain1}{chain2}-{c}"
            sel_str = []
            for i in sub_df_residue[col_name]:
                chain, _, resi, insert = i.split(":")
                sel_str.append(f"(c. {chain} and i. {resi}{insert})")
            sel_str = " or ".join(sel_str)
            # if empty selection string, return None
            if not sel_str:
                return None, None
            sel_str = f"{selection} and ({sel_str})"
            return sel_name, sel_str

        sel_name_1, sel_str_1 = _create_selection_for_one_chain(
            c=c1, chain1=c1, chain2=c2, col_name="residue_1"
        )
        if sel_name_1 and sel_str_1:
            cmd.delete(sel_name_1)
            cmd.select(sel_name_1, sel_str_1)
        else:
            print_msg(f"No interface residues found between {c1} and {c2}", "info")

        sel_name_2, sel_str_2 = _create_selection_for_one_chain(
            c=c2, chain1=c1, chain2=c2, col_name="residue_2"
        )
        if sel_name_2 and sel_str_2:
            cmd.delete(sel_name_2)
            cmd.select(sel_name_2, sel_str_2)
        else:
            print_msg(f"No interface residues found between {c1} and {c2}", "info")

        sel_name_1_2 = f"interface-{c1}{c2}"
        if sel_name_1 and sel_name_2:
            cmd.delete(sel_name_1_2)
            cmd.select(sel_name_1_2, f"{sel_name_1} or {sel_name_2}")

        # HBond: create a group of residue pairs if sub_df_residue is not empty
        sel_name_1_2_hbond = f"{sel_name_1_2}-HBond"
        sub_df_hbond = sub_df_residue.query("interaction == 'HBond'")
        sels = []
        if not sub_df_hbond.empty:
            sel_str_hbond = []
            # iterate over the rows of sub_df_hbond
            n = 1
            for _, row in sub_df_hbond.iterrows():
                # get the residue names and numbers
                res1, res2 = row["residue_1"], row["residue_2"]
                # unpack the residue names and numbers
                chain1, _, resi1, ins1 = res1.split(":")
                chain2, _, resi2, ins2 = res2.split(":")
                # residue pair selection string
                sel_name_i = f"{sel_name_1_2}-HB{n}"
                sel_str_i = f"{selection} and ((c. {chain1} and i. {resi1}{ins1}) or (c. {chain2} and i. {resi2}{ins2}))"
                cmd.select(sel_name_i, sel_str_i)
                sels.append(sel_name_i)
                n += 1
        # delete the group if it exists
        cmd.delete(sel_name_1_2_hbond)
        # group all selections
        cmd.group(sel_name_1_2_hbond, " ".join(sels))

        # measure distances
        sel_dist = []
        for _, row in sub_df_atom.iterrows():
            atom_1, atom_2 = row["atom_1"], row["atom_2"]
            c1, _, r1, i1, a1 = atom_1.split(":")
            c2, _, r2, i2, a2 = atom_2.split(":")
            l = f"{c1}_{r1}{i1}_{a1}-{c2}_{r2}{i2}_{a2}"
            cmd.distance(
                name=l,
                selection1=f"{selection} and (c. {c1} and i. {r1}{i1} and n. {a1})",
                selection2=f"{selection} and (c. {c2} and i. {r2}{i2} and n. {a2})",
            )
            sel_dist.append(l)
        # group all distances
        cmd.delete(f"{sel_name_1_2}-hbond-distance")
        cmd.group(f"{sel_name_1_2}-hbond-distance", " ".join(sel_dist))

    # save output
    if save_results:
        # csv
        out_fp = Path(f"{pdb_file.stem}_contacts.csv")
        final_df_residue.to_csv(out_fp, index=False)
        # txt
        out_fp = Path(f"{pdb_file.stem}_interface_residues.txt")
        interface_residues = (
            final_df_residue["residue_1"].to_list()
            + (final_df_residue["residue_2"]).to_list()
        )
        included_chain_labels = set(chain_labels_1 + chain_labels_2)
        print_msg(f"Interface residues: {interface_residues}", "debug")
        interface_residues_by_chain = {
            c: [r for r in interface_residues if r.startswith(c)]
            for c in included_chain_labels
        }
        print_msg(
            f"Interface residues by chain: {interface_residues_by_chain}", "debug"
        )
        content = create_txt_content(
            chain_labels_1=chain_labels_1,
            chain_labels_2=chain_labels_2,
            atom_df=atom_df,
            interface_residues_by_chain=interface_residues_by_chain,
        )
        with open(out_fp, "w") as f:
            f.write(content)

    # Print Done
    print_msg("Done", "info")


cmd.extend("cl_select_interface", select_interface)


# ----------------------------------------
# Deprecated
# ----------------------------------------
def main(args):
    # create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # process input structure
    # if mmcif, convert to pdb
    if args.struct_fp.suffix == ".cif":
        with timing_context("Convert mmCIF to PDB"):
            pdb_fp = args.output_dir / f"{args.struct_fp.stem}.pdb"
            convert_mmcif_to_pdb(args.struct_fp, pdb_fp)
            args.struct_fp = pdb_fp

    # parse the structure file as a DataFrame
    atom_df: pd.DataFrame = parse_pdb_file_as_atom_df(args.struct_fp)

    # all chain labels
    chain_labels = atom_df["chain_id"].unique().tolist()
    print_msg(
        f"All chain labels in the structure: {chain_labels}({type(chain_labels)})",
        "debug",
    )

    # if neither receptor_chain_labels nor ligand_chain_labels are provided, do it pairwise
    receptor_chain_labels = args.receptor_chain_labels
    ligand_chain_labels = args.ligand_chain_labels
    if not receptor_chain_labels and not ligand_chain_labels:
        print_msg("No chain labels provided. Use all possible chain pairs.", "info")
        # set the receptor and ligand chain labels to all possible combinations
        receptor_chain_labels = chain_labels
        ligand_chain_labels = chain_labels
        print_msg(f"Receptor chain labels: {receptor_chain_labels}", "info")
        print_msg(f"Ligand chain labels: {ligand_chain_labels}", "info")

    with timing_context("Create chain pairs"):
        chain_pairs: List[Tuple[str, str]] = create_chain_pairs(
            receptor_chain_labels, ligand_chain_labels
        )
    print_msg(f"Chain pairs: {chain_pairs}", "info")

    # iterate over all chain pairs
    final_df = pd.DataFrame()
    for chain_label_1, chain_label_2 in chain_pairs:
        with timing_context(
            f"Find contacts between {chain_label_1} and {chain_label_2}"
        ):
            resluts = find_contacts_between_two_chains(
                atom_df=atom_df,
                chain_label_1=chain_label_1,
                chain_label_2=chain_label_2,
            )
            contacts_csv = to_csv(
                contacts_df=resluts["contacts"],
                interaction_annotation_dict=resluts["interaction_annotation_dict"],
            )
            # add to the final dataframe
            final_df = pd.concat([final_df, contacts_csv])

    # save to csv
    out_fp = args.output_dir / f"{args.struct_fp.stem}_contacts.csv"
    with timing_context(f"Save results to {out_fp}"):
        final_df.to_csv(out_fp, index=False)

    # ----------------------------------------
    # create a txt file to record all interface
    # residues for each chain
    # ----------------------------------------
    out_fp = args.output_dir / f"{args.struct_fp.stem}_interface_residues.txt"
    # interface residues
    interface_residues = (
        final_df["residue_1"].to_list() + (final_df["residue_2"]).to_list()
    )
    # all chains
    input_chain_labels = set(receptor_chain_labels + ligand_chain_labels)
    print_msg(f"Input chain labels: {input_chain_labels}", "debug")
    # get interface residues by chain
    interface_residues_by_chain = {
        c: [r for r in interface_residues if r.startswith(c)]
        for c in input_chain_labels
    }
    print_msg(f"Interface residues by chain: {interface_residues_by_chain}", "debug")
    # crate file content
    content = "Interface residues:\n"
    for chain_label in input_chain_labels:
        print_msg(
            f"Interface residues for chain {chain_label}: {interface_residues_by_chain[chain_label]}",
            "debug",
        )
        # e.g. chain D
        content += f"Chain: {chain_label}\n"
        # write its sequence read from atom_df
        three_letter_list = (
            atom_df.query(f'chain_id == "{chain_label}"')
            .drop_duplicates("residue_id")
            .residue_name.to_list()
        )
        if set(three_letter_list) < {"A", "C", "U", "G", "T"}:
            # RNA
            if "U" in three_letter_list:
                chain_type = "RNA"
            elif "T" in three_letter_list:
                chain_type = "DNA"
            else:
                chain_type = "Nucleic"
        else:
            chain_type = "Protein"
        content += f"Type: {chain_type}\n"

        # Add sequence
        if chain_type == "Protein":
            aa1_list = [three_to_one(aa3) for aa3 in three_letter_list]
            # seq
            seq = "".join(aa1_list)
        else:
            seq = "".join(three_letter_list)

        # Add interface mask
        # create a binary mask for interface residues
        chain_df = (
            atom_df.query(f'chain_id == "{chain_label}"')
            .drop_duplicates("residue_id")
            .reset_index(drop=True)
        )
        # got indices of interface residues
        interface_res_indices = chain_df.query(
            "residue_id in @interface_residues_by_chain[@chain_label]"
        ).index
        # create a binary mask
        mask = np.zeros(len(chain_df), dtype=bool)
        mask[interface_res_indices] = True
        # convert Flase to '-' and True to '*'
        mask_str = "".join(["*" if m else "-" for m in mask])

        # add to the content wrap 100 characters a line
        # add a line above to show 10, 20, ... 100 using 1 - 9, 0
        for i in range(0, len(seq), 100):
            content += f"Sequence : {i+1:>4} {seq[i:i+100]}\n"
            content += f"Interface: {i+1:>4} {mask_str[i:i+100]}\n"
            content += "\n"
        # add separator
        content += "\n\n"
    with open(out_fp, "w") as f:
        f.write(content)
