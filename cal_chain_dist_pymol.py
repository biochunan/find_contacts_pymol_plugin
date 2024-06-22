"""
Reference:
- graphein.protein.utils.filter_dataframe
- graphein.protein.edges.distance.compute_distmat
"""

# basic
import copy
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
# Atom Names
# ----------------------------------------
# For RNA nucleotides (A, C, G, U):
RNA_RES_NAMES = ["A", "C", "G", "U"]
DNA_RES_NAMES = ["DA", "DC", "DG", "DT"]
# nucleotide base atoms
BASE_ATOM_NAMES = {
    "A": [
        "N1",
        "C2",
        "N3",
        "C4",
        "C5",
        "C6",  # adenine ring
        "N6",  # amino group off the C6 ring
        "N7",
        "C8",
        "N9",  # additional ring
    ],
    "G": [
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
        "C5",
        "C6",  # guanine ring
        "O6",  # oxygen atom on the C6 carbon
        "N7",
        "C8",
        "N9",  # additional ring
    ],
    "C": [
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "C5",
        "C6",  # cytosine ring
    ],
    "T": [
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",  # similar to cytosine, but with an additional methyl group
    ],
    "U": [
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C6",  # Similar to thymine but without the methyl group.
    ],
}
# backbone atoms
DNA_ATOM_BD_NAMES = [
    # Phosphate group
    "P",
    "OP1",
    "OP2",
    "OP3",
    # Sugar group
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",  # sugar carbon
    # oxygen in the sugar
    "O3'",  # O3' connects to the next nucleotide's phosphate group
    "O4'",
    "O5'",  # O5' connects the sugar to the phosphate group
    # hydrogen atoms connect to the sugar
    "H1'",
    "H2'",
    "H2''",
    "H3'",
    "H4'",
    "H5'",
    "H5''",
]
RNA_ATOM_BD_NAMES = copy.deepcopy(DNA_ATOM_BD_NAMES) + ["O2'"]
# The key difference is DNA lacks the O2' atom on the sugar
ADENOSINE_ATOM_NAMES = copy.deepcopy(BASE_ATOM_NAMES["A"]) + copy.deepcopy(
    RNA_ATOM_BD_NAMES
)
GUANOSINE_ATOM_NAMES = copy.deepcopy(BASE_ATOM_NAMES["G"]) + copy.deepcopy(
    RNA_ATOM_BD_NAMES
)
CYTIDINE_ATOM_NAMES = copy.deepcopy(BASE_ATOM_NAMES["C"]) + copy.deepcopy(
    RNA_ATOM_BD_NAMES
)
THYMIDINE_ATOM_NAMES = copy.deepcopy(BASE_ATOM_NAMES["T"]) + copy.deepcopy(
    DNA_ATOM_BD_NAMES
)
URIDINE_ATOM_NAMES = copy.deepcopy(BASE_ATOM_NAMES["U"]) + copy.deepcopy(
    RNA_ATOM_BD_NAMES
)
# dna and rna all atom names
DNA_ATOM_NAMES = set(
    ADENOSINE_ATOM_NAMES
    + GUANOSINE_ATOM_NAMES
    + CYTIDINE_ATOM_NAMES
    + THYMIDINE_ATOM_NAMES
)
RNA_ATOM_NAMES = set(
    ADENOSINE_ATOM_NAMES
    + GUANOSINE_ATOM_NAMES
    + CYTIDINE_ATOM_NAMES
    + URIDINE_ATOM_NAMES
)

LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
MIN_LOG_LEVEL = "INFO"

# ----------------------------------------
# General utils
# ----------------------------------------
def timestamp() -> str:
    # create time stamp in format 2024May21-123045
    return time.strftime("%Y%b%d-%H%M%S", time.localtime())


def print_msg(msg: str, level: str = "INFO", min_level: str = MIN_LOG_LEVEL):
    level = level.upper()
    assert level in ["INFO", "ERROR", "WARNING", "DEBUG"]
    if LOG_LEVELS[level] >= LOG_LEVELS[min_level]:
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
        print_msg(f"{label} ended. Execution time: {mins:.0f}m {secs:.2f}s", "INFO")


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
    for _, row in contacts_df.iterrows():
        c1, r1, ri1, ins1 = row["residue_1"].split(":")
        c2, r2, ri2, ins2 = row["residue_2"].split(":")

        # map the interaction type
        # Choices: "HBond", "Polar", "Contact"
        # NOTE: in future, need to handle multiple interaction types, use comma may be difficult for frontend to filter
        it_str = ",".join(
            itfd.get(
                (
                    f"{c1}:{r1}:{ri1}:{ins1}",
                    f"{c2}:{r2}:{ri2}:{ins2}",
                ),
                ["Contact"],
            )
        )

        # add to the output
        out["residue_1"].append(f"{c1}:{r1}:{ri1}:{ins1}")
        out["residue_2"].append(f"{c2}:{r2}:{ri2}:{ins2}")
        out["chain_1"].append(c1)
        out["chain_2"].append(c2)
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
        pd.DataFrame: A DataFrame with the minimum atom-pair distances between
            each pair of residues from the two chains.
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

    # include only the pair with the minimum distance
    res_pairs = res_pairs.groupby(["residue_1", "residue_2"]).min().reset_index()
    res_pairs.drop(columns=["atom_1", "atom_2"], inplace=True)

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


def itf_atom_to_res(itf_df: pd.DataFrame, interacting_atoms: np.ndarray):
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
    atom_df: pd.DataFrame,
    by_column: str,
    list_of_values: List[Any],
    dist_thr: float = 4.5
):
    """adapted from graphein.protein.edges.distance.add_hbond_interaction"""
    itf_df = filter_dataframe(
        dataframe=atom_df,
        by_column=by_column,
        list_of_values=list_of_values,
        boolean=True,
    )
    if len(itf_df.index) > 0:
        with timing_context("Computing distance matrix"):
            distmat = compute_distmat(itf_df)
        interacting_atoms = get_interacting_atoms(angstroms=dist_thr, distmat=distmat)

        # convert to interacting residue ids
        interacting_atom_pairs, interacting_res_pairs = itf_atom_to_res(
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


def fix_chain_order(
    payload: List[Tuple[str, str]],
    chain_label_1: str,
    chain_label_2: str,
):
    """
    Check and fix the order of the chain labels in the residue pairs.
    Payload is a list of dictionary with keys as residue pairs.
    Example, payload = [{"A:ASN:21:": xxx}, {"I:G:257:": xxx}]
    if chain_label_1 = "I", chain_label_2 = "A"
    then the order of the keys should be reversed.

    Args:
        payload (List[Dict[Tuple[str, str], Any]]): A list of dictionaries in which the keys are a tuple of residue identifiers.
            The key format e.g. ("A:ASN:21:", "I:G:257:")
            where A is the first chain label, I is the second chain label
        chain_label_1 (str): chain label 1.
        chain_label_2 (str): chain label 2.
    """
    # NOTE: the changes made are in-place
    for i, t in enumerate(payload):
        # i => 0 to len(payload) - 1
        # t => e.g. ('A:ASN:21:', 'I:G:257:') or ('A:ASN:21::0D1', "I:G:257::O5'")
        c1, c2 = t[0][0], t[1][0]
        if c1 == chain_label_1 and c2 == chain_label_2:
            continue
        elif c1 == chain_label_2 and c2 == chain_label_1:
            payload[i] = (t[1], t[0])
        else:
            raise ValueError(
                f"Provided chain labels {chain_label_1, chain_label_2} were not found in the residue pair {t}"
            )


def get_hydrogen_bonds(
    atom_df: pd.DataFrame,
    chain_label_1: str,
    chain_label_2: str,
) -> Dict[str, List[Tuple[str, str]]]:
    dist_thr = {"common": 3.5, "sulphur": 4.0}
    hbond_atoms = [
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
        atom_df=atom_df,
        by_column="atom_name",
        list_of_values=hbond_atoms,
        dist_thr=dist_thr["common"],
    )
    hbond_dict_sulphur = get_atomic_interaction_core(
        atom_df=atom_df,
        by_column="atom_name",
        list_of_values=["SD", "SG"],
        dist_thr=dist_thr["sulphur"],
    )
    # merge the two dict
    hbond_dict = {}
    for k in ["inter-chain-res-pairs", "inter-chain-atom-pairs"]:
        hbond_dict[k] = hbond_dict_common[k] + hbond_dict_sulphur[k]

    # [x] FIXME: add arg to force the order of the keys
    # e.g. key ('A:ASN:21:', 'I:G:257:') should be ('I:G:257:', 'A:ASN:21:') corresponding to chain_label_1='I', chain_label_2='A'
    # deduplicate: e.g. (A:ASN:54, B:TRP:403) and (B:TRP:403, A:ASN:54) are the same
    hbond_dict["inter-chain-res-pairs"] = deduplicate_res_pairs(
        hbond_dict["inter-chain-res-pairs"]
    )
    hbond_dict["inter-chain-atom-pairs"] = deduplicate_res_pairs(
        hbond_dict["inter-chain-atom-pairs"]
    )

    # fix chain order in keys
    fix_chain_order(
        payload=hbond_dict["inter-chain-res-pairs"],
        chain_label_1=chain_label_1,
        chain_label_2=chain_label_2
    )
    fix_chain_order(
        payload=hbond_dict["inter-chain-atom-pairs"],
        chain_label_1=chain_label_1,
        chain_label_2=chain_label_2
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

    dist_thr = 6.0
    ionic_atoms = [
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
        atom_df=atom_df,
        by_column="atom_name",
        list_of_values=ionic_atoms,
        dist_thr=dist_thr,
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


def get_polar_contacts(
    atom_df: pd.DataFrame,
    chain_label_1: str,
    chain_label_2: str,
):
    """
    adapted from graphein.protein.edges.distance.add_ionic_interaction

    NOTE: this function only detects positively and negatively charged residues, and not all ionic interactions e.g. ionic interaction with backbone atoms

    """
    dist_thr = 3.6
    elements = ["N", "O", "S"]
    # Get ionic interactions
    ct_dict = get_atomic_interaction_core(
        atom_df=atom_df,
        by_column="element_symbol",
        list_of_values=elements,
        dist_thr=dist_thr,
    )

    # deduplicate: e.g. (A:ASN:54, B:TRP:403) and (B:TRP:403, A:ASN:54) are the same
    ct_dict["inter-chain-res-pairs"] = deduplicate_res_pairs(
        ct_dict["inter-chain-res-pairs"]
    )
    ct_dict["inter-chain-atom-pairs"] = deduplicate_res_pairs(
        ct_dict["inter-chain-atom-pairs"]
    )

    # fix chain order in keys
    fix_chain_order(
        payload=ct_dict["inter-chain-res-pairs"],
        chain_label_1=chain_label_1,
        chain_label_2=chain_label_2
    )
    fix_chain_order(
        payload=ct_dict["inter-chain-atom-pairs"],
        chain_label_1=chain_label_1,
        chain_label_2=chain_label_2
    )

    return ct_dict


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
        dataframe=atom_df,
        by_column="chain_id",
        list_of_values=[chain_label_1, chain_label_2],
        boolean=True
    )

    # --------------------
    # contact by type
    # --------------------
    # find hydrogen bonds
    with timing_context(f"get_hydrogen_bonds chains {chain_label_1} and {chain_label_2}"):
        hbonds: Dict[str, List[Tuple[str]]] = get_hydrogen_bonds(atom_df=atom_df, chain_label_1=chain_label_1, chain_label_2=chain_label_2)

    # find polar contacts
    with timing_context(f"get_polar_contacts chains {chain_label_1} and {chain_label_2}"):
        polar_contacts: Dict[str, List[Tuple[str]]] = get_polar_contacts(atom_df=atom_df, chain_label_1=chain_label_1, chain_label_2=chain_label_2)

    # --------------------
    # Annotation dict
    # --------------------
    att_dict = {}
    if hbonds["inter-chain-res-pairs"]:
        # hydrogen bonds
        for k in hbonds["inter-chain-res-pairs"]:
            att_dict[k] = att_dict.get(k, []) + ["HBond"]
    if polar_contacts["inter-chain-res-pairs"]:
        # polar contacts
        for k in polar_contacts["inter-chain-res-pairs"]:
            att_dict[k] = att_dict.get(k, []) + ["Polar"]

    # --------------------
    # Annotate contacts
    # --------------------
    min_dist_res_contacts: pd.DataFrame = calculate_min_residue_distances(
        atom_df=atom_df,
        chain_label_1=chain_label_1,
        chain_label_2=chain_label_2,
        thr=thr,
    )

    return {
        "contacts": min_dist_res_contacts,
        "interaction_annotation_dict": att_dict,
        "hbond_atom_pairs": hbonds["inter-chain-atom-pairs"],
        "polar_atom_pairs": polar_contacts["inter-chain-atom-pairs"],
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
    # ----------------------------------------
    # process args
    # ----------------------------------------
    pdb_file = Path(pdb_file) if pdb_file else None
    chain_labels_1 = list(chain_labels_1) if chain_labels_1 else None
    chain_labels_2 = list(chain_labels_2) if chain_labels_2 else None
    dist_cutoff = float(dist_cutoff) if dist_cutoff else 4.5

    # ----------------------------------------
    # Load structure as df
    # ----------------------------------------
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

    """atom_df row example
      record_name  atom_number blank_1 atom_name alt_loc residue_name blank_2
    0        ATOM            1                 N                  MET
    1        ATOM            2                CA                  MET

    chain_id  residue_number insertion blank_3  x_coord  y_coord  z_coord
           A               1                    -11.193   46.130  -32.851
           A               1                     -9.983   45.956  -33.682

    occupancy  b_factor blank_4 segment_id element_symbol  charge  line_idx
          1.0     27.40                                 N     NaN         0
          1.0     32.01                                 C     NaN         1

        node_id residue_id
     A:MET:1::N   A:MET:1:
    A:MET:1::CA   A:MET:1:
    """

    # ----------------------------------------
    # Create chain pairs
    # ----------------------------------------
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

    # ----------------------------------------
    # Find contacts
    # ----------------------------------------
    # iterate over all chain pairs
    final_df_residue = pd.DataFrame()
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
            # residue pairs with hydrogen bonds to csv
            contacts_csv = to_csv(
                contacts_df=results["contacts"],
                interaction_annotation_dict=results["interaction_annotation_dict"],
            )
            final_df_residue = pd.concat([final_df_residue, contacts_csv])

    # ----------------------------------------
    # Create selections in PyMOL
    # ----------------------------------------
    # e.g. selName = f"{c1}{c2}-{c1}" means select interface residues between chain c1 and c2 in chain c1
    for chain_1, chain_2 in chain_pairs:
        # query interface residues between D and E from final_df
        sub_df_residue = final_df_residue.query(f"chain_1=='{chain_1}' & chain_2=='{chain_2}'")

        # --------------------
        # Select interface
        # --------------------
        def _create_selection_for_one_chain(
            c: str, chain1: str, chain2: str, col_name: str
        ):
            """
            Create selection for one chain between chain1 and chain2.
            e.g. interface-DG-D: select interface residues between D and G in chain D
            """
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
            c=chain_1, chain1=chain_1, chain2=chain_2, col_name="residue_1"
        )
        if sel_name_1 and sel_str_1:
            cmd.delete(sel_name_1)
            cmd.select(sel_name_1, sel_str_1)
        else:
            print_msg(f"No interface residues found between {chain_1} and {chain_2}", "info")

        sel_name_2, sel_str_2 = _create_selection_for_one_chain(
            c=chain_2, chain1=chain_1, chain2=chain_2, col_name="residue_2"
        )
        if sel_name_2 and sel_str_2:
            cmd.delete(name=sel_name_2)
            cmd.select(name=sel_name_2, selection=sel_str_2)
        else:
            print_msg(f"No interface residues found between {chain_1} and {chain_2}", "info")

        sel_name_1_2 = f"interface-{chain_1}{chain_2}"
        if sel_name_1 and sel_name_2:
            cmd.delete(name=sel_name_1_2)
            cmd.select(name=sel_name_1_2, selection=f"{sel_name_1} or {sel_name_2}")

        # --------------------
        # HBond res pairs
        # --------------------
        # extract as a function
        def create_selection_str(parent_df: pd.DataFrame, interaction_type: str, obj_name: str, prefix: str) -> List[str]:
            """ Create selections for a given interaction type e.g. 'HBond', 'Polar'. """
            sub_df = parent_df[parent_df.interaction.str.contains(interaction_type, na=False)]  # get rows with interaction_type e.g. HBond, Polar
            selections = []  # output list of selections
            if not sub_df.empty:
                for _, row in sub_df.iterrows():
                    r1, r2 = row["residue_1"], row["residue_2"]
                    c1, _, ri1, ins1 = r1.split(":")
                    c2, _, ri2, ins2 = r2.split(":")
                    name_i = f"{prefix}-{interaction_type}-{c1}.{ri1}{ins1}-{c2}.{ri2}{ins2}"
                    sel_i = f"{obj_name} and ((c. {c1} and i. {ri1}{ins1}) or (c. {c2} and i. {ri2}{ins2}))"
                    cmd.select(name=name_i, selection=sel_i)
                    selections.append(name_i)
            return selections

        # HBond: create a group of residue pairs if sub_df_residue is not empty
        selections = create_selection_str(
            parent_df=sub_df_residue,
            interaction_type="HBond",
            obj_name=selection,
            prefix=sel_name_1_2
        )
        if selections:
            n = f"{sel_name_1_2}-HBond"
            cmd.delete(name=n)
            cmd.group(name=n, members=" ".join(selections))

        # --------------------
        # HBond distance
        # --------------------
        def create_dist_selection_str(atom_pair_list: List[Tuple[str, str]], obj_name: str, prefix: str=None) -> List[str]:
            """ Create distance selections for given atom pairs type e.g. 'hbond_atom_pairs', 'polar_atom_pairs'."""
            dist_sel = []
            for (n1, n2) in atom_pair_list:
                c1, _, ri1, ins1, a1 = n1.split(":")
                c2, _, ri2, ins2, a2 = n2.split(":")
                l = f"{c1}_{ri1}{ins1}_{a1}-{c2}_{ri2}{ins2}_{a2}"
                l = f"{prefix}-{l}" if prefix else l
                l = l.replace("'", "_")  # ' is not allowed in pymol selection name
                cmd.distance(
                    name=l,
                    selection1=f"{obj_name} and (c. {c1} and i. {ri1}{ins1} and n. {a1})",
                    selection2=f"{obj_name} and (c. {c2} and i. {ri2}{ins2} and n. {a2})",
                )
                dist_sel.append(l)
            return dist_sel

        if results["hbond_atom_pairs"]:
            dist_sel = create_dist_selection_str(
                atom_pair_list=results["hbond_atom_pairs"],
                obj_name=selection,
                prefix=f"{sel_name_1_2}-HBond"
            )
            # pymol selection
            cmd.delete(f"{sel_name_1_2}-HBond-distance")
            cmd.group(f"{sel_name_1_2}-HBond-distance", " ".join(dist_sel))
        else:
            print_msg("No hydrogen bond atom pairs found between the chains", "info")
        # --------------------
        # Polar res pairs
        # --------------------
        # Polar contacts: create a group of residue pairs if sub_df_residue is not empty
        selections = create_selection_str(
            parent_df=sub_df_residue,
            interaction_type="Polar",
            obj_name=selection,
            prefix=sel_name_1_2
        )
        if selections:
            name = f"{sel_name_1_2}-Polar"
            cmd.delete(name=name)
            cmd.group(name=name, members=" ".join(selections))

        # --------------------
        # Polar distance
        # --------------------
        if results["polar_atom_pairs"]:
            dist_sel = create_dist_selection_str(
                atom_pair_list=results["polar_atom_pairs"],
                obj_name=selection,
                prefix=f"{sel_name_1_2}-Polar"
            )
            # group all distances
            n = f"{sel_name_1_2}-Polar-distance"
            cmd.delete(n)
            cmd.group(n, " ".join(dist_sel))

    # --------------------
    # Extra steps
    # --------------------
    # show interface-* as sticks
    cmd.show("sticks", f"interface-*")
    # set background to white
    cmd.bg_color("white")

    # --------------------
    # save output
    # --------------------
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
