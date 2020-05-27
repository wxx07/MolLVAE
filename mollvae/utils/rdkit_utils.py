# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:53:50 2020

@author: Olive


"""

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors, Crippen, QED
from rdkit.Chem import rdFMCS as MCS

from multiprocessing import Pool

    
def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')


def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')


## SA_Score import
from rdkit.Chem import RDConfig
import os,sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

## NP_Score import
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
import npscorer
fscore = npscorer.readNPModel()



def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    elif isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate() # if exception raised, terminate all process before exit
            return result

        return _mapper
    else:
        return n_jobs.map
    


def get_mol(smiles):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles, str):
        if len(smiles) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    else:
        return None

def get_canon_smiles(smiles, isomericSmiles=True):
    
    m = get_mol(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=isomericSmiles) if m is not None else "invalid"


def get_WLQSN(rdkit_mols_list, alongwithmols=False):
    """ get weight, logp, qed, sas, nps for a list of rdkit molecules
    
    Deprecated. See `get_molds_props`
    """
    
    
    mols = rdkit_mols_list
    
    if not alongwithmols:
        ##? mod
        logps = []
        logp_error = []
        for idx,m in enumerate(mols):
            try:
                logps.append(Crippen.MolLogP(m))
            except ValueError:
                logp_error.append(idx)
                continue
        print("logp_error:{}".format(len(logp_error)))
        mols = [m for idx,m in enumerate(mols) if idx not in logp_error]            # remove unknown sanitization cases
        
        ## get mols' properties
        wts = [Descriptors.ExactMolWt(mol) for mol in mols]    
        qeds = [QED.qed(mol) for mol in mols]
        sass = [sascorer.calculateScore(mol) for mol in mols]
        npss = [npscorer.scoreMol(mol, fscore) for mol in mols]
    else:
        logps = []
        logp_error = []
        for idx,m in enumerate(mols):
            try:
                logps.append((Crippen.MolLogP(m),m))
            except ValueError:
                logp_error.append(idx)
                continue
        print("logp_error:{}".format(len(logp_error)))
        mols = [m for idx,m in enumerate(mols) if idx not in logp_error]            # remove unknown sanitization cases
        
        ## get mols' properties
        wts = [(Descriptors.ExactMolWt(mol),mol) for mol in mols]    
        qeds = [(QED.qed(mol),mol) for mol in mols]
        sass = [(sascorer.calculateScore(mol),mol) for mol in mols]
        npss = [(npscorer.scoreMol(mol, fscore),mol) for mol in mols]        
    
    return wts, logps, qeds, sass, npss


################ descriptors ################
################ must remove salt first

def get_logp(rdmol):
    
    return float("inf") if rdmol is None else Descriptors.MolLogP(rdmol)

def get_mw(rdmol):
    
    return float("inf") if rdmol is None else Descriptors.MolWt(rdmol)
                                    
def get_qed(rdmol):
    
    return float("inf") if rdmol is None else QED.qed(rdmol)

def get_sas(rdmol):
    
    return float("inf") if rdmol is None else sascorer.calculateScore(rdmol)

################ descriptors ################
    



def get_molds_props(smiles_list, alongwithmols=False, n_cpu=10):
    """ Get molecular dataset properties
    
    Current implement includes: molecular weight(mw), QED, logP, SAS
    
    Args
        smiles_list [list(str)]: allow invalid smiles
        n_cpu [int, 10]: parallel cores
        
    Returns
        [dict]
    
    """
    
    mols = mapper(n_cpu)(get_mol, smiles_list)
    
    logps = mapper(n_cpu)(get_logp, mols)
    mws = mapper(n_cpu)(get_mw, mols)
    qeds = mapper(n_cpu)(get_qed, mols)
    sass = mapper(n_cpu)(get_sas, mols)
    
    return {"logP":logps,"MW":mws,"QED":qeds,"SAS":sass}



def mcs_1molset(rdmol_list, dont_return_empty=False, seedSmarts=''):
    
    res = MCS.FindMCS(rdmol_list, completeRingsOnly=True, seedSmarts=seedSmarts)
    smarts = res.smartsString
    assert isinstance(smarts, str) and not res.canceled, "mcs_1molset error"
    if not dont_return_empty:
        return smarts
    elif len(smarts) > 0:
        return smarts

    
    
    
    
    
    
    
    
    
    