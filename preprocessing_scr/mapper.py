from __future__ import print_function
import pandas as pd 
import sys,os
import numpy as np 
#from scipy.stats import zscore


def expand(df,column,sep="|"):
    '''Expand rows containing separator.'''
    ndx = 0 
    expanded_df = {}
    for row in df.iterrows():
        if type(row[1][column]) == str:
            for x in row[1][column].split(sep):
                row_ = row[1].copy()
                row_[column] = x
                expanded_df[ndx] = row_
                ndx +=1
        else:
            expanded_df[ndx] = row[1]
            ndx +=1
    return pd.DataFrame.from_dict(expanded_df).T


def parse_mapping_table(dataframe, query_id, target_id):
    ''' Takes the dataframe with gene ids mapping and column names of wuery and target ID.
    \nReturns a dictionary with one-to-one, one-to-many, many-to-one mappings and a lsit with unmapped identitfiers (one-to-none). Tested with Homo_sapiens.gene_info from NCBI.'''
    
    mapper = {"one-to-one":{},"one-to-many":{},"many-to-one":{},"one-to-none":[]}
    
    df = dataframe.loc[:,[query_id, target_id]]
    # exclude rows of all NAs
    df_size = df.shape[0]
    df.dropna(how="all",inplace=True)
    if df_size - df.shape[0] > 0:
        print(df_size - df.shape[0],"rows with both",query_id,"and",target_id,"empty",file=sys.stderr)
    else:
        print("Ok: no empty rows detected",file=sys.stderr)
    # duplicated pairs
    df_size = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df_size - df.shape[0] > 0:
        print(df_size - df.shape[0],"duplicated pairs dropped",file=sys.stderr)
    else:
        print("Ok: no duplicated pairs detected",file=sys.stderr)
    # exclude NA query IDs
    df_size = df.shape[0]
    df.dropna(subset=[query_id],axis=0,inplace = True)
    if df_size -df.shape[0] >0:
        print(df_size -df.shape[0],"rows with empty",query_id,"were excluded",file=sys.stderr)
    else:
        print("Ok: All",query_id,"rows are not empty.",file=sys.stderr)
        
    # recognized query ids mapped to no target ids
    found_not_mapped = list(set(df.loc[df[target_id].isnull(),query_id].values))
    if len(found_not_mapped) > 0:
        df = df.loc[~df[query_id].isin(found_not_mapped),:].copy()
        print(len(found_not_mapped),query_id,"ids mapped to no",target_id,file= sys.stderr)
        
    else:
        print("Ok: All",query_id,"are mapped to",target_id,file= sys.stderr)
    
    # uniqueness of query ids; one-to-many is not acceptable
    query_dups = list(set(df.loc[df.duplicated(subset=[query_id],keep = False),:][query_id].values))
    if len(query_dups) > 0:
        print(len(query_dups),query_id,"mapped to multiple",target_id,file= sys.stderr)
        df_one_to_many =  df.loc[df[query_id].isin(query_dups),:].copy()
        df = df.loc[~df[query_id].isin(query_dups),:].copy()
        df_one_to_many = df_one_to_many.groupby(query_id).agg({target_id:list})
        mapper["one-to-many"] = df_one_to_many.to_dict()[target_id]
    else:
        print("Ok: All",query_id,"are unique",file= sys.stderr)
    
    # uniqueness of target ids; many-to-one is ok for synonyms, but not for primary id
    query_ambiguous = list(set(df.loc[df.duplicated(subset=[target_id],keep = False),:][query_id].values))
    if len(query_ambiguous) > 0:
        print(len(query_ambiguous),"different",query_id,
              "mapped to the same",target_id,file= sys.stderr)
        df_many_to_one =  df.loc[df[query_id].isin(query_ambiguous),:].copy()
        df = df.loc[~df[query_id].isin(query_ambiguous),:].copy()
        df_many_to_one.set_index(query_id,inplace=True,drop=True)
        mapper["many-to-one"] = df_many_to_one.to_dict()[target_id]
    else:
        print("Ok: All",target_id,"are unique",file= sys.stderr)
        
    if len(query_dups) == 0 and len(query_ambiguous) == 0:
        print("Ok: One-to-one mapping between",query_id,"and",target_id,file= sys.stderr)
    print(df.shape[0],query_id,"can be mapped directly to",target_id,file= sys.stderr)
    
    # one-to-one
    df.set_index(query_id,inplace=True,drop=True)
    mapper["one-to-one"]=df.to_dict()[target_id]
    
    # query_id without target_id
    mapper["one-to-none"]= found_not_mapped
    return  mapper


def apply_mappers(df, main_mapper, alt_mapper, verbose = True,handle_duplicates = "keep"):
    '''Converts IDs in DF indices.\n
    handle_duplicates  - how to deal with duplicated IDs in the resulted DF:\n
    \tsum - group by index and sum\n
    \taverage - group by index and keep average\n
    \tdrop - drop duplicates\n
    \tkeep - do nothing.'''
    ID_list = list(df.index.values)
    
    # main mapper, e.g. NCBI symbol -> Entrez Gene ID
    symbols_mapped_directly = {}
    recognized_not_mapped = [] # found in target IDs of mapper but not 
    symbol_one2many = [] # not mapped because of ambiguity
    symbol_many2one = [] # not mapped because of ambiguity
    # Alternative mapper
    # applied in case the main mapper failed: e.g. NCBI synonym -> NCBI symbol -> Entrez Gene ID
    via_alt_symbol = {}
    via_nonuniq_alt_symbol = {}
    alt_symbol_one2many = []  # 
    synonym_match_current_symbol = [] # these synonyms are not used in mapping because they match with ID in main mapped
    not_found_at_all =[]
    loc = {}
    loc_not_found =[]
    # store all valid target IDs
    valid_target_ids = main_mapper["one-to-one"].values()+ main_mapper["many-to-one"].values() + alt_mapper["one-to-one"].values() + alt_mapper["many-to-one"].values()
    for l in main_mapper["one-to-many"].values() +alt_mapper["one-to-many"].values():
        valid_target_ids += l
        
       
    for symbol in ID_list:
        if symbol in main_mapper["one-to-one"].keys():
            symbols_mapped_directly[symbol] = main_mapper["one-to-one"][symbol]
        elif  symbol in main_mapper["one-to-none"]:
            recognized_not_mapped.append(symbol)
        elif symbol in main_mapper["one-to-many"].keys():
            symbol_one2many.append(symbol)
        elif symbol in main_mapper["many-to-one"].keys():
            symbol_many2one.append(symbol)
        # alternative mappper
        elif symbol in alt_mapper["one-to-one"].keys():
            via_alt_symbol[symbol] = alt_mapper["one-to-one"][symbol]
        elif symbol in alt_mapper["one-to-many"].keys():
            alt_symbol_one2many.append(symbol)
        elif symbol in alt_mapper["many-to-one"].keys(): # it is Ok if many synonyms match 
            via_nonuniq_alt_symbol[symbol] = alt_mapper["many-to-one"][symbol]
        elif symbol.startswith("LOC"):
            LOC_id = int(symbol[3:])
            if LOC_id in valid_target_ids:
                loc[symbol] = LOC_id
            else:
                loc_not_found.append(symbol)
        else:
            not_found_at_all.append(symbol)
        
    query2target ={}
    for symbol in [symbols_mapped_directly,via_alt_symbol,via_nonuniq_alt_symbol,loc]:
        query2target.update(symbol)
    not_mapped = recognized_not_mapped +symbol_one2many+ alt_symbol_one2many + loc_not_found + not_found_at_all+ symbol_many2one
    
    if verbose:
        print("Mapped:",len(query2target.keys()), 
      "\n\tdirectly via main_mapper",len(symbols_mapped_directly.keys()),
     "\n\tvia alternative mapper",len(via_alt_symbol.keys()),
      "\n\tvia one of multiple synonyms in alternative mapper",len(via_nonuniq_alt_symbol.keys()),
      "\n\tLOC",len(loc.keys()),
      "\nUnmapped:",len(not_mapped),
      "\n\trecognized symbols without Entrez ID",len(recognized_not_mapped),
      "\n\tmultiple query_ids map to the same target_id",len(symbol_many2one),
      "\n\tquery_ids map to multiple target_ids in the main mapper",len(symbol_one2many),
      "\n\tquery_ids map to multiple target_ids in the alternative mapper",len(alt_symbol_one2many),
      "\n\tLOC not found in Entrez",len(loc_not_found),
     "\n\tNot found at all:",len( not_found_at_all))
    
    # find duplicated 
    mapped_symbols = pd.Series(query2target)
    dups = mapped_symbols[mapped_symbols.duplicated(keep=False)].index.values
    if len(dups) >0:
        print("Warning: query IDs mapping to duplicated target IDs in mapping table:", len(dups))
        #if verbose:
        #    print("IDs mapped to multiple target IDs:\n", dups,file=sys.stderr)
    
    # exclude not mapped query IDs and map
    df_size_dif = df.shape[0]
    df = df.loc[~df.index.isin(not_mapped ),:].copy()
    df_size_dif = df_size_dif - df.shape[0]
    if df_size_dif > 0:
        print("Warning: query IDs not mapped to any target IDs excluded:", df_size_dif)
    df.rename(mapper=query2target, axis='index',inplace=True)
    

    # sum genes genes (sum of duplicated Entrez IDs)
    if handle_duplicates == "keep":
        if verbose:
            dups = df.groupby(df.index).size()
            dups = list(set(dups[dups>1].index.values))
            print("IDs mapped to multiple target IDs are kept:\n", dups, file=sys.stderr)
    elif handle_duplicates == "sum":
        df = df.groupby(df.index).apply(sum)
    elif handle_duplicates == "average":
        df = df.groupby(df.index).apply(np.average)
    elif handle_duplicates == "drop":
        df = df.loc[~dups,:].copy()
    else:
        print("'handle_duplicates' must be keep, sum, average or drop.", file =sys.stderr)
        return None
    df.sort_index(inplace=True)
    return (df,query2target,not_mapped)
    