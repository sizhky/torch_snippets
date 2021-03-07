# Utilities for simple needs



## torch snippets does a lot of default importing for you
Whether it is numpy, pandas, matplotlib or the useful functions that are mentioned below
Simply call 
```python
from torch_snippets import *
```
All the imports are lightweight and thus should not take more than a couple of seconds

## Auxiliary Functions
There are simple functions that are overloaded to take inputs and perform repetitive tasks that usually take a few lines to write

#### Images
`show`, `inspect`, `Glob`, `read`, `resize`, `rotate`

#### Files and Paths
`stem`, `Glob`, `parent`, `name`, `fname`,


`makedir`, `zip_files`, `unzip_file`,   


`find`, `extn`,  


`readlines`, `writelines`

#### Lists
`L`, `flatten`

#### Dump and load python objects
`loaddill`,`dumpdill`

#### Misc 
`Tqdm`, `Timer`, `randint`, `Logger`

#### Sets
`unique`, `diff`, `choose`, `common`

#### Pytorch Modules
`Reshape` and `Permute` (`nn.Modules`)

#### Report as Pytorch Lightning Callback
`LightningReport`

#### Charts
`Chart` from altair

and many more to come... 

## Install
`pip install torch_snippets`

## Usage


```python
%time from torch_snippets import *
```

    CPU times: user 1.79 s, sys: 672 ms, total: 2.46 s
    Wall time: 2.62 s


```python
dir()
```




    ['AttrDict',
     'B',
     'BB',
     'Blank',
     'C',
     'Chart',
     'DataLoader',
     'Dataset',
     'Debug',
     'E',
     'Excep',
     'F',
     'Float',
     'Glob',
     'Image',
     'ImportEnum',
     'In',
     'Inf',
     'Info',
     'Int',
     'L',
     'LightningReport',
     'NullType',
     'Out',
     'P',
     'PIL',
     'Path',
     'Permute',
     'PrettyString',
     'Report',
     'Reshape',
     'Self',
     'ShowPrint',
     'Stateful',
     'Str',
     'StrEnum',
     'T',
     'Timer',
     'Tqdm',
     'Warn',
     '_',
     '__',
     '___',
     '__builtin__',
     '__builtins__',
     '__doc__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     '_dh',
     '_i',
     '_i1',
     '_i2',
     '_ih',
     '_ii',
     '_iii',
     '_oh',
     'add',
     'add_props',
     'alt',
     'anno_ret',
     'annotations',
     'arg0',
     'arg1',
     'arg2',
     'arg3',
     'arg4',
     'argnames',
     'argwhere',
     'attrdict',
     'basic_repr',
     'bbfy',
     'bind',
     'camel2snake',
     'charts',
     'choose',
     'chunked',
     'class2attr',
     'common',
     'compose',
     'copy_func',
     'crop_from_bb',
     'custom_dir',
     'cv2',
     'cycle',
     'defaults',
     'detuplify',
     'device',
     'df2bbs',
     'diff',
     'display',
     'dumpdill',
     'enlarge_bbs',
     'eq',
     'even_mults',
     'exec_local',
     'exit',
     'extn',
     'fastcores',
     'fastuple',
     'filter_dict',
     'filter_ex',
     'filter_keys',
     'filter_values',
     'find',
     'first',
     'flatten',
     'fname',
     'fname2',
     'ge',
     'gen',
     'get_class',
     'get_ipython',
     'getattrs',
     'glob',
     'groupby',
     'gt',
     'hasattrs',
     'ifnone',
     'ignore_exceptions',
     'in_',
     'inspect',
     'instantiate',
     'inum_methods',
     'is_',
     'is_array',
     'is_not',
     'isdir',
     'jitter',
     'last_index',
     'le',
     'line',
     'lines',
     'listify',
     'load_torch_model_weights_to',
     'loaddill',
     'loader',
     'logger',
     'lt',
     'lzip',
     'makedir',
     'map_ex',
     'maps',
     'maybe_attr',
     'md5',
     'merge',
     'mk_class',
     'mul',
     'ne',
     'nested_attr',
     'nested_idx',
     'nn',
     'not_',
     'now',
     'np',
     'null',
     'num_cpus',
     'num_methods',
     'nunique',
     'optim',
     'os',
     'otherwise',
     'pad',
     'parent',
     'partialler',
     'patch',
     'patch_property',
     'patch_to',
     'pd',
     'pdb',
     'pdfilter',
     'pl',
     'plt',
     'properties',
     'puttext',
     'quit',
     'rand',
     'randint',
     'range_of',
     're',
     'read',
     'readPIL',
     'readlines',
     'rect',
     'remove_duplicates',
     'rename_batch',
     'renumerate',
     'replicate',
     'resize',
     'risinstance',
     'rnum_methods',
     'rotate',
     'save_torch_model_weights_from',
     'see',
     'set_logging_level',
     'setattrs',
     'setify',
     'show',
     'shrink_bbs',
     'snake2camel',
     'sorted_ex',
     'stem',
     'stems',
     'stop',
     'store_attr',
     'str_enum',
     'sub',
     'subplots',
     'sys',
     'th',
     'to_absolute',
     'to_relative',
     'tonull',
     'torch',
     'torch_loader',
     'torchvision',
     'tqdm',
     'trange',
     'transforms',
     'true',
     'truediv',
     'try_attrs',
     'tuplify',
     'type_hints',
     'typed',
     'uint',
     'unique',
     'uniqueify',
     'unzip_file',
     'using_attr',
     'val2idx',
     'with_cast',
     'wrap_class',
     'write',
     'writelines',
     'xywh2xyXY',
     'zip_cycle',
     'zip_files']


