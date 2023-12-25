# Changelog

#### 0.514

🐞 minor change in `print_shapes_hook`

#### 0.513

🎉 print_module_io_for automates attaching and detaching hooks
🎉 AD2 avoids rich printing

#### 0.512

🐞 `attach_hooks` will accept any custom hook

#### 0.511

🎉 Make `markup2.AD.__repr__` the summary
🎉 Expose `markup2.AD` as `AD2`
🎉 Make `icecream` a requirement
🎉 Min python is 3.8

#### 0.510

🎉 New IO hooks system in `torch_snippets.trainer.hooks`
🎉 Updated markup2.AD.summary and add print_summary methods

#### 0.509

🎉 Experimental `AD` in torch_snippets.markup2 that infers variable names E.g. - `(p=10; AD(p) == {'p': 10})`
🐞 `isin` will not add +1 (useful for both absolute and relative boxes now)
🐞 `write_json` will support numpy, torch and AttrDict

#### 0.508

🎉 add `find_address` to `AttrDict` that can return all path locations for a specific key
🎉 add `summary` to `AttrDict` that can give an outline of the dictionary
🎉 add `write_summary` to `AttrDict` that writes the summary to a textfile
🎉 `show` can now show bb colors `{"r": (255, 0, 0), "g": (0, 255, 0), "b": (0, 0, 255), "y": (255, 0, 255)}` if `df` has column called `color`
🎉 `AD` is an alias for `AttrDict`
🎉 `AD` can directly consume kwargs

#### 0.507

🧹 import only important functins from `dates.py`
🎉 add `backup_all_notebooks` that backs up every notebook present in a specific folder
🎉 `reset_logger` can disable stdout logging if needed, using `disable_stdout=True` kwarg (False by default)
`common_items` will take a list of folders and return common stems from the folders
images will show a black border when grid is True

#### 0.506

Info, Debug, Warn and Excep will format ouputs separated by a `;` when args are passed
`notify_waiting` is a new function that letting you know some process is running for an unknown amount of time
optional `delay` during `shutdown_current_notebook`
Info, Debug, Warn and Excep will all have `X_mode` and `in_X_mode` functions much like in_debug_mode and debug_mode
`__init__` will auto pull from logger now
Better non-linear `Timer` (and `Report` and `track2`)

#### 0.505

🧹 `Info`, `Debug`, `Warn` and `Excep` will accept args (instead of a single arg)
🧹 `show` will show h4 headers instead of h2 for dataframe titles

#### 0.504

🧹 `phasify` loads by default
🧹 `show_big_dataframe` can show more rows
🎉 add a new submodule `trainer.hooks`
🧹 `show` delegated kwargs to `plt.imshow` for a better readme
🎉 `batchify` can batchify multiple containers at once
🎉 `cat_with_padding` new function in `torch_loader`
🧹 `L` is json compatible
🐞 `BB` will not decide if something is relative/absolute
🎉 `__contains__` for config
🎉 `to` works on `AttrDict`
👶🏼 `track2` is a better version of `track` uses corouties
👶🏼 `debug_mode` temporarily activates `DEBUG` mode on
👶🏼 `if in_debug_mode():` lets you know if `DEBUG` mode is on
🧹 `reset_logger` can accept lowercase levels also
🧹 `dumpdill` will return a Path after dumping

#### 0.503

bugfix in `loader.show`
add `today` function to dates
add `are_equal_dates` to dates
add dpi option to pdf

#### 0.502

bugfix in attrdict.map

#### 0.500

All notebooks are formatted with black
`parse` can parse python expressions
Add DeepLearningConfig class that can be used to load model hyperparameters
Add GenericConfig class that can be used to load generic (such as training, evaluation) hyperparameters
Add date utilities
`patch_to`, `Timer`, `timeit`, `io` are loaded by default
`lovely_tensors` is optional
Add phasify function to loader

#### 0.499.29

attrdict can deserialize "L"

#### 0.499.28

- `show` can render a dataframe with a title
- `show` can accept a csv file as input (no need to load it and send)
- `backup_this_notebook` will back up as `backups/file/file__0000.html` instead of `backups/file/0000.html` for easier sharability
- module loads `decorators` by default (`io`, `timeit`, `check_kwargs_not_none`)
- `ishow` is less opinionated
- `shutdown_this_notebook` is a new function

## Todo

override_previous_backup should not trigger when there's no backup to begin with
instead of showing markdown objects using display, directly show HTML objects so that the text is preserved on reopen h2 in Backup instead of h1
