# Changelog

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
