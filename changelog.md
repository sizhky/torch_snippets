# Changelog

#### 0.499.28

- `show` can render a dataframe with a title
- `show` can accept a csv file as input (no need to load it and send)
- `backup_this_notebook` will back up as `backups/file/file__0000.html` instead of `backups/file/0000.html` for easier sharability
- module loads `decorators` by default (`io`, `timeit`, `check_kwargs_not_none`)
- `ishow` is less opinionated
- `shutdown_this_notebook` is a new function
